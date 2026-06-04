"""eval_lagged.py — Single-member full-state rollout for a lagged ensemble.

Runs one autoregressive rollout per invocation (one GPU, one initial date).
Different runs with different ``--start_date`` values form a lagged ensemble.

Saves the full predicted atmospheric state at every 24h step:
  * full horizontal resolution (180 x 360)
  * all model pressure levels (26 levels per upper-air variable)
  * every surface, multi-level, and diagnostic variable in the config

Output layout (one folder per ensemble member):

    {output_root}/{member_name}/
        config.yaml
        meta.yaml
        rollout_checkpoint.pt                    # rollout state for resume (deleted on completion)
        surface/<var>/<var>_<YYYYMM>.nc          # (time, lat, lon)
        multilevel/<var>/<var>_<YYYYMM>.nc       # (time, plev, lat, lon)
        diagnostic/<var>/<var>_<YYYYMM>.nc       # (time, lat, lon)

Per-month NetCDFs keep the in-memory buffer to ~1.2 GB and let the rollout
resume cleanly from month boundaries (the buffer is empty at every
checkpoint, so only the model state + RNG need to be saved).

Resume: if ``rollout_checkpoint.pt`` exists in the member directory, the
script picks up where it left off — restoring the rollout state and the
RNG state of every relevant generator (torch CPU, the active CUDA device,
numpy, and the python ``random`` module) so the diffusion sampler produces
the same noise sequence it would have on an uninterrupted run. Use
``--force_restart`` to discard a checkpoint and start over.

Run examples:

    # First run
    python eval_lagged.py --config=configs/combined_NCAR.yaml \\
        --start_date=1979-01-01 --member_name=member_19790101

    # Resume after walltime kill (same command — auto-resumes)
    python eval_lagged.py --config=configs/combined_NCAR.yaml \\
        --start_date=1979-01-01 --member_name=member_19790101
"""

import argparse
import os
import random
import time
from collections import deque
from datetime import datetime, timedelta
from os.path import join

import cftime
import numpy as np
import torch
import xarray as xr
from lightning.pytorch import seed_everything

import torch_harmonics as th

from common.utils import (
    assemble_forcing,
    disassemble_input,
    get_yaml,
    save_yaml,
)
from data.amip_new import GetDataset
from modules.combined_module import CombinedModule
from modules.train_module import TrainModule


CHECKPOINT_FILENAME = "rollout_checkpoint.pt"


# ---------------------------------------------------------------------------
# Spectral monitoring (2m_temperature)
# ---------------------------------------------------------------------------
#
# Compares the spherical-harmonic power spectrum of the generated 2m_temperature
# field against an ERA5 reference taken from ``spectral_lag_days`` days earlier
# (the spectrum of 2m_T is assumed to drift only slowly over a 10-day window).
# When the per-degree relative spectral RMSE exceeds a threshold the rollout is
# restarted from the ERA5 state ``spectral_lag_days`` days back — the
# stochastic sampler may produce a stable trajectory on retry.


class _SphericalSpectrum:
    """Caches a RealSHT and returns the total power at each spherical-harmonic
    degree l: P(l) = sum_m w_m * |a_{l,m}|^2, with w_m = 1 for m=0 and 2 for
    m>0 (real-field convention). Output is on CPU as a (lmax,) tensor.
    """

    def __init__(self, nlat: int, nlon: int, device, grid: str = "equiangular"):
        self.sht = th.RealSHT(nlat, nlon, grid=grid).float().to(device)
        m_w = torch.ones(self.sht.mmax, device=device)
        m_w[1:] = 2.0
        self.m_w = m_w  # (mmax,)

    def __call__(self, field_2d: torch.Tensor) -> torch.Tensor:
        f = field_2d.to(self.m_w.device, dtype=torch.float32).unsqueeze(0)
        coeffs = self.sht(f).squeeze(0)  # (lmax, mmax), complex
        power = (coeffs.real ** 2 + coeffs.imag ** 2) * self.m_w.unsqueeze(0)
        return power.sum(dim=-1).detach().cpu()  # (lmax,)


def normalized_spectral_rmse(pred_spec: torch.Tensor,
                             ref_spec: torch.Tensor,
                             eps: float = 1e-30) -> float:
    """Per-degree relative RMSE between two spherical-harmonic power spectra.

    Skips l=0 (the global mean) so the metric reflects drift in the
    variability spectrum rather than the global mean.
    """
    rel = (pred_spec - ref_spec) / (ref_spec + eps)
    return float(torch.sqrt(torch.mean(rel[1:] ** 2)).item())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_lat_lon(nlat: int, nlon: int):
    """Cell-centered ERA5-style 1-deg grid: lat 89.5..-89.5, lon 0.5..359.5."""
    dlat = 180.0 / nlat
    dlon = 360.0 / nlon
    lat = np.arange(90.0 - 0.5 * dlat, -90.0, -dlat, dtype=np.float64)
    lon = np.arange(0.5 * dlon, 360.0, dlon, dtype=np.float64)
    assert len(lat) == nlat and len(lon) == nlon
    return lat, lon


def daterange_days(start, end):
    """List of cftime datetimes from ``start`` (inclusive) to ``end`` (inclusive), step 1 day."""
    out = []
    t = start
    while t <= end:
        out.append(t)
        t = t + timedelta(days=1)
    return out


def parse_date(s: str):
    """Parse YYYY-MM-DD into a cftime.DatetimeGregorian (00:00 UTC)."""
    d = datetime.strptime(s, "%Y-%m-%d")
    return cftime.DatetimeGregorian(d.year, d.month, d.day, 0, 0, 0,
                                    has_year_zero=False)


def is_last_day_of_month(t) -> bool:
    """True iff ``t + 1 day`` is in a different month."""
    nxt = t + timedelta(days=1)
    return (nxt.year, nxt.month) != (t.year, t.month)


# ---------------------------------------------------------------------------
# Manual data loading (bypass dataset.__getitem__ so we can drive the rollout
# from any in-range date without lead-time padding restrictions).
# ---------------------------------------------------------------------------

def load_state(dataset, t):
    """Load full atmospheric state at time ``t`` (normalized).

    Returns:
        surface (1, c_sfc, nlat, nlon)
        upper_air (1, c_ua, nlev, nlat, nlon)
        diagnostic (1, c_diag, nlat, nlon)
    """
    raw = dataset._get_data(t, out=False)
    upper_air, surface, diagnostic, _ = dataset._reshape_and_mask_variables(raw, out=False)

    surface = dataset.surface_transform(surface).unsqueeze(0)
    upper_air = dataset.upper_air_transform(upper_air).unsqueeze(0)
    diagnostic = dataset.diagnostic_transform(diagnostic).unsqueeze(0)
    return surface, upper_air, diagnostic


def load_forcing(dataset, t):
    """Load varying boundary forcing at time ``t``.

    Returns:
        boundary (1, c_bnd, nlat, nlon) — normalized, with CO2 stripped if
            ``return_calendar`` is True.
        calendar (1, 3) or None — [seconds_of_day, day_of_year, co2_norm].
    """
    raw = torch.tensor(
        dataset._get_data(t, variable_list=dataset.varying_boundary_variables)
    ).to(torch.float32)
    raw = dataset._fill_mask(raw, dataset.varying_boundary_variables)
    boundary = dataset.boundary_transform(raw)

    calendar = None
    if dataset.return_calendar:
        co2 = boundary[0, 0, 0].clone()
        boundary = boundary[1:]
        sod, doy = dataset._compute_calendar(t)
        calendar = torch.tensor([[sod, doy, co2.item()]], dtype=torch.float32)

    return boundary.unsqueeze(0), calendar


# ---------------------------------------------------------------------------
# Per-month writer: buffer one month of daily snapshots, then flush as NetCDF.
# ---------------------------------------------------------------------------

class MonthBuffer:
    """In-memory buffer for one month's worth of daily snapshots."""

    def __init__(self, year: int, month: int, surf_vars, multi_vars, diag_vars,
                 nlat: int, nlon: int, nlevels: int):
        self.year = year
        self.month = month
        self.surf_vars = list(surf_vars)
        self.multi_vars = list(multi_vars)
        self.diag_vars = list(diag_vars)
        self.nlat = nlat
        self.nlon = nlon
        self.nlevels = nlevels

        self.times = []
        self.surf = {v: [] for v in self.surf_vars}
        self.multi = {v: [] for v in self.multi_vars}
        self.diag = {v: [] for v in self.diag_vars}

    def add(self, t, surf_np, multi_np, diag_np):
        """Add one daily snapshot.

        surf_np:  (c_sfc, nlat, nlon)
        multi_np: (c_ua, nlevels, nlat, nlon)
        diag_np:  (c_diag, nlat, nlon)
        """
        self.times.append(t)
        for c, name in enumerate(self.surf_vars):
            self.surf[name].append(surf_np[c].astype(np.float32, copy=False))
        for c, name in enumerate(self.multi_vars):
            self.multi[name].append(multi_np[c].astype(np.float32, copy=False))
        for c, name in enumerate(self.diag_vars):
            self.diag[name].append(diag_np[c].astype(np.float32, copy=False))

    def stack(self):
        surf = {k: np.stack(v, axis=0) for k, v in self.surf.items() if v}
        multi = {k: np.stack(v, axis=0) for k, v in self.multi.items() if v}
        diag = {k: np.stack(v, axis=0) for k, v in self.diag.items() if v}
        return surf, multi, diag


def write_month(buf: MonthBuffer, out_dir: str,
                lat: np.ndarray, lon: np.ndarray, levels_hpa,
                member_name: str):
    """Flush a MonthBuffer to per-variable NetCDFs."""
    if not buf.times:
        return

    surf, multi, diag = buf.stack()
    times = np.asarray(buf.times)
    tag = f"{buf.year:04d}{buf.month:02d}"

    common_attrs = {
        "Conventions": "CF-1.8",
        "source": "amip lagged-ensemble rollout",
        "variant_label": member_name,
        "frequency": "day",
        "grid_label": "gn",
    }
    time_encoding = {
        "dtype": "float64",
        "units": "days since 1850-01-01 00:00:00",
        "calendar": "standard",
    }

    # 2-D fields: (time, lat, lon)
    for group_name, group in (("surface", surf), ("diagnostic", diag)):
        for var, arr in group.items():
            ds = xr.Dataset(
                {var: (("time", "lat", "lon"), arr)},
                coords={"time": times, "lat": lat, "lon": lon},
                attrs=common_attrs,
            )
            ds["lat"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
            ds["lon"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")
            ds[var].attrs.update(long_name=var)
            path = join(out_dir, group_name, var, f"{var}_{tag}.nc")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            encoding = {
                var: {"compression": "gzip", "compression_opts": 4, "dtype": "float32"},
                "time": time_encoding,
            }
            # Overwrite any partial file from a previous interrupted run.
            if os.path.exists(path):
                os.remove(path)
            ds.to_netcdf(path, format="NETCDF4", engine="h5netcdf",
                         encoding=encoding, unlimited_dims=["time"])

    # 3-D fields: (time, plev, lat, lon)
    plev_pa = np.asarray(levels_hpa, dtype=np.float64) * 100.0
    for var, arr in multi.items():
        ds = xr.Dataset(
            {var: (("time", "plev", "lat", "lon"), arr)},
            coords={"time": times, "plev": plev_pa, "lat": lat, "lon": lon},
            attrs=common_attrs,
        )
        ds["lat"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
        ds["lon"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")
        ds["plev"].attrs.update(units="Pa", standard_name="air_pressure",
                                long_name="pressure", axis="Z", positive="down")
        ds[var].attrs.update(long_name=var)
        path = join(out_dir, "multilevel", var, f"{var}_{tag}.nc")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        encoding = {
            var: {"compression": "gzip", "compression_opts": 4, "dtype": "float32"},
            "time": time_encoding,
        }
        if os.path.exists(path):
            os.remove(path)
        ds.to_netcdf(path, format="NETCDF4", engine="h5netcdf",
                     encoding=encoding, unlimited_dims=["time"])


# ---------------------------------------------------------------------------
# Rollout checkpoint (resume after walltime kill).
# ---------------------------------------------------------------------------
#
# Saved at the end of each month, when the month buffer is empty (just flushed
# to disk). Resume restores ``x`` and the RNG state of every relevant
# generator so the diffusion sampler produces the exact same noise sequence
# it would have on an uninterrupted run.

def _gather_rng_state(device):
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state(device)
                       if torch.cuda.is_available() else None),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng_state(rng, device):
    torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(rng["torch_cuda"], device=device)
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


def save_rollout_checkpoint(out_dir: str, x: torch.Tensor, t_idx: int, device):
    """Atomically save the rollout state for resume."""
    payload = {
        "x": x.detach().cpu(),
        "t_idx": int(t_idx),
        "rng": _gather_rng_state(device),
        "format_version": 1,
    }
    path = join(out_dir, CHECKPOINT_FILENAME)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_rollout_checkpoint(out_dir: str, device):
    """Load + restore RNG state. Returns dict {x, t_idx} or None."""
    path = join(out_dir, CHECKPOINT_FILENAME)
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _restore_rng_state(payload["rng"], device)
    return {"x": payload["x"].to(device), "t_idx": payload["t_idx"]}


def remove_rollout_checkpoint(out_dir: str):
    path = join(out_dir, CHECKPOINT_FILENAME)
    if os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def build_model(config, dataset, checkpoint, downscaler_checkpoint, device):
    is_combined = config["model"].get("model_name", "") == "Combined"
    if is_combined:
        if checkpoint:
            config["training"]["forecaster_checkpoint"] = checkpoint
        if downscaler_checkpoint:
            config["training"]["downscaler_checkpoint"] = downscaler_checkpoint
        model = CombinedModule(config, normalizer=dataset).to(device)
    else:
        model = TrainModule(config, normalizer=dataset).to(device)
        if checkpoint:
            sd = torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"]
            model.load_state_dict(sd)
    model.eval()
    return model, is_combined


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_one(
    model,
    dataset,
    times,
    out_dir: str,
    member_name: str,
    lat_coord,
    lon_coord,
    device,
    force_restart: bool = False,
    spectral_monitor: bool = False,
    spectral_lag_days: int = 10,
    spectral_threshold: float = 1.0,
    spectral_max_restarts: int = 1000,
):
    """Run one autoregressive rollout. Resumes from checkpoint if present.

    If ``spectral_monitor`` is True, after each prediction the zonally averaged
    power spectrum of the generated 2m_temperature field is compared to the
    ERA5 spectrum from ``spectral_lag_days`` days earlier (assumed to be a good
    proxy for the current-day true spectrum). If the per-wavenumber relative
    spectral RMSE exceeds ``spectral_threshold`` the rollout rewinds
    ``spectral_lag_days`` days, reloads the ERA5 state at that earlier date,
    and continues — the stochastic sampler may settle into a stable trajectory
    on retry. ``spectral_max_restarts`` caps the number of rewinds.
    """
    nlat = model.nlat
    nlon = model.nlon
    n_levels = model.nlevels
    levels_hpa = list(np.array(dataset.levels).astype(int))
    n_steps = len(times) - 1

    invariant = model.invariant_input.to(device)

    # Resolve 2m_temperature channel index for spectral monitoring.
    t2m_index = None
    sht_power = None
    if spectral_monitor:
        try:
            t2m_index = list(model.surface_variables).index("2m_temperature")
        except ValueError:
            print("  [spectral_monitor] '2m_temperature' not in surface_variables — disabling monitor.",
                  flush=True)
            spectral_monitor = False
        else:
            sht_power = _SphericalSpectrum(nlat, nlon, device=device)

    # ERA5 reference 2m_T spectra cached by (year, month, day).
    ref_spectrum_cache: dict = {}

    # Adaptive spectral threshold: if more than 2 rewinds keep targeting the
    # same date, bump the threshold by 0.05 (up to 0.75) so the rollout can
    # squeeze past a sticky spot. Reset to the base value once the rollout has
    # advanced past the last rewind target by spectral_lag_days steps (the
    # whole pending window has refilled with predictions made past that date).
    base_threshold = spectral_threshold
    current_threshold = base_threshold
    last_rewind_step = None
    same_target_rewinds = 0
    MAX_THRESHOLD = 0.9
    THRESHOLD_BUMP = 0.05

    def reference_spectrum(t_ref):
        key = (t_ref.year, t_ref.month, t_ref.day)
        cached = ref_spectrum_cache.get(key)
        if cached is not None:
            return cached
        try:
            surface_t, _, _ = load_state(dataset, t_ref)
        except (OSError, FileNotFoundError):
            return None
        surface_denorm = dataset.surface_inv_transform(surface_t.to(device))
        t2m = surface_denorm[0, t2m_index]
        spec = sht_power(t2m)
        ref_spectrum_cache[key] = spec
        return spec

    def fresh_buffer(year, month):
        return MonthBuffer(
            year, month,
            model.surface_variables,
            model.multilevel_variables,
            model.diagnostic_variables,
            nlat, nlon, n_levels,
        )

    def to_np(surf_t, multi_t, diag_t):
        surf_d = dataset.surface_inv_transform(surf_t)
        multi_d = dataset.upper_air_inv_transform(multi_t)
        diag_d = dataset.diagnostic_inv_transform(diag_t)
        return (
            surf_d[0].detach().cpu().numpy(),
            multi_d[0].detach().cpu().numpy(),
            diag_d[0].detach().cpu().numpy(),
        )

    # Mutable buf holder so nested helpers can swap month buffers.
    buf_ref = [None]

    def commit_to_buf(entry):
        """Commit one pending entry into the month buffer; flush at month-end."""
        _, t_out, surf_np, multi_np, diag_np = entry
        buf = buf_ref[0]
        if (t_out.year, t_out.month) != (buf.year, buf.month):
            if buf.times:
                write_month(buf, out_dir, lat_coord, lon_coord,
                            levels_hpa, member_name)
            buf = fresh_buffer(t_out.year, t_out.month)
        buf.add(t_out, surf_np, multi_np, diag_np)
        if is_last_day_of_month(t_out):
            print(f"  Flushing {buf.year:04d}-{buf.month:02d} "
                  f"({len(buf.times)} days) ...", flush=True)
            write_month(buf, out_dir, lat_coord, lon_coord,
                        levels_hpa, member_name)
            next_t = t_out + timedelta(days=1)
            buf = fresh_buffer(next_t.year, next_t.month)
        buf_ref[0] = buf

    # Pending queue of the most-recent predictions, deferred from buf to keep
    # the last lag_days+1 days reversible without touching disk. Without
    # spectral monitoring, capacity=1 (each step's prediction is committed
    # immediately, preserving original behavior).
    pending: deque = deque()
    pending_capacity = (spectral_lag_days + 1) if spectral_monitor else 1

    # ------------------------- Resume vs fresh start --------------------
    resumed = None if force_restart else load_rollout_checkpoint(out_dir, device)

    if resumed is not None:
        x = resumed["x"]
        last_done = resumed["t_idx"]
        start_step = last_done + 1
        if start_step >= n_steps:
            print(f"  Checkpoint indicates rollout already complete (last step {last_done}).")
            #remove_rollout_checkpoint(out_dir)
            return
        # The previous checkpoint was taken at end-of-month with pending drained.
        next_t_out = times[start_step + 1]
        buf_ref[0] = fresh_buffer(next_t_out.year, next_t_out.month)
        print(f"  Resuming from step {start_step}/{n_steps} "
              f"(next prediction: {next_t_out.isoformat()})", flush=True)
    else:
        # Fresh start: load IC, record it, build initial low-res rollout state.
        t0 = times[0]
        surface_t, upper_air_t, diagnostic_t = load_state(dataset, t0)
        surface_t = surface_t.to(device)
        upper_air_t = upper_air_t.to(device)
        diagnostic_t = diagnostic_t.to(device)

        buf_ref[0] = fresh_buffer(t0.year, t0.month)
        surf_np, multi_np, diag_np = to_np(surface_t, upper_air_t, diagnostic_t)
        buf_ref[0].add(t0, surf_np, multi_np, diag_np)

        x = model.preprocess(surface_t, upper_air_t, diagnostic_t)

        # Edge case: IC is itself the last day of its month — flush immediately.
        # Use t_idx = -1 to mean "no model steps run yet"; resume will set
        # start_step = 0 and re-init buf for the next month.
        if is_last_day_of_month(t0):
            write_month(buf_ref[0], out_dir, lat_coord, lon_coord,
                        levels_hpa, member_name)
            save_rollout_checkpoint(out_dir, x, -1, device)
            next_t = t0 + timedelta(days=1)
            buf_ref[0] = fresh_buffer(next_t.year, next_t.month)
        start_step = 0

    # ------------------------- Rollout loop -----------------------------
    log_every = max(1, n_steps // 200)
    start_clock = time.time()
    n_iterations = 0
    n_restarts = 0

    step_idx = start_step
    with torch.no_grad():
        while step_idx < n_steps:
            t_in = times[step_idx]
            t_out = times[step_idx + 1]

            try:
                boundary, calendar = load_forcing(dataset, t_in)
            except (OSError, FileNotFoundError) as e:
                print(f"  Forcing missing for {t_in.isoformat()}: {e} — stopping at step {step_idx}.",
                      flush=True)
                break
            boundary = boundary.to(device, non_blocking=True)
            calendar = calendar.to(device, non_blocking=True) if calendar is not None else None

            c_grid = assemble_forcing(boundary, invariant)

            fwd_kwargs = {"return_model_last": True}
            if calendar is not None:
                fwd_kwargs["c_scalar"] = calendar
            y, y_last = model.forward(x, c_grid, **fwd_kwargs)

            surf_pred, multi_pred, diag_pred = disassemble_input(y_last, nlevels=n_levels)
            surf_np, multi_np, diag_np = to_np(surf_pred, multi_pred, diag_pred)

            pending.append((step_idx, t_out, surf_np, multi_np, diag_np))
            while len(pending) > pending_capacity:
                commit_to_buf(pending.popleft())

            x = y  # autoregressive: feed the Euler-updated state forward
            n_iterations += 1

            # ---- Spectral monitor: trigger rewind on blowup ----
            # Only fires once pending has its full lag_days+1 entries — that
            # guarantees the rewind-target prediction is still in pending and
            # has not been committed to buf.
            restart_triggered = False
            if (spectral_monitor
                    and len(pending) >= spectral_lag_days + 1
                    and n_restarts < spectral_max_restarts):
                ref_t = times[step_idx + 1 - spectral_lag_days]
                ref_spec = reference_spectrum(ref_t)
                if ref_spec is not None:
                    pred_spec = sht_power(torch.from_numpy(surf_np[t2m_index]))
                    rmse = normalized_spectral_rmse(pred_spec, ref_spec)
                    if rmse > current_threshold:
                        n_restarts += 1
                        rewind_step = step_idx + 1 - spectral_lag_days
                        rewind_t = times[rewind_step]

                        if rewind_step == last_rewind_step:
                            same_target_rewinds += 1
                        else:
                            same_target_rewinds = 1
                            current_threshold = base_threshold
                            last_rewind_step = rewind_step

                        print(f"  [spectral_monitor] step {step_idx} t={t_out.isoformat()} "
                              f"2m_T spectral rmse={rmse:.3f} > {current_threshold:.3f} -- "
                              f"rewinding {spectral_lag_days} days to {rewind_t.isoformat()} "
                              f"(restart #{n_restarts}, same-target #{same_target_rewinds}).",
                              flush=True)

                        if same_target_rewinds > 1 and current_threshold < MAX_THRESHOLD:
                            new_threshold = min(current_threshold + THRESHOLD_BUMP, MAX_THRESHOLD)
                            print(f"  [spectral_monitor] {same_target_rewinds} rewinds at step "
                                  f"{rewind_step} ({rewind_t.isoformat()}) -- bumping threshold "
                                  f"{current_threshold:.3f} -> {new_threshold:.3f}.", flush=True)
                            current_threshold = new_threshold
                        # Discard all pending entries (lag_days+1 predictions
                        # including the rewind-target's stale prediction).
                        pending.clear()
                        # Reload ERA5 state at the rewind target and seed pending
                        # with the ground-truth snapshot so the rewind time still
                        # gets recorded in the output.
                        surface_t, upper_air_t, diagnostic_t = load_state(dataset, rewind_t)
                        surface_t = surface_t.to(device)
                        upper_air_t = upper_air_t.to(device)
                        diagnostic_t = diagnostic_t.to(device)
                        x = model.preprocess(surface_t, upper_air_t, diagnostic_t)
                        gt_surf_np, gt_multi_np, gt_diag_np = to_np(
                            surface_t, upper_air_t, diagnostic_t)
                        pending.append((rewind_step - 1, rewind_t,
                                        gt_surf_np, gt_multi_np, gt_diag_np))
                        step_idx = rewind_step
                        restart_triggered = True

            if restart_triggered:
                continue

            # ---- Reset bumped threshold once we've cleared the window ----
            # Once step_idx is lag_days past the last rewind target, the whole
            # pending window holds predictions made after that date, so the
            # sticky spot is behind us and we can drop back to the base
            # threshold.
            if (spectral_monitor
                    and last_rewind_step is not None
                    and step_idx >= last_rewind_step + spectral_lag_days):
                if current_threshold != base_threshold:
                    print(f"  [spectral_monitor] cleared prediction window past step "
                          f"{last_rewind_step} -- resetting threshold "
                          f"{current_threshold:.3f} -> {base_threshold:.3f}.", flush=True)
                current_threshold = base_threshold
                last_rewind_step = None
                same_target_rewinds = 0

            # ---- Checkpoint at month boundaries ----
            # When the current prediction's t_out is end-of-month, drain pending
            # into buf so the on-disk state is complete and a clean
            # (x, step_idx) checkpoint can be saved. Brief cost: the first
            # ``spectral_lag_days`` steps after resume cannot rewind, since
            # pending starts empty.
            if is_last_day_of_month(t_out):
                while pending:
                    commit_to_buf(pending.popleft())
                save_rollout_checkpoint(out_dir, x, step_idx, device)

            if n_iterations % log_every == 0 or step_idx == n_steps - 1:
                elapsed = time.time() - start_clock
                rate = n_iterations / max(elapsed, 1e-6)
                eta = (n_steps - step_idx - 1) / max(rate, 1e-6)
                restart_tag = f"  restarts={n_restarts}" if spectral_monitor else ""
                print(f"  step {step_idx+1}/{n_steps}  t={t_out.isoformat()}  "
                      f"{rate:.2f} steps/s  eta {eta/3600:.2f}h{restart_tag}",
                      flush=True)

            step_idx += 1

    # ------------------------- Finalize ---------------------------------
    # Drain any leftover pending entries (rollout ended mid-window or after
    # an early break above), then flush a possibly-partial trailing month.
    while pending:
        commit_to_buf(pending.popleft())
    buf = buf_ref[0]
    if buf is not None and buf.times:
        print(f"  Flushing trailing {buf.year:04d}-{buf.month:02d} "
              f"({len(buf.times)} days) ...", flush=True)
        write_month(buf, out_dir, lat_coord, lon_coord, levels_hpa, member_name)

    #llout_checkpoint(out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    config = get_yaml(args.config)
    dataconfig = config["data"]
    trainconfig = config["training"]

    if args.checkpoint is not None:
        trainconfig["checkpoint"] = args.checkpoint
        trainconfig["forecaster_checkpoint"] = args.checkpoint
    if args.downscaler_checkpoint is not None:
        trainconfig["downscaler_checkpoint"] = args.downscaler_checkpoint

    if len(args.devices) > 0:
        trainconfig["devices"] = [int(d) for d in args.devices]

    dataconfig["batch_size"] = 1
    dataconfig["epsilon_factor"] = 0
    dataconfig["forecast_lead_times"] = [1]

    seed = args.seed if args.seed is not None else trainconfig.get("seed", 43)
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"

    # ---- Time axis ----
    t_start = parse_date(args.start_date)
    t_end   = parse_date(args.end_date)
    if t_end <= t_start:
        raise SystemExit(f"--end_date ({args.end_date}) must be after --start_date ({args.start_date}).")
    times = daterange_days(t_start, t_end)
    print(f"Rollout: {t_start.isoformat()} -> {t_end.isoformat()} "
          f"({len(times)} dates, {len(times)-1} model steps)")

    # ---- Dataset spans the rollout window (year_end is exclusive). ----
    year_start = t_start.year
    year_end = t_end.year + 1
    dataset = GetDataset(dataconfig, year_start=year_start, year_end=year_end)

    model, _ = build_model(config, dataset, args.checkpoint,
                           args.downscaler_checkpoint, device)

    nlat = model.nlat
    nlon = model.nlon
    lat_coord, lon_coord = make_lat_lon(nlat, nlon)
    print(f"Pressure levels (hPa): {list(np.array(dataset.levels).astype(int))}")
    print(f"Surface vars:    {model.surface_variables}")
    print(f"Multilevel vars: {model.multilevel_variables}")
    print(f"Diagnostic vars: {model.diagnostic_variables}")

    # ---- Output directory: one folder per ensemble member ----
    member_name = args.member_name or f"member_{t_start.year:04d}{t_start.month:02d}{t_start.day:02d}"
    out_dir = join(args.output_root, member_name)
    os.makedirs(out_dir, exist_ok=True)
    save_yaml(config, join(out_dir, "config.yaml"))
    save_yaml(
        {
            "member_name": member_name,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "seed": seed,
            "checkpoint": trainconfig.get("forecaster_checkpoint", trainconfig.get("checkpoint")),
            "downscaler_checkpoint": trainconfig.get("downscaler_checkpoint"),
            "n_steps": len(times) - 1,
            "horizontal_resolution": [nlat, nlon],
            "levels_hpa": list(np.array(dataset.levels).astype(int)),
        },
        join(out_dir, "meta.yaml"),
    )
    print(f"Output directory: {out_dir}")
    if os.path.exists(join(out_dir, CHECKPOINT_FILENAME)) and not args.force_restart:
        print("  (rollout checkpoint detected — will resume)")

    rollout_one(
        model=model,
        dataset=dataset,
        times=times,
        out_dir=out_dir,
        member_name=member_name,
        lat_coord=lat_coord,
        lon_coord=lon_coord,
        device=device,
        force_restart=args.force_restart,
        spectral_monitor=args.spectral_monitor,
        spectral_lag_days=args.spectral_lag_days,
        spectral_threshold=args.spectral_threshold,
        spectral_max_restarts=args.spectral_max_restarts,
    )

    print(f"\nDone. Member '{member_name}' written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-member full-state rollout (lagged ensemble).")
    parser.add_argument("--config", default="configs/combined_NCAR.yaml",
                        help="Path to model YAML config.")
    parser.add_argument("--checkpoint", default=None,
                        help="Forecaster checkpoint (Combined) or full state dict (TrainModule). "
                             "Overrides the YAML config if given.")
    parser.add_argument("--downscaler_checkpoint", default=None,
                        help="(CombinedModule only) downscaler checkpoint override.")
    parser.add_argument("--start_date", required=True,
                        help="Initial-condition date, YYYY-MM-DD (e.g. 1979-01-01). "
                             "Earliest available ERA5 file is 1979-01-01.")
    parser.add_argument("--end_date", default="2025-01-01",
                        help="Final prediction date, YYYY-MM-DD (inclusive). Default 2025-01-01.")
    parser.add_argument("--output_root", default="/glade/campaign/univ/uchi0014",
                        help="Root directory; one subfolder is created per ensemble member.")
    parser.add_argument("--member_name", default=None,
                        help="Folder name under --output_root. Defaults to member_<YYYYMMDD>.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: training.seed from config). "
                             "Ignored on resume — the saved RNG state takes precedence.")
    parser.add_argument("--force_restart", action="store_true",
                        help="Discard any existing rollout checkpoint in the member directory "
                             "and start from the IC.")
    parser.add_argument("--devices", nargs="+", default=[], help="GPU device ids.")
    parser.add_argument("--spectral_monitor", action="store_true",
                        help="Monitor 2m_temperature zonal power spectrum and rewind the "
                             "rollout when its normalized spectral RMSE vs. an ERA5 reference "
                             "exceeds --spectral_threshold.")
    parser.add_argument("--spectral_lag_days", type=int, default=10,
                        help="Look-back window (days) for both the reference spectrum and the "
                             "rewind target. Default 10.")
    parser.add_argument("--spectral_threshold", type=float, default=0.5,
                        help="Per-wavenumber relative spectral RMSE above which a rewind is "
                             "triggered (k=0 excluded). Default 0.5.")
    parser.add_argument("--spectral_max_restarts", type=int, default=100,
                        help="Maximum number of spectral-rewinds before monitoring is suppressed "
                             "and the rollout is allowed to continue unchecked. Default 100.")
    args = parser.parse_args()

    main(args)
