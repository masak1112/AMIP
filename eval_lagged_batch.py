"""eval_lagged_batch.py — Batched ensemble rollout (multiple members at one IC).

Runs an ensemble of ``B = batch_size`` members in a single batched rollout
on one GPU, all sharing one ``--seed`` and starting from the same date.
Each member's predicted trajectory is saved to its own directory;
intra-batch diversity comes from ``torch.randn(B, ...)`` producing
independent noise per batch element inside the diffusion sampler. (Per-member
seeds are not currently supported; all members in a batch share one seed.)

Output layout (one folder per ensemble member):

    {output_root}/seed_{seed}_{batch_idx}/
        config.yaml
        meta.yaml
        rollout_checkpoint.pt
        surface/<var>/<var>_<YYYYMM>.nc
        multilevel/<var>/<var>_<YYYYMM>.nc
        diagnostic/<var>/<var>_<YYYYMM>.nc

Multi-GPU: pass ``--seeds`` (one seed per GPU) and ``--devices`` (matching
length). The script forks one Python subprocess per ``(seed, device)`` pair,
with ``CUDA_VISIBLE_DEVICES=<N>`` set in each child so it sees only its
assigned GPU and uses ``cuda:0`` internally. Subprocess is used (instead of
``torch.multiprocessing.spawn``) because spawn workers that inherit a shared
``CUDA_VISIBLE_DEVICES`` tend to deadlock when each calls
``torch.cuda.manual_seed_all`` simultaneously on the same device set.

Run examples:

    # Single GPU, batch of 4 members sharing seed 60
    python eval_lagged_batch.py --config=configs/combined_NCAR.yaml \\
        --start_date=1980-01-01 --seed 60 --batch_size 4 \\
        --device 0 --output_root=/scratch/.../rollouts/

    # 4 GPUs, 4 members each (16 total), one seed per GPU
    python eval_lagged_batch.py --config=configs/combined_NCAR.yaml \\
        --start_date=1980-01-01 --seeds 60 64 68 72 --batch_size 4 \\
        --devices 0 1 2 3 --output_root=/scratch/.../rollouts/

Resume: each member dir holds a slice of the batched rollout state and a
copy of the global RNG state. All slices must be at the same step; the RNG
is loaded from member 0 (slices share the same RNG snapshot). Use
``--force_restart`` to discard the checkpoints and start from the IC.

Spectral monitor: when ``--spectral_monitor`` is set, the per-member
``2m_temperature`` spectral RMSE is computed; if any member exceeds the
threshold the entire batch is rewound ``spectral_lag_days`` days and
reseeded from ERA5.
"""

import argparse
import os
import random
import subprocess
import sys
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
    """Per-degree relative RMSE between two spherical-harmonic power spectra
    (l=0 excluded so the metric tracks variability drift, not the global mean)."""
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
    out = []
    t = start
    while t <= end:
        out.append(t)
        t = t + timedelta(days=1)
    return out


def parse_date(s: str):
    d = datetime.strptime(s, "%Y-%m-%d")
    return cftime.DatetimeGregorian(d.year, d.month, d.day, 0, 0, 0,
                                    has_year_zero=False)


def is_last_day_of_month(t) -> bool:
    nxt = t + timedelta(days=1)
    return (nxt.year, nxt.month) != (t.year, t.month)


# ---------------------------------------------------------------------------
# Manual data loading (single-batch helpers; expand to B at call site).
# ---------------------------------------------------------------------------

def load_state(dataset, t):
    """Load full atmospheric state at time ``t`` (normalized), with leading
    batch dim of size 1."""
    raw = dataset._get_data(t, out=False)
    upper_air, surface, diagnostic, _ = dataset._reshape_and_mask_variables(raw, out=False)

    surface = dataset.surface_transform(surface).unsqueeze(0)
    upper_air = dataset.upper_air_transform(upper_air).unsqueeze(0)
    diagnostic = dataset.diagnostic_transform(diagnostic).unsqueeze(0)
    return surface, upper_air, diagnostic


def load_forcing(dataset, t):
    """Load varying boundary forcing at time ``t``, with leading batch dim 1."""
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
# Per-month writer (one MonthBuffer per ensemble member).
# ---------------------------------------------------------------------------

class MonthBuffer:
    """In-memory buffer for one month's worth of daily snapshots (one member)."""

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
        """surf_np: (c_sfc, nlat, nlon); multi_np: (c_ua, nlevels, nlat, nlon);
        diag_np: (c_diag, nlat, nlon)."""
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
    if not buf.times:
        return

    surf, multi, diag = buf.stack()
    times = np.asarray(buf.times)
    tag = f"{buf.year:04d}{buf.month:02d}"

    common_attrs = {
        "Conventions": "CF-1.8",
        "source": "amip lagged-ensemble batched rollout",
        "variant_label": member_name,
        "frequency": "day",
        "grid_label": "gn",
    }
    time_encoding = {
        "dtype": "float64",
        "units": "days since 1850-01-01 00:00:00",
        "calendar": "standard",
    }

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
            if os.path.exists(path):
                os.remove(path)
            ds.to_netcdf(path, format="NETCDF4", engine="h5netcdf",
                         encoding=encoding, unlimited_dims=["time"])

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
# Rollout checkpoint (per-member slice; shared global RNG snapshot).
# ---------------------------------------------------------------------------

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


def save_batched_rollout_checkpoint(member_dirs, x: torch.Tensor, t_idx: int,
                                    device, seed: int, batch_size: int):
    """Atomically save one slice of x per member dir, sharing one RNG snapshot."""
    rng = _gather_rng_state(device)
    x_cpu = x.detach().cpu()
    for i, out_dir in enumerate(member_dirs):
        payload = {
            "x_slice": x_cpu[i],
            "t_idx": int(t_idx),
            "rng": rng,
            "seed": int(seed),
            "batch_index": int(i),
            "batch_size": int(batch_size),
            "format_version": 2,
        }
        path = join(out_dir, CHECKPOINT_FILENAME)
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)


def load_batched_rollout_checkpoint(member_dirs, device, seed: int, batch_size: int):
    """Stack per-member slices into a batched x; restore RNG from member 0.
    Returns dict {x, t_idx} or None if any slice is missing."""
    paths = [join(d, CHECKPOINT_FILENAME) for d in member_dirs]
    if not all(os.path.exists(p) for p in paths):
        return None
    payloads = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]

    t_idxs = {p["t_idx"] for p in payloads}
    if len(t_idxs) != 1:
        raise SystemExit(f"Member checkpoints disagree on t_idx: {t_idxs}. "
                         "Re-run with --force_restart to discard them.")
    saved_seeds = {p["seed"] for p in payloads}
    if saved_seeds != {seed}:
        raise SystemExit(f"Saved seed(s) {saved_seeds} don't match --seed {seed}. "
                         "Re-run with --force_restart or pass the original seed.")
    saved_bs = {p["batch_size"] for p in payloads}
    if saved_bs != {batch_size}:
        raise SystemExit(f"Saved batch_size(s) {saved_bs} don't match --batch_size "
                         f"{batch_size}. Re-run with --force_restart.")

    _restore_rng_state(payloads[0]["rng"], device)
    x = torch.stack([p["x_slice"] for p in payloads], dim=0).to(device)
    return {"x": x, "t_idx": payloads[0]["t_idx"]}


def remove_batched_rollout_checkpoint(member_dirs):
    for d in member_dirs:
        path = join(d, CHECKPOINT_FILENAME)
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
# Batched rollout
# ---------------------------------------------------------------------------

def rollout_batch(
    model,
    dataset,
    times,
    member_dirs,
    member_names,
    seed: int,
    batch_size: int,
    lat_coord,
    lon_coord,
    device,
    force_restart: bool = False,
    spectral_monitor: bool = False,
    spectral_lag_days: int = 10,
    spectral_threshold: float = 1.0,
    spectral_max_restarts: int = 1000,
):
    """Batched autoregressive rollout for B=batch_size ensemble members.

    All members share one ``seed``; intra-batch diversity comes from the
    diffusion sampler's ``torch.randn(B, ...)`` calls. Each member's
    prediction is appended to its own MonthBuffer and flushed to its own
    output directory at month boundaries. Resume reconstructs the batched
    state by stacking per-member slices.

    Spectral monitor (when enabled): the per-member 2m_T spectral RMSE is
    compared to an ERA5 reference; if the maximum across members exceeds
    ``spectral_threshold`` the entire batch is rewound ``spectral_lag_days``
    days and re-seeded from ERA5 at the rewind target.
    """
    B = batch_size
    assert len(member_dirs) == B == len(member_names)
    nlat = model.nlat
    nlon = model.nlon
    n_levels = model.nlevels
    levels_hpa = list(np.array(dataset.levels).astype(int))
    n_steps = len(times) - 1

    invariant = model.invariant_input.to(device).expand(B, -1, -1, -1).contiguous()

    t2m_index = None
    sht_power = None
    if spectral_monitor:
        try:
            t2m_index = list(model.surface_variables).index("2m_temperature")
        except ValueError:
            print("  [spectral_monitor] '2m_temperature' not in surface_variables -- disabling monitor.",
                  flush=True)
            spectral_monitor = False
        else:
            sht_power = _SphericalSpectrum(nlat, nlon, device=device)

    ref_spectrum_cache: dict = {}

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

    def fresh_buffers(year, month):
        return [
            MonthBuffer(year, month,
                        model.surface_variables, model.multilevel_variables,
                        model.diagnostic_variables, nlat, nlon, n_levels)
            for _ in range(B)
        ]

    def to_np_batched(surf_t, multi_t, diag_t):
        """Inputs are batched (B, ...). Returns numpy (B, c, ...) per group."""
        surf_d = dataset.surface_inv_transform(surf_t)
        multi_d = dataset.upper_air_inv_transform(multi_t)
        diag_d = dataset.diagnostic_inv_transform(diag_t)
        return (
            surf_d.detach().cpu().numpy(),
            multi_d.detach().cpu().numpy(),
            diag_d.detach().cpu().numpy(),
        )

    def expand_state_to_batch(surface_t, upper_air_t, diagnostic_t):
        """Take (1, ...) tensors on device and expand to (B, ...)."""
        return (
            surface_t.expand(B, -1, -1, -1).contiguous(),
            upper_air_t.expand(B, -1, -1, -1, -1).contiguous(),
            diagnostic_t.expand(B, -1, -1, -1).contiguous(),
        )

    bufs_ref = [None]  # list of B MonthBuffers

    def commit_to_bufs(entry):
        """Commit one pending entry (one timestep, B members) into the buffers."""
        _, t_out, surf_np, multi_np, diag_np = entry  # surf_np: (B, c, h, w)
        bufs = bufs_ref[0]
        if (t_out.year, t_out.month) != (bufs[0].year, bufs[0].month):
            for i, buf in enumerate(bufs):
                if buf.times:
                    write_month(buf, member_dirs[i], lat_coord, lon_coord,
                                levels_hpa, member_names[i])
            bufs = fresh_buffers(t_out.year, t_out.month)
        for i, buf in enumerate(bufs):
            buf.add(t_out, surf_np[i], multi_np[i], diag_np[i])
        if is_last_day_of_month(t_out):
            print(f"  Flushing {bufs[0].year:04d}-{bufs[0].month:02d} "
                  f"({len(bufs[0].times)} days x {B} members) ...", flush=True)
            for i, buf in enumerate(bufs):
                write_month(buf, member_dirs[i], lat_coord, lon_coord,
                            levels_hpa, member_names[i])
            next_t = t_out + timedelta(days=1)
            bufs = fresh_buffers(next_t.year, next_t.month)
        bufs_ref[0] = bufs

    pending: deque = deque()
    pending_capacity = (spectral_lag_days + 1) if spectral_monitor else 1

    # ------------------------- Resume vs fresh start --------------------
    resumed = (None if force_restart
               else load_batched_rollout_checkpoint(member_dirs, device, seed, B))
    if force_restart:
        remove_batched_rollout_checkpoint(member_dirs)

    if resumed is not None:
        x = resumed["x"]
        last_done = resumed["t_idx"]
        start_step = last_done + 1
        if start_step >= n_steps:
            print(f"  Checkpoint indicates rollout already complete (last step {last_done}).")
            remove_batched_rollout_checkpoint(member_dirs)
            return
        next_t_out = times[start_step + 1]
        bufs_ref[0] = fresh_buffers(next_t_out.year, next_t_out.month)
        print(f"  Resuming from step {start_step}/{n_steps} "
              f"(next prediction: {next_t_out.isoformat()})", flush=True)
    else:
        t0 = times[0]
        surface_t, upper_air_t, diagnostic_t = load_state(dataset, t0)
        surface_t, upper_air_t, diagnostic_t = expand_state_to_batch(
            surface_t.to(device), upper_air_t.to(device), diagnostic_t.to(device))

        bufs_ref[0] = fresh_buffers(t0.year, t0.month)
        surf_np, multi_np, diag_np = to_np_batched(surface_t, upper_air_t, diagnostic_t)
        for i, buf in enumerate(bufs_ref[0]):
            buf.add(t0, surf_np[i], multi_np[i], diag_np[i])

        x = model.preprocess(surface_t, upper_air_t, diagnostic_t)

        if is_last_day_of_month(t0):
            for i, buf in enumerate(bufs_ref[0]):
                write_month(buf, member_dirs[i], lat_coord, lon_coord,
                            levels_hpa, member_names[i])
            save_batched_rollout_checkpoint(member_dirs, x, -1, device, seed, B)
            next_t = t0 + timedelta(days=1)
            bufs_ref[0] = fresh_buffers(next_t.year, next_t.month)
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
                print(f"  Forcing missing for {t_in.isoformat()}: {e} -- stopping at step {step_idx}.",
                      flush=True)
                break
            boundary = boundary.to(device, non_blocking=True).expand(B, -1, -1, -1).contiguous()
            if calendar is not None:
                calendar = calendar.to(device, non_blocking=True).expand(B, -1).contiguous()

            c_grid = assemble_forcing(boundary, invariant)

            fwd_kwargs = {"return_model_last": True}
            if calendar is not None:
                fwd_kwargs["c_scalar"] = calendar
            y, y_last = model.forward(x, c_grid, **fwd_kwargs)

            surf_pred, multi_pred, diag_pred = disassemble_input(y_last, nlevels=n_levels)
            surf_np, multi_np, diag_np = to_np_batched(surf_pred, multi_pred, diag_pred)

            pending.append((step_idx, t_out, surf_np, multi_np, diag_np))
            while len(pending) > pending_capacity:
                commit_to_bufs(pending.popleft())

            x = y
            n_iterations += 1

            # ---- Spectral monitor: trigger rewind on max-member blowup ----
            restart_triggered = False
            if (spectral_monitor
                    and len(pending) >= spectral_lag_days + 1
                    and n_restarts < spectral_max_restarts):
                ref_t = times[step_idx + 1 - spectral_lag_days]
                ref_spec = reference_spectrum(ref_t)
                if ref_spec is not None:
                    member_rmses = []
                    for i in range(B):
                        pred_spec = sht_power(torch.from_numpy(surf_np[i, t2m_index]))
                        member_rmses.append(normalized_spectral_rmse(pred_spec, ref_spec))
                    rmse = max(member_rmses)
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

                        worst = int(np.argmax(member_rmses))
                        print(f"  [spectral_monitor] step {step_idx} t={t_out.isoformat()} "
                              f"max 2m_T spectral rmse={rmse:.3f} "
                              f"(member {worst}/{B}) > {current_threshold:.3f} -- "
                              f"rewinding {spectral_lag_days} days to {rewind_t.isoformat()} "
                              f"(restart #{n_restarts}, same-target #{same_target_rewinds}).",
                              flush=True)

                        if same_target_rewinds > 1 and current_threshold < MAX_THRESHOLD:
                            new_threshold = min(current_threshold + THRESHOLD_BUMP, MAX_THRESHOLD)
                            print(f"  [spectral_monitor] {same_target_rewinds} rewinds at step "
                                  f"{rewind_step} ({rewind_t.isoformat()}) -- bumping threshold "
                                  f"{current_threshold:.3f} -> {new_threshold:.3f}.", flush=True)
                            current_threshold = new_threshold

                        pending.clear()
                        # Reload ERA5 IC at rewind target, expand to batch B.
                        surface_t, upper_air_t, diagnostic_t = load_state(dataset, rewind_t)
                        surface_t, upper_air_t, diagnostic_t = expand_state_to_batch(
                            surface_t.to(device), upper_air_t.to(device), diagnostic_t.to(device))
                        x = model.preprocess(surface_t, upper_air_t, diagnostic_t)
                        gt_surf_np, gt_multi_np, gt_diag_np = to_np_batched(
                            surface_t, upper_air_t, diagnostic_t)
                        pending.append((rewind_step - 1, rewind_t,
                                        gt_surf_np, gt_multi_np, gt_diag_np))
                        step_idx = rewind_step
                        restart_triggered = True

            if restart_triggered:
                continue

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
            if is_last_day_of_month(t_out):
                while pending:
                    commit_to_bufs(pending.popleft())
                save_batched_rollout_checkpoint(member_dirs, x, step_idx, device,
                                                seed, B)

            if n_iterations % log_every == 0 or step_idx == n_steps - 1:
                elapsed = time.time() - start_clock
                rate = n_iterations / max(elapsed, 1e-6)
                eta = (n_steps - step_idx - 1) / max(rate, 1e-6)
                restart_tag = f"  restarts={n_restarts}" if spectral_monitor else ""
                print(f"  step {step_idx+1}/{n_steps}  t={t_out.isoformat()}  "
                      f"{rate:.2f} steps/s (B={B})  eta {eta/3600:.2f}h{restart_tag}",
                      flush=True)

            step_idx += 1

    # ------------------------- Finalize ---------------------------------
    while pending:
        commit_to_bufs(pending.popleft())
    bufs = bufs_ref[0]
    if bufs is not None and bufs[0].times:
        print(f"  Flushing trailing {bufs[0].year:04d}-{bufs[0].month:02d} "
              f"({len(bufs[0].times)} days x {B} members) ...", flush=True)
        for i, buf in enumerate(bufs):
            write_month(buf, member_dirs[i], lat_coord, lon_coord,
                        levels_hpa, member_names[i])

    remove_batched_rollout_checkpoint(member_dirs)


# ---------------------------------------------------------------------------
# Single-GPU worker
# ---------------------------------------------------------------------------

def _run_single_gpu(args, seed: int, device_id: int):
    """One batched rollout: ``args.batch_size`` members on one GPU sharing ``seed``.

    Member directories are ``{output_root}/seed_{seed}_{batch_idx}/``.
    """
    label = os.environ.get("_WORKER_LABEL", "")
    log_prefix = f"[{label}] " if label else ""

    config = get_yaml(args.config)
    dataconfig = config["data"]
    trainconfig = config["training"]

    if args.checkpoint is not None:
        trainconfig["checkpoint"] = args.checkpoint
        trainconfig["forecaster_checkpoint"] = args.checkpoint
    if args.downscaler_checkpoint is not None:
        trainconfig["downscaler_checkpoint"] = args.downscaler_checkpoint

    dataconfig["batch_size"] = 1
    dataconfig["epsilon_factor"] = 0
    dataconfig["forecast_lead_times"] = [1]

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = f"cuda:{device_id}"
    else:
        device = "cpu"
    print(f"{log_prefix}Using device {device} (seed={seed}).", flush=True)

    seed_everything(seed)
    print(f"{log_prefix}seed_everything({seed}) done.", flush=True)
    torch.set_float32_matmul_precision("high")

    # ---- Time axis ----
    t_start = parse_date(args.start_date)
    t_end = parse_date(args.end_date)
    if t_end <= t_start:
        raise SystemExit(f"--end_date ({args.end_date}) must be after --start_date ({args.start_date}).")
    times = daterange_days(t_start, t_end)
    B = args.batch_size
    print(f"{log_prefix}Rollout: {t_start.isoformat()} -> {t_end.isoformat()} "
          f"({len(times)} dates, {len(times)-1} model steps)", flush=True)
    print(f"{log_prefix}Batch: B={B} members on {device}, seed={seed}", flush=True)

    # ---- Dataset / model ----
    year_start = t_start.year
    year_end = t_end.year + 1
    dataset = GetDataset(dataconfig, year_start=year_start, year_end=year_end)
    print(f"{log_prefix}Dataset ready (years {year_start}..{year_end-1}).", flush=True)

    model, _ = build_model(config, dataset, args.checkpoint,
                           args.downscaler_checkpoint, device)
    print(f"{log_prefix}Model loaded on {device}.", flush=True)

    nlat = model.nlat
    nlon = model.nlon
    lat_coord, lon_coord = make_lat_lon(nlat, nlon)
    print(f"{log_prefix}Pressure levels (hPa): {list(np.array(dataset.levels).astype(int))}", flush=True)
    print(f"{log_prefix}Surface vars:    {model.surface_variables}", flush=True)
    print(f"{log_prefix}Multilevel vars: {model.multilevel_variables}", flush=True)
    print(f"{log_prefix}Diagnostic vars: {model.diagnostic_variables}", flush=True)

    # ---- Per-member output dirs ----
    member_names = [f"seed_{seed}_{i}" for i in range(B)]
    member_dirs = [join(args.output_root, m) for m in member_names]
    for i, (m, d) in enumerate(zip(member_names, member_dirs)):
        os.makedirs(d, exist_ok=True)
        save_yaml(config, join(d, "config.yaml"))
        save_yaml(
            {
                "member_name": m,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "seed": int(seed),
                "batch_index": i,
                "batch_size": B,
                "checkpoint": trainconfig.get("forecaster_checkpoint", trainconfig.get("checkpoint")),
                "downscaler_checkpoint": trainconfig.get("downscaler_checkpoint"),
                "n_steps": len(times) - 1,
                "horizontal_resolution": [nlat, nlon],
                "levels_hpa": list(np.array(dataset.levels).astype(int)),
            },
            join(d, "meta.yaml"),
        )
        print(f"{log_prefix}  Member dir: {d}", flush=True)

    if (all(os.path.exists(join(d, CHECKPOINT_FILENAME)) for d in member_dirs)
            and not args.force_restart):
        print(f"{log_prefix}  (rollout checkpoints detected -- will resume)", flush=True)

    rollout_batch(
        model=model,
        dataset=dataset,
        times=times,
        member_dirs=member_dirs,
        member_names=member_names,
        seed=seed,
        batch_size=B,
        lat_coord=lat_coord,
        lon_coord=lon_coord,
        device=device,
        force_restart=args.force_restart,
        spectral_monitor=args.spectral_monitor,
        spectral_lag_days=args.spectral_lag_days,
        spectral_threshold=args.spectral_threshold,
        spectral_max_restarts=args.spectral_max_restarts,
    )

    print(f"{log_prefix}Done. {B} members written under {args.output_root}.", flush=True)


# ---------------------------------------------------------------------------
# Multi-GPU dispatcher (one subprocess per GPU)
# ---------------------------------------------------------------------------

def _build_subprocess_cmd(args, seed: int):
    """Reconstruct a single-GPU CLI invocation for one (seed, device) worker.
    The child sees one GPU via CUDA_VISIBLE_DEVICES and uses cuda:0 internally,
    so we always pass --device 0 to the child."""
    cmd = [sys.executable, sys.argv[0]]
    skip = {"seed", "seeds", "device", "devices"}
    for name, val in vars(args).items():
        if name in skip or val is None:
            continue
        flag = f"--{name}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        elif isinstance(val, list):
            if not val:
                continue
            cmd.append(flag)
            cmd.extend(str(v) for v in val)
        else:
            cmd.append(flag)
            cmd.append(str(val))
    cmd.extend(["--seed", str(seed), "--device", "0"])
    return cmd


def _dispatch_subprocesses(args, seeds, devices):
    """Launch one subprocess per (seed, device) and wait for them all."""
    print(f"Dispatching {len(seeds)} workers: seeds={seeds}, devices={devices}, "
          f"batch_size={args.batch_size} (= {len(seeds) * args.batch_size} total members).",
          flush=True)
    procs = []
    for seed, device_id in zip(seeds, devices):
        env = os.environ.copy()
        # Child sees only its assigned physical GPU; inside the child it is cuda:0.
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
        env["_WORKER_LABEL"] = f"gpu{device_id}"
        cmd = _build_subprocess_cmd(args, seed)
        print(f"  Launching: gpu={device_id} seed={seed} -> "
              f"CUDA_VISIBLE_DEVICES={device_id} {' '.join(cmd)}", flush=True)
        p = subprocess.Popen(cmd, env=env)
        procs.append((p, seed, device_id))

    rcs = []
    for p, seed, device_id in procs:
        rc = p.wait()
        rcs.append((seed, device_id, rc))
        print(f"  Worker (gpu={device_id}, seed={seed}) exited rc={rc}.", flush=True)
    if any(rc != 0 for _, _, rc in rcs):
        failed = [(s, d) for s, d, rc in rcs if rc != 0]
        raise SystemExit(f"Workers failed: {failed}")
    total = len(seeds) * args.batch_size
    print(f"\nAll {len(seeds)} workers complete. {total} members written under "
          f"{args.output_root}.", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    """Top-level entry: dispatch one subprocess per GPU when ``--seeds`` is
    given (multi-GPU); otherwise run a single-GPU batched rollout in-process."""
    seeds_given = args.seeds is not None and len(args.seeds) > 0
    seed_given = args.seed is not None

    if seeds_given and seed_given:
        raise SystemExit("Pass either --seed (single-GPU worker) "
                         "or --seeds (multi-GPU dispatcher), not both.")
    if not (seeds_given or seed_given):
        raise SystemExit("Pass --seed (single-GPU worker) "
                         "or --seeds (multi-GPU dispatcher).")

    if seeds_given:
        seeds = [int(s) for s in args.seeds]
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"--seeds must be unique, got: {seeds}")
        if args.devices:
            devices = [int(d) for d in args.devices]
        else:
            devices = list(range(len(seeds)))
        if len(seeds) != len(devices):
            raise SystemExit(f"len(--seeds)={len(seeds)} must equal "
                             f"len(--devices)={len(devices)}.")
        _dispatch_subprocesses(args, seeds, devices)
    else:
        if args.device is not None:
            device_id = int(args.device)
        elif args.devices and len(args.devices) == 1:
            device_id = int(args.devices[0])
        else:
            device_id = 0
        _run_single_gpu(args, seed=int(args.seed), device_id=device_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batched ensemble rollout (multiple members at one IC).")
    parser.add_argument("--config", default="configs/combined_NCAR.yaml",
                        help="Path to model YAML config.")
    parser.add_argument("--checkpoint", default=None,
                        help="Forecaster checkpoint (Combined) or full state dict (TrainModule). "
                             "Overrides the YAML config if given.")
    parser.add_argument("--downscaler_checkpoint", default=None,
                        help="(CombinedModule only) downscaler checkpoint override.")
    parser.add_argument("--start_date", required=True,
                        help="Initial-condition date, YYYY-MM-DD (e.g. 1980-01-01). "
                             "All members in the batch start from this date.")
    parser.add_argument("--end_date", default="2025-01-01",
                        help="Final prediction date, YYYY-MM-DD (inclusive). Default 2025-01-01.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of ensemble members per GPU (batched together). "
                             "All members in a batch share one --seed; intra-batch "
                             "diversity comes from torch.randn(B, ...) in the diffusion "
                             "sampler. Default 4.")
    parser.add_argument("--seed", type=int, default=None,
                        help="(Single-GPU mode) Seed for this batched rollout. Mutually "
                             "exclusive with --seeds.")
    parser.add_argument("--device", type=int, default=None,
                        help="(Single-GPU mode) GPU device id for this rollout. "
                             "Default 0.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="(Multi-GPU dispatcher mode) One seed per GPU. The script "
                             "forks one Python subprocess per (seed, device) pair, with "
                             "CUDA_VISIBLE_DEVICES set in each child. Total members = "
                             "len(--seeds) * --batch_size.")
    parser.add_argument("--devices", type=int, nargs="+", default=None,
                        help="(Multi-GPU dispatcher mode) Physical GPU ids matching "
                             "len(--seeds). Default: 0..len(seeds)-1.")
    parser.add_argument("--output_root", default="/glade/campaign/univ/uchi0014",
                        help="Root directory; one subfolder is created per ensemble member: "
                             "{output_root}/seed_{seed}_{batch_idx}/.")
    parser.add_argument("--force_restart", action="store_true",
                        help="Discard any existing rollout checkpoints in the member "
                             "directories and start from the IC.")
    parser.add_argument("--spectral_monitor", action="store_true",
                        help="Monitor 2m_temperature spectral RMSE per member; rewind the "
                             "WHOLE batch when the maximum across members exceeds "
                             "--spectral_threshold.")
    parser.add_argument("--spectral_lag_days", type=int, default=10,
                        help="Look-back window (days) for both the reference spectrum and the "
                             "rewind target. Default 10.")
    parser.add_argument("--spectral_threshold", type=float, default=1.0,
                        help="Per-wavenumber relative spectral RMSE above which a rewind is "
                             "triggered (k=0 excluded). Default 1.0.")
    parser.add_argument("--spectral_max_restarts", type=int, default=100,
                        help="Maximum number of spectral-rewinds before monitoring is "
                             "suppressed and the rollout is allowed to continue unchecked. "
                             "Default 100.")
    args = parser.parse_args()

    main(args)
