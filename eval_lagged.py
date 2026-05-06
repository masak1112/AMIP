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
from datetime import datetime, timedelta
from os.path import join

import cftime
import numpy as np
import torch
import xarray as xr
from lightning.pytorch import seed_everything

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
):
    """Run one autoregressive rollout. Resumes from checkpoint if present."""
    nlat = model.nlat
    nlon = model.nlon
    n_levels = model.nlevels
    levels_hpa = list(np.array(dataset.levels).astype(int))
    n_steps = len(times) - 1

    invariant = model.invariant_input.to(device)

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

    def flush_and_checkpoint(buf, x, step_idx):
        """Write the month's NetCDFs and save a resume checkpoint."""
        write_month(buf, out_dir, lat_coord, lon_coord, levels_hpa, member_name)
        save_rollout_checkpoint(out_dir, x, step_idx, device)

    # ------------------------- Resume vs fresh start --------------------
    resumed = None if force_restart else load_rollout_checkpoint(out_dir, device)
    if force_restart:
        remove_rollout_checkpoint(out_dir)

    if resumed is not None:
        x = resumed["x"]
        last_done = resumed["t_idx"]
        start_step = last_done + 1
        if start_step >= n_steps:
            print(f"  Checkpoint indicates rollout already complete (last step {last_done}).")
            remove_rollout_checkpoint(out_dir)
            return
        # The previous checkpoint was taken at end-of-month. The next predicted
        # time (times[start_step + 1]) starts a new month — initialize buf for it.
        next_t_out = times[start_step + 1]
        buf = fresh_buffer(next_t_out.year, next_t_out.month)
        print(f"  Resuming from step {start_step}/{n_steps} "
              f"(next prediction: {next_t_out.isoformat()})", flush=True)
    else:
        # Fresh start: load IC, record it, build initial low-res rollout state.
        t0 = times[0]
        surface_t, upper_air_t, diagnostic_t = load_state(dataset, t0)
        surface_t = surface_t.to(device)
        upper_air_t = upper_air_t.to(device)
        diagnostic_t = diagnostic_t.to(device)

        buf = fresh_buffer(t0.year, t0.month)
        surf_np, multi_np, diag_np = to_np(surface_t, upper_air_t, diagnostic_t)
        buf.add(t0, surf_np, multi_np, diag_np)

        x = model.preprocess(surface_t, upper_air_t, diagnostic_t)

        # Edge case: IC is itself the last day of its month — flush immediately.
        # Use t_idx = -1 to mean "no model steps run yet"; resume will set
        # start_step = 0 and re-init buf for the next month.
        if is_last_day_of_month(t0):
            flush_and_checkpoint(buf, x, -1)
            buf = fresh_buffer((t0 + timedelta(days=1)).year,
                               (t0 + timedelta(days=1)).month)
        start_step = 0

    # ------------------------- Rollout loop -----------------------------
    log_every = max(1, n_steps // 200)
    start_clock = time.time()

    with torch.no_grad():
        for step_idx in range(start_step, n_steps):
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

            # Defensive: if t_out belongs to a different month than the current
            # buffer (shouldn't happen given our checkpoint-at-month-end logic,
            # but guard against drift), flush the old buffer first.
            if (t_out.year, t_out.month) != (buf.year, buf.month):
                if buf.times:
                    write_month(buf, out_dir, lat_coord, lon_coord,
                                levels_hpa, member_name)
                buf = fresh_buffer(t_out.year, t_out.month)

            buf.add(t_out, surf_np, multi_np, diag_np)
            x = y  # autoregressive: feed the Euler-updated state forward

            # Flush + checkpoint at month boundaries.
            if is_last_day_of_month(t_out):
                print(f"  Flushing {buf.year:04d}-{buf.month:02d} "
                      f"({len(buf.times)} days) ...", flush=True)
                flush_and_checkpoint(buf, x, step_idx)
                if step_idx + 1 < n_steps:
                    next_t_out = times[step_idx + 2]
                    buf = fresh_buffer(next_t_out.year, next_t_out.month)

            if (step_idx + 1) % log_every == 0 or step_idx == n_steps - 1:
                elapsed = time.time() - start_clock
                rate = (step_idx + 1 - start_step) / max(elapsed, 1e-6)
                eta = (n_steps - step_idx - 1) / max(rate, 1e-6)
                print(f"  step {step_idx+1}/{n_steps}  t={t_out.isoformat()}  "
                      f"{rate:.2f} steps/s  eta {eta/3600:.2f}h",
                      flush=True)

    # ------------------------- Finalize ---------------------------------
    # Flush any trailing partial month (rollout end didn't fall on a
    # month boundary, or an early break above).
    if buf.times:
        print(f"  Flushing trailing {buf.year:04d}-{buf.month:02d} "
              f"({len(buf.times)} days) ...", flush=True)
        write_month(buf, out_dir, lat_coord, lon_coord, levels_hpa, member_name)

    remove_rollout_checkpoint(out_dir)


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
    args = parser.parse_args()

    main(args)
