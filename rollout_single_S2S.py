import argparse
import importlib.util
import os
import time
from datetime import datetime, timedelta
from os.path import join

import cftime
import numpy as np
import pandas as pd
import torch
import xarray as xr
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader, Dataset

from common.utils import (
    assemble_forcing,
    disassemble_input,
    get_yaml,
    save_yaml,
)
from data.amip_new import GetDataset
from modules.combined_module import CombinedModule
from modules.train_module import TrainModule


HAS_H5NETCDF = importlib.util.find_spec("h5netcdf") is not None
HAS_NETCDF4 = importlib.util.find_spec("netCDF4") is not None


def make_lat_lon(nlat: int, nlon: int):
    dlat = 180.0 / nlat
    dlon = 360.0 / nlon
    lat = np.arange(-90.0 + 0.5 * dlat, 90.0, dlat, dtype=np.float64)
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


class ForcingDataset(Dataset):
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.return_calendar = dataset.return_calendar

    def __len__(self):
        return len(self.times)

    def load_forcing(self, t):
        raw = torch.tensor(
            self.dataset._get_data(t, variable_list=self.dataset.varying_boundary_variables)
        ).to(torch.float32)
        raw = self.dataset._fill_mask(raw, self.dataset.varying_boundary_variables)
        boundary = self.dataset.boundary_transform(raw)

        calendar = None
        if self.dataset.return_calendar:
            co2 = boundary[0, 0, 0].clone()
            boundary = boundary[1:]
            sod, doy = self.dataset._compute_calendar(t)
            calendar = torch.tensor([[sod, doy, co2.item()]], dtype=torch.float32)

        return boundary, calendar

    def __getitem__(self, idx):
        t = self.times[idx]
        boundary, calendar = self.load_forcing(t)
        if self.return_calendar:
            return boundary, calendar.squeeze(0)
        return boundary


def make_forcing_loader(dataset, times, num_workers):
    """Build a DataLoader that yields forcing for every entry in ``times``."""
    forcing_ds = ForcingDataset(dataset, times)
    return DataLoader(
        forcing_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def build_model(config, dataset, device):
    """Instantiate either CombinedModule or TrainModule based on config.model.model_name.

    CombinedModule loads ``forecaster_checkpoint`` + ``downscaler_checkpoint``
    internally during ``__init__``; TrainModule needs its checkpoint loaded
    manually here.
    """
    is_combined = config["model"].get("model_name", "") == "Combined"
    if is_combined:
        model = CombinedModule(config, normalizer=dataset).to(device)
        forecaster = config["training"].get("forecaster_checkpoint")
        downscaler = config["training"].get("downscaler_checkpoint")
        print(f"  Loaded CombinedModule "
              f"(forecaster={forecaster}, downscaler={downscaler})", flush=True)
    else:
        model = TrainModule(config, normalizer=dataset).to(device)
        ckpt = config["training"]["checkpoint"]
        state_dict = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
        model.load_state_dict(state_dict)
        print(f"  Loaded TrainModule from {ckpt}", flush=True)

    model.eval()
    return model, is_combined

def get_initialization_dates(year):

    """
    Get initialization dates (Calendar dates corresponding to 
    Mondays and Thursdays of 2024 May-July) for a given year.
    """
    # Define date range from May 1 to July 31 of 2024 (template)
    
    start_date = datetime(2024, 5, 1)
    end_date = datetime(2024, 7, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    # Find Mondays (weekday=0) and Thursdays (weekday=3)
    is_monday = date_range.weekday == 0
    is_thursday = date_range.weekday == 3
    filtered_dates = date_range[is_monday | is_thursday]
    
    # Convert to the requested year
    filtered_dates_yr = pd.to_datetime(filtered_dates.strftime(f'{year}-%m-%d'))
    return filtered_dates_yr

def rollout_ensemble(
    model,
    dataset,
    times,
    n_ensemble: int,
    device,
    is_combined: bool,
    num_data_workers: int = 4,
):
    """Run a batched n-member ensemble rollout over ``times`` (len = lead_time + 1).

    Works with either ``TrainModule`` (low-res output) or ``CombinedModule``
    (full-res downscaled output). Saved arrays live at the prediction
    resolution, and the t=0 IC is downsampled to match it when needed.

    Returns three float32 numpy arrays in physical units:
        surf_arr  (n_ensemble, lead_time + 1, c_sfc, nlat, nlon)
        multi_arr (n_ensemble, lead_time + 1, c_ua, n_levels, nlat, nlon)
        diag_arr  (n_ensemble, lead_time + 1, c_diag, nlat, nlon)
    """
    n_levels = model.nlevels
    # TrainModule has an ``x_pred`` flag controlling whether the scheduler can
    # return a meaningful y_last; CombinedModule's forward always returns
    # ``(y_lowres, y_highres)`` so we always want the second tensor.
    ask_y_last = is_combined or getattr(model, "x_pred", False)

    invariant = model.invariant_input.to(device)

    # ---- Initial condition (shared across ensemble members) ----
    surface_t, upper_air_t, diagnostic_t = load_state(dataset, times[0])
    surface_t = surface_t.to(device)
    upper_air_t = upper_air_t.to(device)
    diagnostic_t = diagnostic_t.to(device)

    # Replicate IC along batch dim (one slot per ensemble member).
    surface_t = surface_t.expand(n_ensemble, *surface_t.shape[1:]).contiguous()
    upper_air_t = upper_air_t.expand(n_ensemble, *upper_air_t.shape[1:]).contiguous()
    diagnostic_t = diagnostic_t.expand(n_ensemble, *diagnostic_t.shape[1:]).contiguous()

    def denorm_to_np(surf_t, multi_t, diag_t):
        """Inverse-normalize then move to CPU; preserves the batch (member) dim."""
        surf_d = dataset.surface_inv_transform(surf_t)
        multi_d = dataset.upper_air_inv_transform(multi_t)
        diag_d = dataset.diagnostic_inv_transform(diag_t)
        return (
            surf_d.detach().cpu().numpy().astype(np.float32, copy=False),
            multi_d.detach().cpu().numpy().astype(np.float32, copy=False),
            diag_d.detach().cpu().numpy().astype(np.float32, copy=False),
        )

    # ``preprocess`` handles downsample (if any) + assemble for both modules,
    # so x is always at the forecaster's working (low-res) resolution.
    x = model.preprocess(surface_t, upper_air_t, diagnostic_t)

    # IC saved at the *prediction* resolution: full-res for CombinedModule
    # (downscaler output), low-res for TrainModule (forecaster output).
    if is_combined or model.downsample is None:
        ic_surf, ic_multi, ic_diag = surface_t, upper_air_t, diagnostic_t
    else:
        ic_surf, ic_multi, ic_diag = model.downsample(
            surface_t, upper_air_t, diagnostic_t)

    surf_ic_np, multi_ic_np, diag_ic_np = denorm_to_np(ic_surf, ic_multi, ic_diag)
    surf_steps = [surf_ic_np]
    multi_steps = [multi_ic_np]
    diag_steps = [diag_ic_np]

    # ---- Forcing for each predicted step ----
    loader = make_forcing_loader(dataset, times[1:], num_workers=num_data_workers)

    try:
        with torch.no_grad():
            for step_idx, batch in enumerate(loader):
                if dataset.return_calendar:
                    boundary, calendar = batch
                    calendar = calendar.to(device, non_blocking=True)
                    calendar = calendar.expand(n_ensemble, *calendar.shape[1:]).contiguous()
                else:
                    boundary = batch
                    calendar = None
                    
                boundary = boundary.to(device, non_blocking=True)
                boundary = boundary.expand(n_ensemble, *boundary.shape[1:]).contiguous()

                print("boundary shape", boundary.shape, flush=True)
                
                invariant= invariant.expand(n_ensemble, *invariant.shape[1:]).contiguous()
                print("invariant shape", invariant.shape, flush=True)
                c_grid = assemble_forcing(boundary, invariant)

                if ask_y_last:
                    # TrainModule: (y_euler, y_xpred); CombinedModule: (y_lowres, y_highres).
                    y, save_target = model.forward(
                        x, c_grid, c_scalar=calendar, return_model_last=True)
                else:
                    y = model.forward(
                        x, c_grid, c_scalar=calendar, return_model_last=False)
                    save_target = y

                surf_pred, multi_pred, diag_pred = disassemble_input(
                    save_target, nlevels=n_levels)
                surf_np, multi_np, diag_np = denorm_to_np(
                    surf_pred, multi_pred, diag_pred)

                surf_steps.append(surf_np)
                multi_steps.append(multi_np)
                diag_steps.append(diag_np)

                # Roll the low-res forecaster state forward; for CombinedModule
                # this is the forecaster's Euler update, not the downscaled output.
                x = y
    finally:
        del loader

    # Stack along time -> (M, T, C, [L,] H, W)
    surf_arr = np.stack(surf_steps, axis=1)
    multi_arr = np.stack(multi_steps, axis=1)
    diag_arr = np.stack(diag_steps, axis=1)
    return surf_arr, multi_arr, diag_arr


def save_ensemble_nc(
    out_path: str,
    surf_arr: np.ndarray,
    multi_arr: np.ndarray,
    diag_arr: np.ndarray,
    times,
    surf_vars,
    multi_vars,
    diag_vars,
    lat,
    lon,
    levels_hpa,
    start_date_iso: str,
):
    """Write the full ensemble forecast as a single NetCDF.

    Arrays must already be in physical units. Dims:
        2D vars: (member, time, lat, lon)
        3D vars: (member, time, plev, lat, lon)
    """
    n_ensemble = surf_arr.shape[0]
    plev_pa = np.asarray(levels_hpa, dtype=np.float64) * 100.0
    member = np.arange(n_ensemble, dtype=np.int32)

    data_vars = {}
    for c, name in enumerate(surf_vars):
        data_vars[name] = (("member", "time", "lat", "lon"), surf_arr[:, :, c])
    for c, name in enumerate(multi_vars):
        data_vars[name] = (("member", "time", "plev", "lat", "lon"), multi_arr[:, :, c])
    for c, name in enumerate(diag_vars):
        data_vars[name] = (("member", "time", "lat", "lon"), diag_arr[:, :, c])

    ds = xr.Dataset(
        data_vars,
        coords={
            "member": member,
            "time": np.asarray(times),
            "plev": plev_pa,
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "Conventions": "CF-1.8",
            "source": "amip S2S ensemble rollout",
            "start_date": start_date_iso,
            "n_ensemble": n_ensemble,
            "frequency": "day",
            "grid_label": "gn",
        },
    )
    ds["lat"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
    ds["lon"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")
    ds["plev"].attrs.update(units="Pa", standard_name="air_pressure",
                            long_name="pressure", axis="Z", positive="down")
    ds["member"].attrs.update(long_name="ensemble member index")

    encoding = {
        v: {"compression": "gzip", "compression_opts": 4, "dtype": "float32"}
        for v in data_vars
    }
    encoding["time"] = {
        "dtype": "float64",
        "units": "days since 1850-01-01 00:00:00",
        "calendar": "standard",
    }

    if HAS_H5NETCDF:
        engine = "h5netcdf"
        format_name = "NETCDF4"
    elif HAS_NETCDF4:
        engine = "netcdf4"
        format_name = "NETCDF4"
    else:
        engine = "scipy"
        format_name = "NETCDF3_64BIT"
        encoding = {
            name: {"dtype": spec["dtype"]}
            for name, spec in encoding.items()
        }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    ds.to_netcdf(out_path, format=format_name, engine=engine,
                 encoding=encoding, unlimited_dims=["time"])


def main(args):
    config = get_yaml(args.config)
    dataconfig = config["data"]
    trainconfig = config["training"]

    is_combined = config["model"].get("model_name", "") == "Combined"

    # Checkpoint routing: CombinedModule reads forecaster/downscaler keys from
    # config['training']; TrainModule reads a single 'checkpoint' key. The
    # CLI flag --checkpoint sets the forecaster checkpoint for CombinedModule
    # or the full checkpoint for TrainModule.
    if is_combined:
        if args.checkpoint is not None:
            trainconfig["forecaster_checkpoint"] = args.checkpoint
        if args.downscaler_checkpoint is not None:
            trainconfig["downscaler_checkpoint"] = args.downscaler_checkpoint
        if not trainconfig.get("forecaster_checkpoint"):
            raise SystemExit("CombinedModule requires training.forecaster_checkpoint "
                             "(set via --checkpoint or in the YAML).")
        if not trainconfig.get("downscaler_checkpoint"):
            raise SystemExit("CombinedModule requires training.downscaler_checkpoint "
                             "(set via --downscaler_checkpoint or in the YAML).")
    else:
        if args.checkpoint is not None:
            trainconfig["checkpoint"] = args.checkpoint
        if args.downscaler_checkpoint is not None:
            print("  Warning: --downscaler_checkpoint ignored for TrainModule.")

    dataconfig["batch_size"] = 1
    dataconfig["epsilon_factor"] = 0
    dataconfig["forecast_lead_times"] = [1]

    seed = args.seed if args.seed is not None else trainconfig.get("seed", 43)
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"

    lead_time = int(args.lead_time)
    n_ensemble = int(args.n_ensemble)

    # ---- Time bounds: roll out starting from every day in
    # [start_date, end_date - lead_time]. Each rollout consumes lead_time+1 dates. ----
    
    
            

    dates_all = get_initialization_dates(args.year)   

    
    t_start = parse_date(dates_all[0].strftime("%Y-%m-%d"))
    t_end = parse_date(dates_all[-1].strftime("%Y-%m-%d"))
    last_start = t_end - timedelta(days=lead_time)
    if last_start < t_start:
        raise SystemExit(
            f"--end_date ({args.end_date}) must be at least lead_time={lead_time} days "
            f"after --start_date ({args.start_date}).")
    start_dates = dates_all
    print(f"Ensemble S2S rollout: {len(start_dates)} start dates from "
          f"{t_start.isoformat()} to {last_start.isoformat()} "
          f"(lead_time={lead_time}, n_ensemble={n_ensemble})")

    # Dataset must cover [t_start, t_end] inclusive.
    year_start = t_start.year
    year_end = t_end.year + 1
    dataset = GetDataset(dataconfig, year_start=year_start, year_end=year_end)

    model, is_combined = build_model(config, dataset, device)

    # CombinedModule outputs at the full (downscaler) resolution; TrainModule
    # outputs at the forecaster's downsampled resolution.
    if is_combined:
        out_nlat, out_nlon = model.nlat, model.nlon
        downsample_factor = 1
    else:
        downsample_factor = (
            model.downsample.downsample_factor if model.downsample is not None else 1
        )
        out_nlat = model.nlat // downsample_factor
        out_nlon = model.nlon // downsample_factor
    lat_coord, lon_coord = make_lat_lon(out_nlat, out_nlon)
    levels_hpa = list(np.array(dataset.levels).astype(int))

    print(f"Module: {'CombinedModule' if is_combined else 'TrainModule'}")
    print(f"Output resolution: {out_nlat} x {out_nlon} "
          f"(downsample factor {downsample_factor} from {model.nlat} x {model.nlon})")
    print(f"Pressure levels (hPa): {levels_hpa}")
    print(f"Surface vars:    {model.surface_variables}")
    print(f"Multilevel vars: {model.multilevel_variables}")
    print(f"Diagnostic vars: {model.diagnostic_variables}")

    os.makedirs(args.output_root, exist_ok=True)
    save_yaml(config, join(args.output_root, "config.yaml"))
    meta = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "lead_time": lead_time,
        "n_ensemble": n_ensemble,
        "n_start_dates": len(start_dates),
        "seed": seed,
        "module": "CombinedModule" if is_combined else "TrainModule",
        "horizontal_resolution": [out_nlat, out_nlon],
        "downsample_factor": downsample_factor,
        "levels_hpa": levels_hpa,
    }
    if is_combined:
        meta["forecaster_checkpoint"] = trainconfig.get("forecaster_checkpoint")
        meta["downscaler_checkpoint"] = trainconfig.get("downscaler_checkpoint")
    else:
        meta["checkpoint"] = trainconfig.get("checkpoint")
    save_yaml(meta, join(args.output_root, "meta.yaml"))
    print(f"Output root: {args.output_root}")

    overall_start = time.time()
    for k, sd in enumerate(start_dates):
        date_tag = f"{sd.year:04d}{sd.month:02d}{sd.day:02d}"
        seed_tag = f"seed{seed}"
        case_dir = join(args.output_root, date_tag)
        out_path = join(case_dir, f"ensemble_{date_tag}_{seed_tag}.nc")
        if os.path.exists(out_path):
            print(f"[{k+1}/{len(start_dates)}] {date_tag}: {out_path} already exists, skipping.",
                  flush=True)
            continue

        times = [sd + timedelta(days=i) for i in range(lead_time + 1)]

        t0 = time.time()
        print(f"[{k+1}/{len(start_dates)}] {date_tag}: rolling out "
              f"{n_ensemble} members x {lead_time} steps ...", flush=True)
        
        
        surf_arr, multi_arr, diag_arr = rollout_ensemble(
            model=model,
            dataset=dataset,
            times=times,
            n_ensemble=n_ensemble,
            device=device,
            is_combined=is_combined,
            num_data_workers=args.num_data_workers,
        )

        save_ensemble_nc(
            out_path=out_path,
            surf_arr=surf_arr,
            multi_arr=multi_arr,
            diag_arr=diag_arr,
            times=times,
            surf_vars=model.surface_variables,
            multi_vars=model.multilevel_variables,
            diag_vars=model.diagnostic_variables,
            lat=lat_coord,
            lon=lon_coord,
            levels_hpa=levels_hpa,
            start_date_iso=sd.isoformat(),
        )

        elapsed = time.time() - t0
        total_elapsed = time.time() - overall_start
        eta = total_elapsed / (k + 1) * (len(start_dates) - k - 1)
        print(f"  -> {out_path}  ({elapsed:.1f}s, eta {eta/3600:.2f}h)", flush=True)

    print(f"\nDone. {len(start_dates)} ensemble forecasts written to {args.output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="S2S ensemble rollout: one n-member, lead_time-day forecast "
                    "per start date in [start_date, end_date - lead_time].")
    parser.add_argument("--config", default="configs/SI_NCAR.yaml",
                        help="Path to model YAML config.")
    parser.add_argument("--checkpoint", default=None,
                        help="For TrainModule: full Lightning checkpoint (overrides "
                             "training.checkpoint). For CombinedModule: forecaster "
                             "checkpoint (overrides training.forecaster_checkpoint).")
    parser.add_argument("--downscaler_checkpoint", default=None,
                        help="(CombinedModule only) downscaler checkpoint; overrides "
                             "training.downscaler_checkpoint.")
    parser.add_argument("--start_date",  default="2019-01-01",
                        help="First IC date, YYYY-MM-DD (inclusive).")
    parser.add_argument("--end_date",  default="2020-12-31",
                        help="Final date (inclusive), YYYY-MM-DD. The last IC used "
                             "is end_date - lead_time.")
    parser.add_argument("--lead_time", type=int, default=45,
                        help="Forecast horizon in days (default 45).")
    parser.add_argument("--n_ensemble", type=int, default=4,
                        help="Ensemble size; members run as a single batched "
                             "forward per step (default 4).")
    parser.add_argument("--output_root", default="/glade/derecho/scratch/bgong/amip_s2s_ensembles",
                        help="Root directory; one subfolder per IC date is created.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: training.seed from config).")
    parser.add_argument("--devices", nargs="+", default=[], help="GPU device ids.")
    parser.add_argument("--num_data_workers", type=int, default=4,
                        help="DataLoader worker processes for forcing prefetch "
                             "(default 4).")
    parser.add_argument("--year", type=int, default=2019,
                        help="Year for the forecast (default 2019).")
    args = parser.parse_args()

    main(args)
