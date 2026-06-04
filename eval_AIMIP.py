"""eval_AIMIP.py — Generate an AIMIP-1 compliant AMIP-style rollout submission.

Rolls out a trained model from 1 Jan 1979 to 1 Jan 2025 with prescribed
SST/sea-ice forcing from ERA5 (5-member stochastic ensemble) and writes
per-variable monthly-mean and daily NetCDF files in the AIMIP-1 directory
layout described at https://github.com/ai2cm/AIMIP.

What's produced (per ensemble member r{N}i1p1f1):

  * Monthly-mean files for the full period:  197901 .. 202412
      ts, ps, tas, huss, uas, vas, pr, zg500, ta, hus, ua, va
      (3D fields output at AIMIP pressure levels: 1000, 850, 700, 500,
      250, 100, 50 hPa.)

  * Daily-mean files for two periods: 1 Jan 1979 .. 31 Dec 1979 (initial
    spin-up + first full year) and 1 Jan 2024 .. 31 Dec 2024 (out-of-sample).

Layout:

    {output_dir}/{institute}/{aimip_model_name}/aimip/r{N}i1p1f1/
        {Amon|day}/{var}/gn/{version}/
            {var}_{Amon|day}_{aimip_model_name}_aimip_r{N}i1p1f1_gn_{start}-{end}.nc

Run example:

    python eval_AIMIP.py --config=configs/combined_NCAR.yaml \
        --checkpoint=$CHECKPOINT_PATH \
        --output_dir=/glade/derecho/scratch/ayz/AIMIP_submission \
        --institute=CMU --aimip_model_name=CMU-AMIP \
        --ensemble_size=5
"""

import argparse
import os
import time
from datetime import timedelta
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


# ---------------------------------------------------------------------------
# AIMIP-1 output spec
# ---------------------------------------------------------------------------

G = 9.80665  # standard gravity (m/s^2), used for geopotential -> geopotential height

# AIMIP-1 7-level subset of standard rawinsonde reporting levels (hPa).
AIMIP_LEVELS_HPA = [1000, 850, 700, 500, 250, 100, 50]

# Surface (2-D) outputs. Maps internal model var -> CMIP metadata.
#   (cmip_name, units, long_name, standard_name)
SURFACE_OUTPUT_SPEC = {
    "skin_temperature":         ("ts",   "K",          "Surface Temperature",            "surface_temperature"),
    "surface_pressure":         ("ps",   "Pa",         "Surface Air Pressure",           "surface_air_pressure"),
    "2m_temperature":           ("tas",  "K",          "Near-Surface Air Temperature",   "air_temperature"),
    "2m_specific_humidity":     ("huss", "1",          "Near-Surface Specific Humidity", "specific_humidity"),
    "10m_u_component_of_wind":  ("uas",  "m s-1",      "Eastward Near-Surface Wind",     "eastward_wind"),
    "10m_v_component_of_wind":  ("vas",  "m s-1",      "Northward Near-Surface Wind",    "northward_wind"),
}

# Diagnostic (2-D) outputs.
DIAG_OUTPUT_SPEC = {
    "PRATEsfc_24h":             ("pr",   "kg m-2 s-1", "Precipitation",                  "precipitation_flux"),
}

# 3-D fields (output at AIMIP pressure levels).
MULTILEVEL_OUTPUT_SPEC = {
    "temperature":              ("ta",   "K",          "Air Temperature",                "air_temperature"),
    "specific_total_water":     ("hus",  "1",          "Specific Humidity",              "specific_humidity"),
    "u_component_of_wind":      ("ua",   "m s-1",      "Eastward Wind",                  "eastward_wind"),
    "v_component_of_wind":      ("va",   "m s-1",      "Northward Wind",                 "northward_wind"),
}

# 500 hPa geopotential height (single level, output as 2-D field).
ZG500_SPEC = ("zg500", "m", "Geopotential Height at 500 hPa", "geopotential_height")


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


def aimip_level_indices(model_levels):
    """Return indices into ``model_levels`` for each AIMIP requested level."""
    idx = []
    for lev in AIMIP_LEVELS_HPA:
        matches = [i for i, l in enumerate(model_levels) if int(l) == lev]
        if not matches:
            raise ValueError(f"Model is missing AIMIP level {lev} hPa "
                             f"(available: {list(model_levels)})")
        idx.append(matches[0])
    return idx


def daterange_days(start, end):
    """List of cftime datetimes from ``start`` (inclusive) to ``end`` (exclusive), step 1 day."""
    out = []
    t = start
    while t < end:
        out.append(t)
        t = t + timedelta(days=1)
    return out


def month_index(t, ref_year):
    """Months elapsed since (ref_year, 1) for cftime datetime ``t``."""
    return (t.year - ref_year) * 12 + (t.month - 1)


# ---------------------------------------------------------------------------
# Data loading (manual; bypasses dataset.__getitem__ so we can iterate to
# end of available data without lead-time padding restrictions).
# ---------------------------------------------------------------------------

def load_state(dataset, t):
    """Load full atmospheric state at time ``t`` (normalized).

    Returns:
        surface (1, c_sfc, nlat, nlon)
        upper_air (1, c_ua, nlev, nlat, nlon)
        diagnostic (1, c_diag, nlat, nlon)
    """
    raw = dataset._get_data(t, out=False)  # uses variable_list_in (includes diagnostic since diagnostic_input=True)
    upper_air, surface, diagnostic, _vbnd = dataset._reshape_and_mask_variables(raw, out=False)

    surface = dataset.surface_transform(surface).unsqueeze(0)
    upper_air = dataset.upper_air_transform(upper_air).unsqueeze(0)
    diagnostic = dataset.diagnostic_transform(diagnostic).unsqueeze(0)
    return surface, upper_air, diagnostic


def load_forcing(dataset, t):
    """Load varying boundary forcing at time ``t``.

    Returns:
        boundary (1, c_bnd, nlat, nlon) — normalized, with CO2 stripped if
            ``return_calendar`` is True (matching dataset._getitem_single_step).
        calendar (1, 3) or None — [seconds_of_day, day_of_year, co2_norm].
    """
    raw = torch.tensor(
        dataset._get_data(t, variable_list=dataset.varying_boundary_variables)
    ).to(torch.float32)
    raw = dataset._fill_mask(raw, dataset.varying_boundary_variables)
    boundary = dataset.boundary_transform(raw)  # (c, nlat, nlon)

    calendar = None
    if dataset.return_calendar:
        # First channel of varying_boundary_variables is global_mean_co2 (scalar field).
        co2 = boundary[0, 0, 0].clone()
        boundary = boundary[1:]
        sod, doy = dataset._compute_calendar(t)
        calendar = torch.tensor([[sod, doy, co2.item()]], dtype=torch.float32)

    return boundary.unsqueeze(0), calendar


# ---------------------------------------------------------------------------
# CMIP-style NetCDF writer
# ---------------------------------------------------------------------------

def write_var_netcdf(
    out_path: str,
    cmip_name: str,
    units: str,
    long_name: str,
    standard_name: str,
    data: np.ndarray,            # (T, nlat, nlon) or (T, n_aimip_levels, nlat, nlon)
    times,                       # list/array of cftime datetimes (length T)
    lat: np.ndarray,
    lon: np.ndarray,
    levels_hpa=None,             # optional, required iff data.ndim == 4
    frequency: str = "Amon",     # 'Amon' or 'day'
    institute: str = "",
    aimip_model_name: str = "",
    member: str = "r1i1p1f1",
    grid_label: str = "gn",
    experiment: str = "aimip",
):
    """Write one variable to a CF/CMIP-style NetCDF file."""
    is_3d = data.ndim == 4
    coords = {
        "time": ("time", np.asarray(times)),
        "lat":  ("lat",  lat),
        "lon":  ("lon",  lon),
    }
    if is_3d:
        assert levels_hpa is not None
        coords["plev"] = ("plev", np.asarray(levels_hpa, dtype=np.float64) * 100.0)  # store in Pa
        dims = ("time", "plev", "lat", "lon")
    else:
        dims = ("time", "lat", "lon")

    da = xr.DataArray(
        data.astype(np.float32),
        dims=dims,
        coords=coords,
        name=cmip_name,
        attrs={
            "units": units,
            "long_name": long_name,
            "standard_name": standard_name,
            "cell_methods": "time: mean" if frequency == "Amon" else "time: mean",
        },
    )

    # Coordinate attributes
    da["lat"].attrs.update(units="degrees_north", standard_name="latitude", long_name="latitude", axis="Y")
    da["lon"].attrs.update(units="degrees_east",  standard_name="longitude", long_name="longitude", axis="X")
    if is_3d:
        da["plev"].attrs.update(units="Pa", standard_name="air_pressure", long_name="pressure", axis="Z", positive="down")

    ds = da.to_dataset()
    ds.attrs.update(
        title=f"AIMIP-1 {experiment} simulation: {cmip_name} ({frequency})",
        institution=institute,
        source=aimip_model_name,
        experiment_id=experiment,
        variant_label=member,
        frequency=frequency,
        grid_label=grid_label,
        Conventions="CF-1.8",
        license="CC-BY 4.0",
        product="model-output",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # We use the h5netcdf engine because the project's other dependencies pull
    # in h5py with an HDF5 ABI that conflicts with netCDF4's bundled HDF5;
    # writing through the netCDF4 engine in the same process triggers spurious
    # `RuntimeError: NetCDF: HDF error`. h5netcdf goes through the same h5py
    # that everything else in the repo uses.
    encoding = {
        cmip_name: {"compression": "gzip", "compression_opts": 4, "dtype": "float32"},
        # CMIP-standard time anchor for reproducibility.
        "time": {"dtype": "float64",
                 "units": "days since 1850-01-01 00:00:00",
                 "calendar": "standard"},
    }
    ds.to_netcdf(out_path, format="NETCDF4", engine="h5netcdf",
                 encoding=encoding, unlimited_dims=["time"])


def aimip_paths(base_dir, institute, aimip_model_name, experiment, member, frequency, var, version, start, end):
    """Build the AIMIP-1 directory + filename for one variable file."""
    rel_dir = join(institute, aimip_model_name, experiment, member, frequency, var, "gn", version)
    fname = f"{var}_{frequency}_{aimip_model_name}_{experiment}_{member}_gn_{start}-{end}.nc"
    return join(base_dir, rel_dir, fname)


# ---------------------------------------------------------------------------
# Buffers
# ---------------------------------------------------------------------------

class RolloutBuffers:
    """Per-ensemble-member running accumulators for monthly + daily output.

    Monthly: for every CMIP variable, accumulate a sum and a count per month
    so we can compute monthly means at the end of the run. Stored as float32
    on CPU (fits in memory: ~5 GB total for one member at 180x360x36 fields x 552 months).

    Daily: for every CMIP variable, store the daily snapshot only for the
    AIMIP-requested daily-output periods (1979 + 2024). Other days are dropped.
    """

    def __init__(self, n_months: int, daily_dates_per_period: dict, nlat: int, nlon: int, n_aimip_levels: int):
        self.n_months = n_months
        self.daily_dates = daily_dates_per_period  # {period_label: [datetimes...]}
        self.daily_index = {
            label: {t: i for i, t in enumerate(dates)}
            for label, dates in daily_dates_per_period.items()
        }
        self.nlat, self.nlon = nlat, nlon
        self.n_lev = n_aimip_levels

        # Monthly accumulators
        self.monthly_sum_2d = {}  # cmip_name -> (n_months, nlat, nlon)
        self.monthly_sum_3d = {}  # cmip_name -> (n_months, n_lev, nlat, nlon)
        self.monthly_count = np.zeros(n_months, dtype=np.int32)

        # Daily snapshots — one buffer per period
        self.daily_2d = {label: {} for label in daily_dates_per_period}
        self.daily_3d = {label: {} for label in daily_dates_per_period}

    def _ensure_2d(self, cmip_name):
        if cmip_name not in self.monthly_sum_2d:
            self.monthly_sum_2d[cmip_name] = np.zeros((self.n_months, self.nlat, self.nlon), dtype=np.float32)
            for label, dates in self.daily_dates.items():
                self.daily_2d[label][cmip_name] = np.zeros((len(dates), self.nlat, self.nlon), dtype=np.float32)

    def _ensure_3d(self, cmip_name):
        if cmip_name not in self.monthly_sum_3d:
            self.monthly_sum_3d[cmip_name] = np.zeros((self.n_months, self.n_lev, self.nlat, self.nlon), dtype=np.float32)
            for label, dates in self.daily_dates.items():
                self.daily_3d[label][cmip_name] = np.zeros((len(dates), self.n_lev, self.nlat, self.nlon), dtype=np.float32)

    def add_2d(self, cmip_name: str, t, m_idx: int, field: np.ndarray):
        """Add a daily 2D snapshot for time t into monthly + (if applicable) daily buffer."""
        self._ensure_2d(cmip_name)
        self.monthly_sum_2d[cmip_name][m_idx] += field
        for label, idx_map in self.daily_index.items():
            if t in idx_map:
                self.daily_2d[label][cmip_name][idx_map[t]] = field
                break

    def add_3d(self, cmip_name: str, t, m_idx: int, field: np.ndarray):
        """Add a daily 3D snapshot (n_lev, nlat, nlon)."""
        self._ensure_3d(cmip_name)
        self.monthly_sum_3d[cmip_name][m_idx] += field
        for label, idx_map in self.daily_index.items():
            if t in idx_map:
                self.daily_3d[label][cmip_name][idx_map[t]] = field
                break

    def increment_count(self, m_idx: int):
        self.monthly_count[m_idx] += 1

    def finalize_monthly(self):
        counts = np.maximum(self.monthly_count, 1).astype(np.float32)
        out_2d = {k: v / counts.reshape(-1, 1, 1) for k, v in self.monthly_sum_2d.items()}
        out_3d = {k: v / counts.reshape(-1, 1, 1, 1) for k, v in self.monthly_sum_3d.items()}
        return out_2d, out_3d


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def build_model(config, dataset, checkpoint, downscaler_checkpoint, device):
    is_combined = config["model"].get("model_name", "") == "Combined"
    if is_combined:
        # CombinedModule loads both checkpoints itself from config['training'].
        # Allow CLI overrides; otherwise fall back to whatever the YAML config sets.
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

def rollout_member(
    member_idx: int,
    seed: int,
    model,
    is_combined: bool,
    dataset,
    config,
    args,
    times,                     # full list of cftime datetimes (IC + predictions)
    daily_periods,             # {label: (start_datetime, end_datetime, year_for_filename)}
    base_output_dir,
    aimip_level_idx,
    lat_coord,
    lon_coord,
    device,
):
    """Run a single ensemble member end-to-end and write its NetCDF outputs."""
    print(f"\n=== Ensemble member r{member_idx+1}i1p1f1 (seed={seed}) ===", flush=True)
    seed_everything(seed)

    nlat = model.nlat
    nlon = model.nlon
    n_levels = model.nlevels
    n_aimip = len(aimip_level_idx)

    # ------------------------- Buffers ----------------------------------
    ref_year = times[0].year  # 1979
    end_excl = times[-1] + timedelta(days=1)  # one past last
    n_months = (end_excl.year - ref_year) * 12 + (end_excl.month - 1)
    if n_months <= 0:
        n_months = 1

    # Daily output dates for each period (clipped to available rollout times)
    daily_dates_per_period = {}
    for label, (p_start, p_end, _yr) in daily_periods.items():
        dates = [t for t in times if p_start <= t < p_end]
        daily_dates_per_period[label] = dates
        print(f"  {label}: {len(dates)} daily timesteps "
              f"({dates[0].isoformat() if dates else 'empty'} .. "
              f"{dates[-1].isoformat() if dates else 'empty'})")

    buf = RolloutBuffers(n_months, daily_dates_per_period, nlat, nlon, n_aimip)

    # ------------------------- Initial condition ------------------------
    t0 = times[0]
    surface_t, upper_air_t, diagnostic_t = load_state(dataset, t0)
    surface_t = surface_t.to(device)
    upper_air_t = upper_air_t.to(device)
    diagnostic_t = diagnostic_t.to(device)

    # Record IC into buffer (so the first day of the spinup year is included).
    accumulate_step(buf, model, dataset, t0, ref_year, surface_t, upper_air_t, diagnostic_t,
                    aimip_level_idx, surf_pred=None, multi_pred=None, diag_pred=None,
                    is_initial=True)

    invariant = model.invariant_input.to(device)  # (1, c_inv, nlat, nlon)

    # Preprocess IC into low-res forecaster input (CombinedModule downsamples here).
    x = model.preprocess(surface_t, upper_air_t, diagnostic_t)

    # ------------------------- Rollout loop -----------------------------
    n_steps = len(times) - 1  # number of model forward calls
    log_every = max(1, n_steps // 200)
    start_clock = time.time()

    with torch.no_grad():
        for step_idx in range(n_steps):
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

            # y_last is the cleaner x-prediction (full-res for CombinedModule).
            surf_pred, multi_pred, diag_pred = disassemble_input(y_last, nlevels=n_levels)

            accumulate_step(buf, model, dataset, t_out, ref_year,
                            None, None, None,
                            aimip_level_idx,
                            surf_pred=surf_pred, multi_pred=multi_pred, diag_pred=diag_pred,
                            is_initial=False)

            x = y  # autoregressive: feed Euler-updated state forward

            if (step_idx + 1) % log_every == 0 or step_idx == n_steps - 1:
                elapsed = time.time() - start_clock
                rate = (step_idx + 1) / elapsed
                eta = (n_steps - step_idx - 1) / max(rate, 1e-6)
                print(f"  step {step_idx+1}/{n_steps}  t={t_out.isoformat()}  "
                      f"{rate:.2f} steps/s  eta {eta/3600:.2f}h",
                      flush=True)

    # ------------------------- Write output ------------------------------
    monthly_2d, monthly_3d = buf.finalize_monthly()
    member = f"r{member_idx+1}i1p1f1"
    version = args.version

    # Monthly time axis: middle of each month (15th 12:00).
    monthly_times = []
    for m in range(n_months):
        yr = ref_year + (m // 12)
        mo = (m % 12) + 1
        monthly_times.append(cftime.DatetimeGregorian(yr, mo, 15, 12, 0, 0, has_year_zero=False))

    monthly_start = f"{ref_year:04d}{1:02d}"
    monthly_end_year = ref_year + (n_months - 1) // 12
    monthly_end_month = ((n_months - 1) % 12) + 1
    monthly_end = f"{monthly_end_year:04d}{monthly_end_month:02d}"

    print(f"  Writing monthly NetCDFs ({monthly_start}-{monthly_end}) ...", flush=True)
    for cmip_name, arr in monthly_2d.items():
        spec = _spec_by_cmip(cmip_name)
        out = aimip_paths(base_output_dir, args.institute, args.aimip_model_name,
                          "aimip", member, "Amon", cmip_name, version,
                          monthly_start, monthly_end)
        write_var_netcdf(out, cmip_name, spec[1], spec[2], spec[3], arr,
                         monthly_times, lat_coord, lon_coord, levels_hpa=None,
                         frequency="Amon", institute=args.institute,
                         aimip_model_name=args.aimip_model_name, member=member)

    for cmip_name, arr in monthly_3d.items():
        spec = _spec_by_cmip(cmip_name)
        out = aimip_paths(base_output_dir, args.institute, args.aimip_model_name,
                          "aimip", member, "Amon", cmip_name, version,
                          monthly_start, monthly_end)
        write_var_netcdf(out, cmip_name, spec[1], spec[2], spec[3], arr,
                         monthly_times, lat_coord, lon_coord, levels_hpa=AIMIP_LEVELS_HPA,
                         frequency="Amon", institute=args.institute,
                         aimip_model_name=args.aimip_model_name, member=member)

    # Daily NetCDFs (one period at a time)
    for label, (p_start, p_end, year_tag) in daily_periods.items():
        dates = daily_dates_per_period[label]
        if not dates:
            continue
        # Nominal mid-day timestamp for each daily mean (12:00 UTC).
        daily_times = [cftime.DatetimeGregorian(t.year, t.month, t.day, 12, 0, 0,
                                                has_year_zero=False) for t in dates]
        d_start = f"{dates[0].year:04d}{dates[0].month:02d}{dates[0].day:02d}"
        d_end   = f"{dates[-1].year:04d}{dates[-1].month:02d}{dates[-1].day:02d}"
        print(f"  Writing daily NetCDFs ({label}: {d_start}-{d_end}) ...", flush=True)

        for cmip_name, arr in buf.daily_2d[label].items():
            spec = _spec_by_cmip(cmip_name)
            out = aimip_paths(base_output_dir, args.institute, args.aimip_model_name,
                              "aimip", member, "day", cmip_name, version,
                              d_start, d_end)
            write_var_netcdf(out, cmip_name, spec[1], spec[2], spec[3], arr,
                             daily_times, lat_coord, lon_coord, levels_hpa=None,
                             frequency="day", institute=args.institute,
                             aimip_model_name=args.aimip_model_name, member=member)

        for cmip_name, arr in buf.daily_3d[label].items():
            spec = _spec_by_cmip(cmip_name)
            out = aimip_paths(base_output_dir, args.institute, args.aimip_model_name,
                              "aimip", member, "day", cmip_name, version,
                              d_start, d_end)
            write_var_netcdf(out, cmip_name, spec[1], spec[2], spec[3], arr,
                             daily_times, lat_coord, lon_coord, levels_hpa=AIMIP_LEVELS_HPA,
                             frequency="day", institute=args.institute,
                             aimip_model_name=args.aimip_model_name, member=member)


def _spec_by_cmip(cmip_name):
    """Look up (internal, units, long_name, standard_name) for a CMIP variable."""
    for table in (SURFACE_OUTPUT_SPEC, DIAG_OUTPUT_SPEC, MULTILEVEL_OUTPUT_SPEC):
        for internal, spec in table.items():
            if spec[0] == cmip_name:
                return (internal,) + spec[1:]
    if cmip_name == ZG500_SPEC[0]:
        return ("geopotential",) + ZG500_SPEC[1:]
    raise KeyError(cmip_name)


def accumulate_step(buf, model, dataset, t, ref_year,
                    surf_norm, multi_norm, diag_norm,
                    aimip_level_idx,
                    surf_pred=None, multi_pred=None, diag_pred=None,
                    is_initial: bool = False):
    """Denormalize a single timestep's state and add it to the running buffers.

    For ``is_initial`` (the IC) we accept normalized inputs directly (load_state
    output). Otherwise we denormalize the model's predictions.
    """
    m_idx = month_index(t, ref_year)
    if m_idx < 0 or m_idx >= buf.n_months:
        return

    if is_initial:
        # Use the IC values themselves to populate t0.
        surf_d = dataset.surface_inv_transform(surf_norm)
        multi_d = dataset.upper_air_inv_transform(multi_norm)
        diag_d = dataset.diagnostic_inv_transform(diag_norm)
    else:
        surf_d = dataset.surface_inv_transform(surf_pred)
        multi_d = dataset.upper_air_inv_transform(multi_pred)
        diag_d = dataset.diagnostic_inv_transform(diag_pred)

    surf_np = surf_d[0].detach().cpu().numpy()    # (c_sfc, nlat, nlon)
    multi_np = multi_d[0].detach().cpu().numpy()  # (c_ua, n_lev_full, nlat, nlon)
    diag_np = diag_d[0].detach().cpu().numpy()    # (c_diag, nlat, nlon)

    # Surface variables
    for c, internal_name in enumerate(model.surface_variables):
        if internal_name in SURFACE_OUTPUT_SPEC:
            cmip_name = SURFACE_OUTPUT_SPEC[internal_name][0]
            buf.add_2d(cmip_name, t, m_idx, surf_np[c])

    # Diagnostic variables
    for c, internal_name in enumerate(model.diagnostic_variables):
        if internal_name in DIAG_OUTPUT_SPEC:
            cmip_name = DIAG_OUTPUT_SPEC[internal_name][0]
            buf.add_2d(cmip_name, t, m_idx, diag_np[c])

    # Multi-level variables (subset levels to AIMIP request)
    for c, internal_name in enumerate(model.multilevel_variables):
        if internal_name in MULTILEVEL_OUTPUT_SPEC:
            cmip_name = MULTILEVEL_OUTPUT_SPEC[internal_name][0]
            field = multi_np[c, aimip_level_idx]  # (n_aimip, nlat, nlon)
            buf.add_3d(cmip_name, t, m_idx, field)
        if internal_name == "geopotential":
            # zg500 = geopotential at 500 hPa / g
            try:
                lev500 = list(np.array(dataset.levels).astype(int)).index(500)
            except ValueError:
                lev500 = None
            if lev500 is not None:
                buf.add_2d(ZG500_SPEC[0], t, m_idx, multi_np[c, lev500] / G)

    buf.increment_count(m_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    config = get_yaml(args.config)
    dataconfig = config["data"]
    trainconfig = config["training"]

    if args.checkpoint is not None:
        trainconfig["checkpoint"] = args.checkpoint
        # CombinedModule loads forecaster_checkpoint from this key.
        trainconfig["forecaster_checkpoint"] = args.checkpoint

    if len(args.devices) > 0:
        trainconfig["devices"] = [int(d) for d in args.devices]

    # The dataset only loads files when asked, so single-batch config is fine.
    dataconfig["batch_size"] = 1
    # Disable any input noise for inference reproducibility.
    dataconfig["epsilon_factor"] = 0
    # Forecast lead times not used (we drive the rollout manually) but reset to
    # avoid validate=True surprises if dataset.__getitem__ is ever called.
    dataconfig["forecast_lead_times"] = [1]

    torch.set_float32_matmul_precision("high")

    device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"

    # Need a long-enough year span for the dataset to allow loading any time
    # in [1979-01-01, 2024-12-31]. ``year_end`` is exclusive, so 2025 covers
    # all of 2024 (and the dataset never reads 2025_*.h5).
    dataset = GetDataset(dataconfig, year_start=args.start_year, year_end=args.end_year)

    model, is_combined = build_model(config, dataset, args.checkpoint,
                                     args.downscaler_checkpoint, device)

    nlat = model.nlat
    nlon = model.nlon
    lat_coord, lon_coord = make_lat_lon(nlat, nlon)
    aimip_level_idx = aimip_level_indices(np.array(dataset.levels).astype(int))

    # ---- Output directory ----
    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    save_yaml(config, join(base_output_dir, f"config_{args.aimip_model_name}.yaml"))

    # ---- Time axis ----
    # IC at start_year-01-01; final prediction at end_year-01-01 (per AIMIP
    # spec: "run 46 years until 00 UTC 1 Jan 2025"). Last forcing read is at
    # (end_year-1)-12-31. ``times`` length = N_predictions + 1 (IC).
    t_start = cftime.DatetimeGregorian(args.start_year, 1, 1, has_year_zero=False)
    t_final = cftime.DatetimeGregorian(args.end_year,   1, 1, has_year_zero=False)
    times = daterange_days(t_start, t_final + timedelta(days=1))
    print(f"Rollout: {t_start.isoformat()} -> {t_final.isoformat()} "
          f"({len(times)} dates, {len(times)-1} model steps)")
    print(f"AIMIP pressure levels: {AIMIP_LEVELS_HPA} hPa  (model levels {dataset.levels})")

    # ---- Daily output periods (per AIMIP-1, adapted to start_year=1979) ----
    spinup_year = args.start_year
    test_year = args.end_year - 1
    daily_periods = {
        f"spinup_{spinup_year}": (
            cftime.DatetimeGregorian(spinup_year, 1, 1, has_year_zero=False),
            cftime.DatetimeGregorian(spinup_year + 1, 1, 1, has_year_zero=False),
            spinup_year,
        ),
        f"test_{test_year}": (
            cftime.DatetimeGregorian(test_year, 1, 1, has_year_zero=False),
            cftime.DatetimeGregorian(test_year + 1, 1, 1, has_year_zero=False),
            test_year,
        ),
    }

    # ---- Run ensemble ----
    base_seed = trainconfig.get("seed", 42)
    for member_idx in range(args.ensemble_size):
        seed = base_seed + member_idx
        rollout_member(
            member_idx=member_idx,
            seed=seed,
            model=model,
            is_combined=is_combined,
            dataset=dataset,
            config=config,
            args=args,
            times=times,
            daily_periods=daily_periods,
            base_output_dir=base_output_dir,
            aimip_level_idx=aimip_level_idx,
            lat_coord=lat_coord,
            lon_coord=lon_coord,
            device=device,
        )

    print("\nAll ensemble members complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIMIP-1 rollout")
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint. For TrainModule: full state dict. "
                             "For CombinedModule: forecaster checkpoint (overrides YAML).")
    parser.add_argument("--downscaler_checkpoint", default=None,
                        help="(CombinedModule only) downscaler checkpoint override.")
    parser.add_argument("--output_dir", required=True, help="Root directory for AIMIP-1 submission tree")
    parser.add_argument("--institute", default="CMU", help="Institute label (CMIP)")
    parser.add_argument("--aimip_model_name", default="CMU-AMIP",
                        help="MMM model name in CMIP filename (e.g. ACE2-ERA5). No underscores.")
    parser.add_argument("--ensemble_size", type=int, default=5,
                        help="Number of stochastic ensemble members to run")
    parser.add_argument("--start_year", type=int, default=1979,
                        help="First year of rollout (IC at year-01-01 00:00). Default 1979.")
    parser.add_argument("--end_year", type=int, default=2025,
                        help="One past last year of rollout (last prediction at end_year-01-01). Default 2025.")
    parser.add_argument("--version", default=time.strftime("v%Y%m%d"),
                        help="Version label for AIMIP directory (e.g. v20251130)")
    parser.add_argument("--devices", nargs="+", default=[], help="GPU device ids")
    args = parser.parse_args()

    if "_" in args.aimip_model_name:
        raise SystemExit("AIMIP model name must not contain underscores (use dashes).")

    main(args)
