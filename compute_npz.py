#!/usr/bin/env python -u
"""Compute normalization statistics (mean, std) for the ERA5 AMIP dataset.

Parallel version: every (variable, year) pair is submitted to a
ProcessPoolExecutor simultaneously, so all workers stay busy from start
to finish.  Results are aggregated on the main process once all futures
complete, then saved in a single pass.

Data layout expected at root_dir:
  forcing/
    {const_var}/180x360.nc               (time-invariant constants)
    {varying_var}/{year}_180x360.nc      (non-repeating varying BCs)
  prognostic/
    {single_var}/{year}_180x360.nc
    3D_PL/{var}/{year}_180x360.nc
  diagnostic/
    {diag_var}/{year}_180x360.nc

Outputs (saved to save_dir):
  normalize_mean.npz / normalize_std.npz
  normalize_diff_mean_{N}.npz / normalize_diff_std_{N}.npz  (--lead_time N)
"""

import os
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Variable lists
# ---------------------------------------------------------------------------

CONSTANTS = [
]

VARYING_BOUNDARY = [
]

SINGLE_LEVEL_VARS_PROGNOSTIC = [
]

SINGLE_LEVEL_VARS_DIAGNOSTIC = [
]

PRESSURE_LEVEL_VARS = [
    "specific_humidity",
]

# 1-D (time-only) variables whose annual files live at
# forcing/{var}/forcing_{year}.nc.  At each timestep the scalar value is
# broadcast to a constant (lat, lon) field before being stored in the h5
# files.  Normalization statistics are computed from the scalar time series
# (no spatial broadcasting required).
CONSTANT_SCALARS = [
]

MASK_FILL_DEFAULT = 0.0
MASK_FILL_OVERRIDES = {
    "sea_surface_temperature": 270.0,
    "skin_temperature": 270.0,
    "2m_temperature": 270.0,
    "2m_dewpoint_temperature": 270.0,
    "soil_temperature_level_1": 270.0,
    "soil_temperature_level_2": 270.0,
    "soil_temperature_level_3": 270.0,
    "soil_temperature_level_4": 270.0,
}

LEVEL_DIM = "level"


# ---------------------------------------------------------------------------
# Helpers (must be importable at module level for pickling)
# ---------------------------------------------------------------------------

def _open_dataset(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xr.open_dataset(path)


def _get_nc_var_name(ds, dir_name):
    data_vars = list(ds.data_vars)
    if dir_name in data_vars:
        return dir_name
    if len(data_vars) == 1:
        return data_vars[0]
    raise ValueError(
        f"Cannot determine nc variable name for '{dir_name}'; "
        f"candidates: {data_vars}"
    )


def _get_data_path(root_dir, var, year):
    fname = f"{year}_180x360.nc"
    if var in VARYING_BOUNDARY:
        return os.path.join(root_dir, "forcing", var, fname)
    if var in SINGLE_LEVEL_VARS_PROGNOSTIC:
        return os.path.join(root_dir, "prognostic", var, fname)
    if var in SINGLE_LEVEL_VARS_DIAGNOSTIC:
        return os.path.join(root_dir, "diagnostic", var, fname)
    if var in PRESSURE_LEVEL_VARS:
        return os.path.join(root_dir, "prognostic", "3D_PL", var, fname)
    if var in CONSTANT_SCALARS:
        return os.path.join(root_dir, "forcing", var, f"forcing_{year}.nc")
    raise ValueError(f"Unknown variable: '{var}'")


def _combine_mean_std(mean_list, std_list):
    """Law of total variance: E[Var] + Var[E] = Var(total)."""
    m = np.asarray(mean_list, dtype=np.float64)
    s = np.asarray(std_list, dtype=np.float64)
    mean = m.mean()
    variance = (s ** 2).mean() + (m ** 2).mean() - mean ** 2
    return float(mean), float(np.sqrt(max(variance, 0.0)))


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess)
# ---------------------------------------------------------------------------

def _process_var_year(args):
    """Compute per-chunk mean/std for one (variable, year) pair.

    Returns
    -------
    dict
        Maps each normalization key (e.g. ``"temperature_500.0"``) to a
        tuple ``([chunk_means], [chunk_stds])``.  Empty dict on failure.
    """
    var, year, root_dir, backup_root_dir, chunk_size, steps, fill = args

    path = _get_data_path(root_dir, var, year)
    if not os.path.exists(path) and backup_root_dir is not None:
        path = _get_data_path(backup_root_dir, var, year)
    if not os.path.exists(path):
        return {}

    try:
        ds = _open_dataset(path)
        nc_var = _get_nc_var_name(ds, var)
        n_time = len(ds["time"])
        n_chunks = n_time // chunk_size + (1 if n_time % chunk_size else 0)

        result = {}

        if var in PRESSURE_LEVEL_VARS:
            levels = ds[LEVEL_DIM].values
            # Pre-allocate per-level accumulators.
            level_means = {f"{var}_{float(l)}": [] for l in levels}
            level_stds  = {f"{var}_{float(l)}": [] for l in levels}

            for chunk_id in range(n_chunks):
                t_slice = slice(chunk_id * chunk_size, (chunk_id + 1) * chunk_size)
                # Shape: (T, n_levels, lat, lon)
                chunk = ds[nc_var].isel(time=t_slice).values.astype(np.float64)
                for i, level in enumerate(levels):
                    lev = chunk[:, i]          # (T, lat, lon)
                    if steps is not None:
                        if len(lev) <= steps:
                            continue
                        lev = lev[steps:] - lev[:-steps]
                    key = f"{var}_{float(level)}"
                    level_means[key].append(np.nanmean(lev))
                    level_stds[key].append(np.nanstd(lev))

            result = {k: (level_means[k], level_stds[k]) for k in level_means}

        elif var in CONSTANT_SCALARS:
            # 1-D (time-only) data — no lat/lon dims.
            # Spatial broadcasting to (lat, lon) is handled at h5-write time.
            means, stds = [], []
            for chunk_id in range(n_chunks):
                t_slice = slice(chunk_id * chunk_size, (chunk_id + 1) * chunk_size)
                chunk = ds[nc_var].isel(time=t_slice).values.astype(np.float64)
                if steps is not None:
                    if len(chunk) <= steps:
                        continue
                    chunk = chunk[steps:] - chunk[:-steps]
                means.append(np.nanmean(chunk))
                stds.append(np.nanstd(chunk))
            result = {var: (means, stds)}

        else:
            means, stds = [], []
            for chunk_id in range(n_chunks):
                t_slice = slice(chunk_id * chunk_size, (chunk_id + 1) * chunk_size)
                chunk = ds[nc_var].isel(time=t_slice).values.astype(np.float64)
                chunk = np.where(np.isnan(chunk), fill, chunk)
                if steps is not None:
                    if len(chunk) <= steps:
                        continue
                    chunk = chunk[steps:] - chunk[:-steps]
                means.append(np.nanmean(chunk))
                stds.append(np.nanstd(chunk))
            result = {var: (means, stds)}

        ds.close()
        return result

    except Exception as exc:  # noqa: BLE001
        # Surface the error on the main process via the future's exception,
        # but keep the pool alive for other tasks.
        raise RuntimeError(f"Failed processing {var} year {year}: {exc}") from exc


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for the ERA5 AMIP dataset (parallel)."
    )
    parser.add_argument(
        "--root_dir", type=str, required=True,
        help="Root directory of the ERA5 AMIP data."
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help="Directory where normalization .npz files will be written."
    )
    parser.add_argument(
        "--start_year", type=int, default=1979,
        help="First year to include (default: 1979)."
    )
    parser.add_argument(
        "--end_year", type=int, default=2024,
        help="Last year to include, inclusive (default: 2024)."
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100,
        help="Timesteps loaded per read (controls peak memory per worker; default: 100)."
    )
    parser.add_argument(
        "--lead_time", type=int, default=None,
        help="Lead time in hours for difference normalization. Omit for absolute stats."
    )
    parser.add_argument(
        "--data_frequency", type=int, default=6,
        help="Data time step in hours (default: 6)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=min(32, os.cpu_count() or 1),
        help=(
            "Number of parallel worker processes "
            f"(default: min(32, cpu_count) = {min(32, os.cpu_count() or 1)})."
        )
    )
    parser.add_argument(
        "--backup_root_dir", type=str, default=None,
        help=(
            "Optional backup root directory.  When a file is not found under "
            "--root_dir, the script will search for it here before skipping."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    root_dir        = args.root_dir
    backup_root_dir = args.backup_root_dir
    save_dir        = args.save_dir
    years           = list(range(args.start_year, args.end_year + 1))
    chunk_size      = args.chunk_size
    lead_time       = args.lead_time
    data_freq       = args.data_frequency
    num_workers     = args.num_workers

    os.makedirs(save_dir, exist_ok=True)

    mean_file = (
        f"normalize_diff_mean_{lead_time}.npz" if lead_time is not None
        else "normalize_mean.npz"
    )
    std_file = (
        f"normalize_diff_std_{lead_time}.npz" if lead_time is not None
        else "normalize_std.npz"
    )

    steps = lead_time // data_freq if lead_time is not None else None

    # Discover pressure levels from the first available year.
    sample_path = _get_data_path(root_dir, PRESSURE_LEVEL_VARS[0], years[0])
    if not os.path.exists(sample_path) and backup_root_dir is not None:
        sample_path = _get_data_path(backup_root_dir, PRESSURE_LEVEL_VARS[0], years[0])
    sample_ds = _open_dataset(sample_path)
    pressure_levels = sample_ds[LEVEL_DIM].values
    sample_ds.close()
    print(f"Pressure levels ({len(pressure_levels)}): {pressure_levels}")
    print(f"Using {num_workers} worker processes.")

    all_single_vars = (
        VARYING_BOUNDARY
        + SINGLE_LEVEL_VARS_PROGNOSTIC
        + SINGLE_LEVEL_VARS_DIAGNOSTIC
    )

    # Build the full task list.  Each task is a single (var, year) pair.
    tasks = []
    for var in all_single_vars:
        fill = MASK_FILL_OVERRIDES.get(var, MASK_FILL_DEFAULT)
        for year in years:
            tasks.append((var, year, root_dir, backup_root_dir, chunk_size, steps, fill))
    for var in PRESSURE_LEVEL_VARS:
        for year in years:
            tasks.append((var, year, root_dir, backup_root_dir, chunk_size, steps, MASK_FILL_DEFAULT))
    for var in CONSTANT_SCALARS:
        fill = MASK_FILL_OVERRIDES.get(var, MASK_FILL_DEFAULT)
        for year in years:
            tasks.append((var, year, root_dir, backup_root_dir, chunk_size, steps, fill))

    print(f"Total tasks: {len(tasks)}  ({len(all_single_vars)} single-level vars "
          f"+ {len(PRESSURE_LEVEL_VARS)} pressure-level vars"
          f"+ {len(CONSTANT_SCALARS)} constant-scalar vars) × {len(years)} years")

    # Accumulator: key -> ([chunk_means], [chunk_stds])
    accumulated: dict[str, tuple[list, list]] = {}

    # ---------------------------------------------------------------------------
    # Execution (serial if num_workers <= 1, else parallel)
    # ---------------------------------------------------------------------------
    if num_workers <= 1:
        with tqdm(total=len(tasks), desc="(var, year) tasks") as pbar:
            for task in tasks:
                var, year = task[0], task[1]
                pbar.set_postfix_str(f"{var} {year}", refresh=False)
                try:
                    result = _process_var_year(task)
                    for key, (means, stds) in result.items():
                        if key not in accumulated:
                            accumulated[key] = ([], [])
                        accumulated[key][0].extend(means)
                        accumulated[key][1].extend(stds)
                except RuntimeError as exc:
                    tqdm.write(f"  WARNING: {exc}")
                pbar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {
                executor.submit(_process_var_year, task): task
                for task in tasks
            }

            with tqdm(total=len(tasks), desc="(var, year) tasks") as pbar:
                for future in as_completed(future_to_task):
                    var, year = future_to_task[future][0], future_to_task[future][1]
                    pbar.set_postfix_str(f"{var} {year}", refresh=False)
                    try:
                        result = future.result()  # dict{key: ([means], [stds])}
                        for key, (means, stds) in result.items():
                            if key not in accumulated:
                                accumulated[key] = ([], [])
                            accumulated[key][0].extend(means)
                            accumulated[key][1].extend(stds)
                    except (RuntimeError, Exception) as exc:
                        tqdm.write(f"  WARNING: {exc}")
                    pbar.update(1)

    # ---------------------------------------------------------------------------
    # Compute final statistics
    # ---------------------------------------------------------------------------
    print("Combining per-chunk statistics...")
    normalize_mean: dict[str, np.ndarray] = {}
    normalize_std:  dict[str, np.ndarray] = {}

    for key, (means, stds) in accumulated.items():
        if means:
            m, s = _combine_mean_std(means, stds)
            normalize_mean[key] = np.array([m])
            normalize_std[key]  = np.array([s])
        else:
            normalize_mean[key] = np.array([np.nan])
            normalize_std[key]  = np.array([np.nan])

    # ---------------------------------------------------------------------------
    # Time-invariant constants (tiny — handled on main process)
    # ---------------------------------------------------------------------------
    print("Computing constant field statistics...")
    for var in CONSTANTS:
        if steps is not None:
            normalize_mean[var] = np.array([0.0])
            normalize_std[var]  = np.array([0.0])
        else:
            path = os.path.join(root_dir, "forcing", var, "180x360.nc")
            if not os.path.exists(path) and backup_root_dir is not None:
                path = os.path.join(backup_root_dir, "forcing", var, "180x360.nc")
            if not os.path.exists(path):
                print(f"  Warning: missing constant file {path}, skipping.")
                normalize_mean[var] = np.array([np.nan])
                normalize_std[var]  = np.array([np.nan])
                continue
            ds = _open_dataset(path)
            nc_var = _get_nc_var_name(ds, var)
            data = ds[nc_var].values.astype(np.float64)
            data = np.where(np.isnan(data), 0.0, data)
            normalize_mean[var] = np.array([float(np.nanmean(data))])
            normalize_std[var]  = np.array([float(np.nanstd(data))])
            ds.close()

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    np.savez(os.path.join(save_dir, mean_file), **normalize_mean)
    np.savez(os.path.join(save_dir, std_file),  **normalize_std)
    print(f"Saved {mean_file} and {std_file} to {save_dir}.")
    print(f"Keys written: {sorted(normalize_mean.keys())[:5]} ... "
          f"({len(normalize_mean)} total)")


if __name__ == "__main__":
    main()