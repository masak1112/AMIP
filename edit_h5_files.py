#!/usr/bin/env python -u
"""Convert ERA5 AMIP NetCDF files to per-timestep HDF5 files.

Data layout expected at root_dir:
  forcing/
    {const_var}/180x360.nc               (time-invariant constants)
    {varying_var}/{year}_180x360.nc      (non-repeating varying BCs)
  prognostic/
    {single_var}/{year}_180x360.nc
    3D_PL/{var}/{year}_180x360.nc
  diagnostic/
    {diag_var}/{year}_180x360.nc

Output layout (save_dir):
  {year}_{idx:04d}.h5
    input/
      {var_name}                 (2-D fields, float32)
      {var_name}_{level}         (pressure-level slices, float32)
      time                       (string scalar)

Variable names in the HDF5 files use the directory name as the key,
regardless of the internal nc variable name (e.g. '2m_specific_humidity'
is stored as '2m_specific_humidity' even though the nc variable is 'q2m').
"""

import os
import argparse
import numpy as np
import xarray as xr
import h5py
from tqdm import tqdm
import warnings


# ---------------------------------------------------------------------------
# Variable lists  (must match compute_normalization_era5.py)
# ---------------------------------------------------------------------------

CONSTANTS = [
    "geopotential_at_surface",
    "land_sea_mask",
]

VARYING_BOUNDARY = [
    "DSWRFtoa_24h",
    "sea_ice_cover",
    "sea_surface_temperature",
]

SINGLE_LEVEL_VARS_PROGNOSTIC = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_specific_humidity",
    "2m_temperature",
    "skin_temperature",
    "snow_depth",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "surface_pressure",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]

SINGLE_LEVEL_VARS_DIAGNOSTIC = [
    "DLWRFsfc_24h",
    "DSWRFsfc_24h",
    "LHTFLsfc_24h",
    "PRATEsfc_24h",
    "SHTFLsfc_24h",
    "ULWRFsfc_24h",
    "ULWRFtoa_24h",
    "USWRFsfc_24h",
    "USWRFtoa_24h",
    "hcc_24h",
    "lcc_24h",
    "mcc_24h",
    "mn2t_24h",
    "mx2t_24h",
    "mxtpr_24h"
]

PRESSURE_LEVEL_VARS = [
    #"fraction_of_cloud_cover",
    "geopotential",
    #"specific_cloud_ice_water_content",
    #"specific_cloud_liquid_water_content",
    #"specific_humidity",
    "specific_total_water",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

# 1-D (time-only) variables whose annual files live at
# forcing/{var}/forcing_{year}.nc.  At each timestep the scalar value is
# broadcast to a constant (lat, lon) field before being stored in the h5 file.
# Must match CONSTANT_SCALARS in compute_normalization_era5_parallel.py.
CONSTANT_SCALARS = [
    "global_mean_co2"
]

LEVEL_DIM = "level"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_dataset(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_dataset(path)
    return ds


def get_nc_var_name(ds, dir_name):
    """Return the actual variable name stored in *ds*."""
    data_vars = list(ds.data_vars)
    if dir_name in data_vars:
        return dir_name
    if len(data_vars) == 1:
        return data_vars[0]
    raise ValueError(
        f"Cannot determine nc variable name for '{dir_name}'; "
        f"candidates: {data_vars}"
    )


def get_data_path(root_dir, var, year):
    """Return the path to the annual nc file for *var* in *year*."""
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


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def create_one_step_dataset(root_dir, save_dir, years, chunk_size=100,
                            backup_root_dir=None):
    os.makedirs(save_dir, exist_ok=True)

    all_single_vars = VARYING_BOUNDARY + SINGLE_LEVEL_VARS_PROGNOSTIC + SINGLE_LEVEL_VARS_DIAGNOSTIC

    # Load time-invariant constants once (they are the same for every timestep).
    constant_arrays = {}
    for var in CONSTANTS:
        path = os.path.join(root_dir, "forcing", var, "180x360.nc")
        if not os.path.exists(path) and backup_root_dir is not None:
            path = os.path.join(backup_root_dir, "forcing", var, "180x360.nc")
        if not os.path.exists(path):
            print(f"Warning: constant file not found: {path}. Skipping '{var}'.")
            continue
        ds = open_dataset(path)
        nc_var = get_nc_var_name(ds, var)
        arr = ds[nc_var].values.astype(np.float32)
        arr = arr.reshape(arr.shape[-2:])  # ensure (lat, lon) shape
        constant_arrays[var] = arr
        ds.close()
    print(f"Loaded {len(constant_arrays)} constant field(s).")

    # Spatial shape (lat, lon) for broadcasting CONSTANT_SCALARS; determined lazily.
    spatial_shape = next(iter(constant_arrays.values())).shape if constant_arrays else None

    for year in tqdm(years, desc="years", position=0):
        print(f"\nProcessing year {year}...")

        # Open all single-level datasets for this year.
        single_level_datasets = {}
        for var in all_single_vars:
            path = get_data_path(root_dir, var, year)
            if not os.path.exists(path) and backup_root_dir is not None:
                path = get_data_path(backup_root_dir, var, year)
            if not os.path.exists(path):
                print(f"  Warning: missing {path}, skipping '{var}' for year {year}.")
                continue
            single_level_datasets[var] = open_dataset(path)

        # Open all pressure-level datasets for this year.
        pl_datasets = {}
        pl_levels = {}
        for var in PRESSURE_LEVEL_VARS:
            path = get_data_path(root_dir, var, year)
            if not os.path.exists(path) and backup_root_dir is not None:
                path = get_data_path(backup_root_dir, var, year)
            if not os.path.exists(path):
                print(f"  Warning: missing {path}, skipping '{var}' for year {year}.")
                continue
            ds = open_dataset(path)
            pl_datasets[var] = ds
            pl_levels[var] = ds[LEVEL_DIM].values

        # Open CONSTANT_SCALARS datasets for this year (1-D time series).
        cs_datasets = {}
        for var in CONSTANT_SCALARS:
            path = get_data_path(root_dir, var, year)
            if not os.path.exists(path) and backup_root_dir is not None:
                path = get_data_path(backup_root_dir, var, year)
            if not os.path.exists(path):
                print(f"  Warning: missing {path}, skipping '{var}' for year {year}.")
                continue
            cs_datasets[var] = open_dataset(path)

        # Use the first available dataset to get the time axis length.
        all_ds = list(single_level_datasets.values()) + list(pl_datasets.values())
        if not all_ds:
            print(f"  No data found for year {year}, skipping.")
            continue
        n_time = len(all_ds[0].time)

        if chunk_size is not None:
            n_chunks = n_time // chunk_size + (1 if n_time % chunk_size else 0)
        else:
            n_chunks = 1
            chunk_size = n_time

        idx_in_year = 0

        for chunk_id in tqdm(range(n_chunks), desc="chunks", position=1, leave=False):
            t_slice = slice(chunk_id * chunk_size, (chunk_id + 1) * chunk_size)

            # Build numpy arrays for this chunk.
            # dict_np: key -> (T, lat, lon) for single-level,
            #                  (T, lat, lon) per level for PL (stored as key_level).
            dict_np = {}

            for var, ds in single_level_datasets.items():
                nc_var = get_nc_var_name(ds, var)
                chunk_data = ds[nc_var].isel(time=t_slice).values.astype(np.float32)
                dict_np[var] = chunk_data  # shape (T, lat, lon)

            for var, ds in pl_datasets.items():
                nc_var = get_nc_var_name(ds, var)
                chunk_data = ds[nc_var].isel(time=t_slice).values.astype(np.float32)
                # chunk_data shape: (T, levels, lat, lon)
                for i, level in enumerate(pl_levels[var]):
                    dict_np[f"{var}_{level}"] = chunk_data[:, i]  # (T, lat, lon)

            # Determine spatial shape from first 2-D field (needed to broadcast scalars).
            if spatial_shape is None and dict_np:
                spatial_shape = next(iter(dict_np.values())).shape[1:]  # (lat, lon)

            # Read CONSTANT_SCALARS for this chunk: 1-D scalar per timestep.
            cs_chunk = {}
            for var, ds in cs_datasets.items():
                nc_var = get_nc_var_name(ds, var)
                cs_chunk[var] = ds[nc_var].isel(time=t_slice).values.astype(np.float32)

            # Retrieve time stamps from the first available dataset.
            ref_ds = list(single_level_datasets.values())[0] if single_level_datasets else list(pl_datasets.values())[0]
            nc_time_var = get_nc_var_name(ref_ds, list(ref_ds.data_vars)[0])
            # Use the time coordinate of the reference dataset directly.
            time_stamps = ref_ds.isel(time=t_slice).time.values
            chunk_len = len(time_stamps)

            for i in tqdm(range(chunk_len), desc="timesteps", position=2, leave=False):
                data_dict = {"input": {"time": str(time_stamps[i])}}

                # Single-level and pressure-level slices
                for key, arr in dict_np.items():
                    data_dict["input"][key] = arr[i]

                # Time-invariant constants
                for var, arr in constant_arrays.items():
                    data_dict["input"][var] = arr

                # CONSTANT_SCALARS: broadcast scalar value to a (lat, lon) field.
                if spatial_shape is not None:
                    for var, cs_arr in cs_chunk.items():
                        data_dict["input"][var] = np.full(
                            spatial_shape, cs_arr[i], dtype=np.float32
                        )

                h5_path = os.path.join(save_dir, f"{year}_{idx_in_year:04d}.h5")
                with h5py.File(h5_path, "w", libver="latest") as f:
                    for group_name, sub_dict in data_dict.items():
                        grp = f.create_group(group_name)
                        for key, value in sub_dict.items():
                            if key == "time":
                                grp.create_dataset(key, data=value, compression=None)
                            else:
                                grp.create_dataset(
                                    key, data=value, compression=None, dtype=np.float32
                                )

                idx_in_year += 1

        # Close all open datasets for this year.
        for ds in single_level_datasets.values():
            ds.close()
        for ds in pl_datasets.values():
            ds.close()
        for ds in cs_datasets.values():
            ds.close()

        print(f"  Year {year}: wrote {idx_in_year} timestep file(s).")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ERA5 AMIP NetCDF files to per-timestep HDF5 files."
    )
    parser.add_argument(
        "--root_dir", type=str, required=True,
        help="Root directory of the ERA5 AMIP data (contains forcing/, prognostic/, diagnostic/)."
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help="Directory where the .h5 files will be written."
    )
    parser.add_argument(
        "--start_year", type=int, default=1979,
        help="First year to convert (default: 1979)."
    )
    parser.add_argument(
        "--end_year", type=int, default=2024,
        help="Last year to convert, inclusive (default: 2024)."
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100,
        help="Number of timesteps to load into memory at once (default: 100)."
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
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    create_one_step_dataset(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        years=list(range(args.start_year, args.end_year + 1)),
        chunk_size=args.chunk_size,
        backup_root_dir=args.backup_root_dir,
    )


if __name__ == "__main__":
    main()