#!/usr/bin/env python -u
"""Add specific_humidity pressure-level fields into existing per-timestep HDF5 files.

The h5 files have already been created by another script (one file per timestep,
named ``{year}_{idx:04d}.h5``).  This script opens each existing file in append
mode and adds ``input/specific_humidity_{level}`` datasets read from the
corresponding NetCDF file at::

    {root_dir}/prognostic/3D_PL/specific_humidity/{year}_180x360.nc

The number of timesteps in the nc file for a given year must match the number
of existing ``{year}_*.h5`` files in ``save_dir``.
"""

import os
import argparse
import glob
import numpy as np
import xarray as xr
import h5py
from tqdm import tqdm
import warnings


VAR_NAME = "specific_humidity"
VAR_SUBPATH = os.path.join("prognostic", "3D_PL", VAR_NAME)
LEVEL_DIM = "level"


def open_dataset(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xr.open_dataset(path, engine = "h5netcdf")


def get_nc_var_name(ds, dir_name):
    data_vars = list(ds.data_vars)
    if dir_name in data_vars:
        return dir_name
    if len(data_vars) == 1:
        return data_vars[0]
    raise ValueError(
        f"Cannot determine nc variable name for '{dir_name}'; "
        f"candidates: {data_vars}"
    )


def add_specific_humidity(root_dir, save_dir, years, chunk_size=100,
                          backup_root_dir=None, overwrite=False):
    for year in tqdm(years, desc="years", position=0):
        nc_path = os.path.join(root_dir, VAR_SUBPATH, f"{year}_180x360.nc")
        if not os.path.exists(nc_path) and backup_root_dir is not None:
            nc_path = os.path.join(backup_root_dir, VAR_SUBPATH, f"{year}_180x360.nc")
        if not os.path.exists(nc_path):
            print(f"  Warning: missing {nc_path}, skipping year {year}.")
            continue

        ds = open_dataset(nc_path)
        nc_var = get_nc_var_name(ds, VAR_NAME)
        levels = ds[LEVEL_DIM].values
        n_time = len(ds.time)

        # Check that the number of h5 files for this year matches n_time.
        h5_files = sorted(glob.glob(os.path.join(save_dir, f"{year}_*.h5")))
        if len(h5_files) != n_time:
            print(
                f"  Warning: year {year} has {n_time} timesteps in nc but "
                f"{len(h5_files)} h5 files in {save_dir}. Skipping."
            )
            ds.close()
            continue

        n_chunks = n_time // chunk_size + (1 if n_time % chunk_size else 0)
        idx_in_year = 0

        for chunk_id in tqdm(range(n_chunks), desc="chunks", position=1, leave=False):
            t_slice = slice(chunk_id * chunk_size, (chunk_id + 1) * chunk_size)
            # Shape: (T, n_levels, lat, lon)
            chunk = ds[nc_var].isel(time=t_slice).values.astype(np.float32)
            chunk_len = chunk.shape[0]

            for i in tqdm(range(chunk_len), desc="timesteps", position=2, leave=False):
                h5_path = os.path.join(save_dir, f"{year}_{idx_in_year:04d}.h5")
                if not os.path.exists(h5_path):
                    print(f"  Warning: missing {h5_path}, skipping.")
                    idx_in_year += 1
                    continue

                with h5py.File(h5_path, "a", libver="latest") as f:
                    grp = f.require_group("input")
                    for j, level in enumerate(levels):
                        key = f"{VAR_NAME}_{level}"
                        if key in grp:
                            if overwrite:
                                del grp[key]
                            else:
                                continue
                        grp.create_dataset(
                            key, data=chunk[i, j], compression=None, dtype=np.float32
                        )

                idx_in_year += 1

        ds.close()
        print(f"  Year {year}: updated {idx_in_year} file(s).")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add specific_humidity pressure-level fields into existing "
            "per-timestep HDF5 files."
        )
    )
    parser.add_argument(
        "--root_dir", type=str, required=True,
        help=(
            "Root directory of the ERA5 AMIP data (containing "
            "prognostic/3D_PL/specific_humidity/{year}_180x360.nc)."
        ),
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help="Directory containing existing {year}_{idx:04d}.h5 files to update.",
    )
    parser.add_argument(
        "--start_year", type=int, default=1979,
        help="First year to process (default: 1979).",
    )
    parser.add_argument(
        "--end_year", type=int, default=2024,
        help="Last year to process, inclusive (default: 2024).",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100,
        help="Number of timesteps to load into memory at once (default: 100).",
    )
    parser.add_argument(
        "--backup_root_dir", type=str, default=None,
        help=(
            "Optional backup root directory.  When a file is not found under "
            "--root_dir, the script will search for it here before skipping."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing specific_humidity_{level} datasets in the h5 files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    add_specific_humidity(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        years=list(range(args.start_year, args.end_year + 1)),
        chunk_size=args.chunk_size,
        backup_root_dir=args.backup_root_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
