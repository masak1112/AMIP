"""Convert ERA5 normalization .npz files to .nc files.

Each .npz file (e.g. normalize_mean.npz, normalize_std.npz,
normalize_diff_mean_6.npz, ...) is converted to a corresponding .nc file
in the same directory.

Output nc structure
-------------------
Coordinates:
    level  : pressure levels discovered from the npz keys (hPa)

Variables (pressure-level, dim = level):
    geopotential, specific_humidity, temperature,
    u_component_of_wind, v_component_of_wind, vertical_velocity

Variables (scalar / 0-D):
    all remaining keys (single-level prognostic, diagnostic,
    varying boundary conditions, time-invariant constants)

Usage
-----
    python normalize_npz_to_nc_era5.py --npz_dir /path/to/norm_dir
    python normalize_npz_to_nc_era5.py --npz_dir /path/to/norm_dir --save_dir /path/to/out
"""

import argparse
import glob
import os

import numpy as np
import xarray as xr


# Must match compute_normalization_era5*.py
PRESSURE_LEVEL_VARS = [
    #"fraction_of_cloud_cover",
    "geopotential",
    #"specific_cloud_ice_water_content",
    #"specific_cloud_liquid_water_content",
    "specific_humidity",
    "specific_total_water",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    #"vertical_velocity",
]

LEVEL_DIM = "level"


def extract_levels_from_keys(keys, pl_var):
    """Return sorted pressure-level values for *pl_var* found in *keys*."""
    prefix = f"{pl_var}_"
    levels = []
    for k in keys:
        if k.startswith(prefix):
            suffix = k[len(prefix):]
            try:
                levels.append(float(suffix))
            except ValueError:
                pass
    return sorted(levels)


def convert_npz(npz_path, save_dir):
    """Convert a single normalization .npz file to a .nc file."""
    data = np.load(npz_path)
    keys = list(data.keys())

    # --- Discover pressure levels from the first available PL variable ---
    pressure_levels = None
    for pl_var in PRESSURE_LEVEL_VARS:
        lvls = extract_levels_from_keys(keys, pl_var)
        if lvls:
            pressure_levels = np.array(lvls, dtype=np.float32)
            break

    if pressure_levels is None:
        raise ValueError(
            f"No pressure-level keys found in {npz_path}. "
            f"Expected keys like 'geopotential_1000.0'."
        )

    # --- Build the dataset ---
    ds = xr.Dataset(coords={LEVEL_DIM: pressure_levels})

    # Identify which keys belong to pressure-level variables
    pl_keys = set()
    for pl_var in PRESSURE_LEVEL_VARS:
        var_levels = extract_levels_from_keys(keys, pl_var)
        if not var_levels:
            continue
        values = np.array(
            [data[f"{pl_var}_{float(l)}"][0] for l in var_levels],
            dtype=np.float32,
        )
        ds[pl_var] = xr.DataArray(values, dims=[LEVEL_DIM],
                                  coords={LEVEL_DIM: pressure_levels})
        for l in var_levels:
            pl_keys.add(f"{pl_var}_{float(l)}")

    # Remaining keys are single-level (scalar) variables
    for key in keys:
        if key in pl_keys:
            continue
        ds[key] = xr.DataArray(float(data[key][0]))

    # --- Save ---
    basename = os.path.splitext(os.path.basename(npz_path))[0]
    nc_path = os.path.join(save_dir, f"{basename}.nc")
    ds.to_netcdf(nc_path)
    print(f"  {os.path.basename(npz_path)}  ->  {os.path.basename(nc_path)}")
    return nc_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ERA5 normalization .npz files to .nc files."
    )
    parser.add_argument(
        "--npz_dir", type=str, required=True,
        help="Directory containing the normalization .npz files.",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Output directory for .nc files (default: same as --npz_dir).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    npz_dir  = args.npz_dir
    save_dir = args.save_dir if args.save_dir is not None else npz_dir
    os.makedirs(save_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(npz_dir, "normalize*.npz")))
    if not npz_files:
        print(f"No normalize*.npz files found in {npz_dir}")
        return

    print(f"Found {len(npz_files)} .npz file(s) in {npz_dir}")
    for npz_path in npz_files:
        convert_npz(npz_path, save_dir)

    print("Done.")


if __name__ == "__main__":
    main()
