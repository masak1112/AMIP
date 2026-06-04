"""Scan AMIP h5 files for missing upper-air variable keys.

Motivated by a training-time crash:
    KeyError: 'specific_humidity_5.0'
raised from `get_data_given_path` in data/amip_new.py while loading
1989_1188.h5 under configs/SI_midway_AIMIP.yaml.

This script:
  1) loads SI_midway_AIMIP.yaml to recover the exact expected key list
     (specifically `{var}_{int(level)}.0` for every upper-air variable x level),
  2) inspects the suspect file 1989_1188.h5 in detail,
  3) randomly samples N other h5 files from data_dir and checks every
     `specific_humidity_*` key (and optionally the full key list) for each,
  4) prints a summary of files with missing keys.

Run on the cluster where /project/pedramh/AMIP/h5 is accessible.
"""

import os
import random
from itertools import product

import h5py
import numpy as np

from common.utils import get_yaml


CONFIG_PATH = "configs/SI_midway_AIMIP.yaml"
SUSPECT_FILE = "1989_1188.h5"        # the file from the training error
N_RANDOM_SAMPLES = 30                # how many other files to spot-check
RANDOM_SEED = 0
CHECK_ALL_INPUT_KEYS = False         # if True, check the full variable_list_in
                                     # (input). Default: only specific_humidity_*.


def expected_specific_humidity_keys(upper_air_variables, levels, level_units=".0"):
    """Mirror amip_new.GetDataset._build_variable_lists for one variable."""
    if "specific_humidity" not in upper_air_variables:
        return []
    return [f"specific_humidity_{int(lev)}{level_units}" for lev in levels]


def expected_input_keys(dataconfig, level_units=".0"):
    """Mirror the full input variable list from amip_new.GetDataset."""
    upper_air = dataconfig["upper_air_variables"]
    surface = dataconfig["surface_variables"]
    diagnostic = dataconfig["diagnostic_variables"]
    varying = dataconfig.get("varying_boundary_variables", [])
    levels = dataconfig["levels"]

    keys = [f"{v}_{int(l)}{level_units}" for v, l in product(upper_air, levels)]
    keys.extend(surface)
    keys.extend(varying)
    # diagnostic_input is True in this config; add diagnostics to input list too.
    if dataconfig.get("diagnostic_input", False):
        keys.extend(diagnostic)
    return keys


def inspect_file(path, sh_keys, all_keys=None, verbose=True):
    """Return (missing_sh, missing_all, error_msg). None for the lists on error."""
    if not os.path.exists(path):
        return None, None, "file does not exist"
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return None, None, f"stat failed: {e}"
    if size == 0:
        return None, None, f"empty file (0 bytes)"

    try:
        with h5py.File(path, "r") as f:
            if "input" not in f:
                return None, None, "missing 'input' group"
            input_keys = set(f["input"].keys())
            missing_sh = [k for k in sh_keys if k not in input_keys]
            missing_all = None
            if all_keys is not None:
                missing_all = [k for k in all_keys if k not in input_keys]
            if verbose:
                present_sh = len(sh_keys) - len(missing_sh)
                print(f"  input has {len(input_keys)} datasets, "
                      f"specific_humidity present {present_sh}/{len(sh_keys)}")
                if missing_sh:
                    print(f"  MISSING specific_humidity keys: {missing_sh}")
                else:
                    print("  all specific_humidity keys present")
                if all_keys is not None and missing_all:
                    print(f"  MISSING other input keys: {missing_all}")
            return missing_sh, missing_all, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def list_h5_files(data_dir):
    files = []
    for name in os.listdir(data_dir):
        if name.endswith(".h5"):
            files.append(name)
    files.sort()
    return files


def main():
    config = get_yaml(CONFIG_PATH)
    dataconfig = config["data"]
    data_dir = dataconfig["data_dir"]

    sh_keys = expected_specific_humidity_keys(
        dataconfig["upper_air_variables"], dataconfig["levels"]
    )
    all_keys = expected_input_keys(dataconfig) if CHECK_ALL_INPUT_KEYS else None

    print(f"Config:                  {CONFIG_PATH}")
    print(f"Data dir:                {data_dir}")
    print(f"Expected SH keys ({len(sh_keys)}): {sh_keys}")
    if all_keys is not None:
        print(f"Expected total input keys: {len(all_keys)}")

    if not os.path.isdir(data_dir):
        print(f"\nERROR: data_dir does not exist on this machine: {data_dir}")
        print("Run this script on the midway cluster.")
        return

    # 1) Inspect the suspect file.
    suspect_path = os.path.join(data_dir, SUSPECT_FILE)
    print(f"\n--- [SUSPECT] {suspect_path} ---")
    miss_sh, miss_all, err = inspect_file(suspect_path, sh_keys, all_keys)
    if err:
        print(f"  ERROR: {err}")

    # 2) Random spot-check.
    all_files = list_h5_files(data_dir)
    print(f"\nFound {len(all_files)} h5 files in {data_dir}")

    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(all_files, min(N_RANDOM_SAMPLES, len(all_files)))
    # Ensure the suspect file is excluded from the random sample (already inspected).
    sample = [f for f in sample if f != SUSPECT_FILE]

    issues = []
    print(f"\n--- Spot-checking {len(sample)} random files ---")
    for name in sample:
        path = os.path.join(data_dir, name)
        miss_sh, miss_all, err = inspect_file(path, sh_keys, all_keys, verbose=False)
        if err is not None:
            print(f"  BAD  {name}  -- {err}")
            issues.append((name, err))
        elif miss_sh:
            print(f"  BAD  {name}  -- missing SH keys: {miss_sh}")
            issues.append((name, f"missing SH keys: {miss_sh}"))
        elif all_keys is not None and miss_all:
            print(f"  BAD  {name}  -- missing input keys: {miss_all}")
            issues.append((name, f"missing input keys: {miss_all}"))
        else:
            print(f"  OK   {name}")

    # 3) Optional broader scan: every file whose name starts with the suspect's year,
    # so we can see if the problem is isolated to one file or a whole year.
    year_prefix = SUSPECT_FILE.split("_")[0] + "_"
    year_files = [f for f in all_files if f.startswith(year_prefix)]
    print(f"\n--- Scanning all {len(year_files)} files for year {year_prefix[:-1]} ---")
    year_issues = []
    for name in year_files:
        path = os.path.join(data_dir, name)
        miss_sh, _, err = inspect_file(path, sh_keys, None, verbose=False)
        if err is not None:
            print(f"  BAD  {name}  -- {err}")
            year_issues.append((name, err))
        elif miss_sh:
            print(f"  BAD  {name}  -- missing SH keys: {miss_sh}")
            year_issues.append((name, f"missing SH keys: {miss_sh}"))

    print("\n=== SUMMARY ===")
    print(f"Random sample: {len(issues)} of {len(sample)} files had problems.")
    for name, msg in issues:
        print(f"  {name}: {msg}")
    print(f"\nYear {year_prefix[:-1]} scan: "
          f"{len(year_issues)} of {len(year_files)} files had problems.")
    for name, msg in year_issues:
        print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
