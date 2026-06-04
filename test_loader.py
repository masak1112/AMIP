"""Diagnose a forcing-missing error from eval_lagged.py.

eval_lagged.py reports lines like:
    Forcing missing for 1989-10-25T00:00:00: Unable to synchronously open
    file (file signature not found) -- stopping at step 3224.

The cause is `load_forcing` -> `dataset._get_data(t, variable_list=
varying_boundary_variables)` -> `get_data_given_path(...)` which calls
`h5py.File(path, 'r')`. Either the file is corrupt/empty/missing or one of
the requested variable keys is absent from the `input` group.

This script:
  1) reconstructs the exact h5 path that eval_lagged.py would open for the
     suspect datetime,
  2) reports its size / openability / file signature,
  3) if openable, lists which expected forcing keys are missing,
  4) compares against an adjacent good file to make the diff obvious,
  5) optionally scans a wider window for any other corrupted/incomplete files.
"""

import os
from datetime import timedelta

import h5py
import numpy as np

from common.utils import get_yaml
from data.amip_new import GetDataset, get_out_path

CONFIG_PATH = "configs/combined_NCAR.yaml"
SUSPECT_DATETIME = "1989-10-25T00:00:00"  # from the eval_lagged.py log
SCAN_WINDOW_DAYS = 5                       # +/- days around the suspect file
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"      # standard HDF5 magic bytes


def datetime_to_h5_path(dataset, year, month, day, hour=0):
    """Replicate the path-construction used by GetDataset._get_data."""
    t = dataset.datetime_class(year, month, day, hour=hour,
                               has_year_zero=dataset.has_year_zero)
    seconds_into_year = int(
        (t - dataset.datetime_class(year, 1, 1, hour=0,
                                     has_year_zero=dataset.has_year_zero)).total_seconds()
    )
    data_idx = seconds_into_year // 3600 // dataset.data_timedelta_hours
    return get_out_path(dataset.data_dir, year, data_idx), data_idx, t


def file_signature_ok(path):
    if not os.path.exists(path):
        return False, "file does not exist"
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return False, f"stat failed: {e}"
    if size < len(HDF5_SIGNATURE):
        return False, f"file too small ({size} bytes)"
    with open(path, "rb") as f:
        head = f.read(len(HDF5_SIGNATURE))
    if head != HDF5_SIGNATURE:
        return False, f"bad signature: {head!r}"
    return True, f"ok ({size} bytes)"


def inspect_file(path, expected_keys, label=""):
    """Print a one-block report for a single h5 file."""
    print(f"\n--- {label} {path} ---")
    sig_ok, sig_msg = file_signature_ok(path)
    print(f"  signature: {sig_msg}")
    if not sig_ok:
        return  # nothing more we can do

    try:
        with h5py.File(path, "r") as f:
            top_keys = list(f.keys())
            print(f"  top-level groups: {top_keys}")
            if "input" not in f:
                print("  MISSING 'input' group — cannot check forcing keys.")
                return
            input_keys = set(f["input"].keys())
            print(f"  'input' has {len(input_keys)} datasets")

            missing = [k for k in expected_keys if k not in input_keys]
            present = [k for k in expected_keys if k in input_keys]
            print(f"  expected forcing keys present ({len(present)}/{len(expected_keys)}): {present}")
            if missing:
                print(f"  MISSING forcing keys: {missing}")
            else:
                print("  all expected forcing keys are present")

            # Sanity-check that each expected dataset can actually be read,
            # not just listed -- a truncated file may list keys but fail on read.
            unreadable = []
            for k in present:
                try:
                    _ = np.asarray(f["input"][k][...])
                except Exception as e:
                    unreadable.append((k, str(e)))
            if unreadable:
                print(f"  UNREADABLE present keys: {unreadable}")
    except Exception as e:
        print(f"  h5py.File(...) raised: {type(e).__name__}: {e}")


def main():
    config = get_yaml(CONFIG_PATH)
    dataconfig = config["data"]

    # GetDataset needs explicit year_start/year_end; cover the suspect year.
    suspect_year = int(SUSPECT_DATETIME[:4])
    dataset = GetDataset(dataconfig,
                         year_start=suspect_year,
                         year_end=suspect_year + 1)

    expected_keys = list(dataset.varying_boundary_variables)
    print(f"\nConfig:                  {CONFIG_PATH}")
    print(f"Data dir:                {dataset.data_dir}")
    print(f"data_timedelta_hours:    {dataset.data_timedelta_hours}")
    print(f"Expected forcing keys:   {expected_keys}")

    y, m, d = (int(SUSPECT_DATETIME[0:4]),
               int(SUSPECT_DATETIME[5:7]),
               int(SUSPECT_DATETIME[8:10]))
    suspect_path, suspect_idx, suspect_t = datetime_to_h5_path(dataset, y, m, d)
    print(f"\nSuspect datetime:        {suspect_t.isoformat()}")
    print(f"Suspect file index:      {suspect_idx}")
    print(f"Suspect file path:       {suspect_path}")

    # 1) Inspect the suspect file itself.
    inspect_file(suspect_path, expected_keys, label="[SUSPECT]")

    # 2) Inspect a known-good neighbor (one day earlier) for comparison.
    prior_t = suspect_t - timedelta(days=1)
    prior_path, _, _ = datetime_to_h5_path(dataset,
                                            prior_t.year, prior_t.month, prior_t.day)
    inspect_file(prior_path, expected_keys, label="[REF: T-1d]")

    # 3) Scan a wider window so we know whether the corruption is isolated or
    # spans multiple files.
    print(f"\n--- Scanning +/-{SCAN_WINDOW_DAYS} days around suspect ---")
    issues = []
    for delta in range(-SCAN_WINDOW_DAYS, SCAN_WINDOW_DAYS + 1):
        t = suspect_t + timedelta(days=delta)
        # Scan all data_timedelta_hours-spaced steps within each day so we
        # also catch a single bad 06h/12h/18h file.
        steps_per_day = 24 // dataset.data_timedelta_hours
        for step in range(steps_per_day):
            hour = step * dataset.data_timedelta_hours
            path, idx, t_full = datetime_to_h5_path(
                dataset, t.year, t.month, t.day, hour=hour)
            sig_ok, sig_msg = file_signature_ok(path)
            tag = "OK " if sig_ok else "BAD"
            row = f"  {tag}  {t_full.isoformat()}  idx={idx:04d}  {os.path.basename(path)}  ({sig_msg})"
            print(row)
            if not sig_ok:
                issues.append((t_full.isoformat(), path, sig_msg))
            elif sig_ok:
                # Quick key-presence check on every scanned file.
                try:
                    with h5py.File(path, "r") as f:
                        if "input" not in f:
                            issues.append((t_full.isoformat(), path, "no 'input' group"))
                            continue
                        ks = set(f["input"].keys())
                        missing = [k for k in expected_keys if k not in ks]
                        if missing:
                            issues.append((t_full.isoformat(), path,
                                           f"missing keys: {missing}"))
                except Exception as e:
                    issues.append((t_full.isoformat(), path,
                                   f"open failed: {type(e).__name__}: {e}"))

    print("\n=== SUMMARY ===")
    if not issues:
        print("No corrupt or incomplete files found in the scan window.")
    else:
        print(f"{len(issues)} problem(s) found:")
        for iso, path, msg in issues:
            print(f"  {iso}  {path}  -- {msg}")


if __name__ == "__main__":
    main()
