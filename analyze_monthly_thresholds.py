"""Estimate per-month spectral RMSE thresholds for eval_lagged's monitor.

For every daily 2m_temperature snapshot in a stable model rollout we compute the
spherical-harmonic power spectrum (same metric as the runtime monitor) and the
normalized spectral RMSE against the snapshot ``lag_days`` earlier. The
resulting values describe the *natural* spectral drift in a non-blowup
rollout; thresholds should sit comfortably above this baseline so the monitor
fires only on genuine instabilities.
"""

import argparse
import glob
import re
from collections import defaultdict

import numpy as np
import torch
import xarray as xr

from eval_lagged import _SphericalSpectrum, normalized_spectral_rmse


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout_dir",
                    default="/glade/campaign/univ/uchi0014/ayz/member_19800101")
    ap.add_argument("--lag_days", type=int, default=30,
                    help="Look-back window — must match the runtime monitor (default 30).")
    ap.add_argument("--safety_factor", type=float, default=1.5,
                    help="Multiplier applied to the per-month max drift to suggest a threshold.")
    ap.add_argument("--floor", type=float, default=0.3,
                    help="Minimum threshold (drift can be tiny in stable months but the "
                         "monitor still needs to catch real blowups).")
    ap.add_argument("--year", type=int, default=None,
                    help="Restrict to a single calendar year (e.g. 1980) when later years "
                         "in the rollout are known to be unstable.")
    args = ap.parse_args()

    # ---- Collect all monthly files and stitch into a contiguous time series. ----
    pattern = f"{args.rollout_dir}/surface/2m_temperature/2m_temperature_??????.nc"
    files = sorted(glob.glob(pattern))
    if args.year is not None:
        # Need lag_days of context from the previous year so the early-month
        # comparisons are well-defined; the rmse loop then filters out anything
        # outside ``year``.
        files = [f for f in files if re.search(rf"_({args.year})\d\d\.nc$", f)]
    if not files:
        raise SystemExit(f"No files matched {pattern}")

    print(f"Found {len(files)} monthly files: {files[0].split('/')[-1]} ... "
          f"{files[-1].split('/')[-1]}")

    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 31},
                           engine="h5netcdf")
    t2m = ds["2m_temperature"]
    times = ds["time"].values
    nlat = ds.sizes["lat"]
    nlon = ds.sizes["lon"]
    print(f"Stitched series: {len(times)} days, grid {nlat}x{nlon}, "
          f"{times[0]} .. {times[-1]}")

    # ---- Compute per-day SHT spectrum. ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sht_power = _SphericalSpectrum(nlat, nlon, device=device)

    print(f"Computing spherical-harmonic spectra on {device} ...")
    spectra = []
    for i in range(len(times)):
        field = torch.from_numpy(t2m.isel(time=i).values).to(device).float()
        spectra.append(sht_power(field))
        if (i + 1) % 90 == 0 or i == len(times) - 1:
            print(f"  {i + 1}/{len(times)}")
    spectra = torch.stack(spectra, dim=0)  # (T, lmax)

    # ---- Per-day rmse vs lag-days-earlier. Bucket by month of the *later* day. ----
    lag = args.lag_days
    months = np.array([np.datetime64(t, "M").astype(object).month for t in times])
    by_month = defaultdict(list)
    for i in range(lag, len(times)):
        rmse = normalized_spectral_rmse(spectra[i], spectra[i - lag])
        by_month[months[i]].append(rmse)

    # ---- Report + suggest. ----
    print(f"\nNatural spectral drift over {lag}-day windows "
          f"(2m_T spherical-harmonic spectrum, l>=1):")
    print(f"  {'mon':>3}  {'n':>4}  {'mean':>7}  {'p50':>7}  {'p95':>7}  "
          f"{'max':>7}   suggested")
    suggestion = {}
    for m in range(1, 13):
        vals = np.array(by_month.get(m, []))
        if vals.size == 0:
            print(f"  {MONTH_NAMES[m-1]:>3}  --  (no data)")
            continue
        mx = float(vals.max())
        thr = max(args.floor, args.safety_factor * mx)
        suggestion[m] = thr
        print(f"  {MONTH_NAMES[m-1]:>3}  {vals.size:4d}  "
              f"{vals.mean():7.4f}  {np.percentile(vals,50):7.4f}  "
              f"{np.percentile(vals,95):7.4f}  {mx:7.4f}   "
              f"{thr:.2f}")

    # ---- Bash-array form for easy paste into the submit script. ----
    print("\nSuggested thresholds (safety_factor=" + f"{args.safety_factor}, "
          f"floor={args.floor}):")
    print("  ", " ".join(f"{MONTH_NAMES[m-1]}={suggestion.get(m, args.floor):.2f}"
                          for m in range(1, 13)))


if __name__ == "__main__":
    main()
