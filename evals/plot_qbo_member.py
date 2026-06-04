"""
Time-height plot of equatorial (5S-5N), zonally-averaged zonal wind
from 200 hPa to 5 hPa, comparing the SI stochastic emulator (member_19790103)
against ERA5 monthly-mean reanalysis on NCAR RDA (ds633.1).

Model data: 197 monthly NetCDF files (197901..199505), each holding daily U
on a 26-level pressure grid (Pa) and a 1 deg lat/lon grid.
ERA5: yearly NetCDF files of monthly means, 37 levels (hPa), 0.25 deg.
"""
import os, glob, datetime as dt
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

MEMBER   = "member_19800131"
DATA_DIR = f"/glade/campaign/univ/uchi0014/ayz/rollouts_new/{MEMBER}/multilevel/u_component_of_wind"
ERA5_DIR = "/glade/campaign/collections/rda/data/d633001/e5.moda.an.pl"
OUT_DIR  = "/glade/u/home/ayz/amip/logs/qbo"

# Top-11 model levels: stratosphere through upper troposphere (hPa).
LEV_HPA = np.array([5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200])


def eq_zonal_mean(da, lat_name, lon_name):
    """cos(lat)-weighted mean over 5S-5N, then zonal mean. Returns (..., lev)."""
    lat = da[lat_name].values
    mask = np.abs(lat) <= 5.0
    w = np.cos(np.deg2rad(lat[mask]))
    w = w / w.sum()
    sub = da.isel({lat_name: np.where(mask)[0]}).mean(lon_name)
    return (sub * xr.DataArray(w, dims=[lat_name])).sum(lat_name)


def load_model():
    """Returns months (list of 'YYYYMM') and U(months, lev) in m/s."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "u_component_of_wind_*.nc")))
    months, vals = [], []
    for fp in files:
        with xr.open_dataset(fp) as ds:
            plev_hpa = ds["plev"].values / 100.0     # Pa -> hPa
            lev_idx = [int(np.argmin(np.abs(plev_hpa - L))) for L in LEV_HPA]
            u = ds["u_component_of_wind"].isel(plev=lev_idx)   # (time, lev, lat, lon)
            eq = eq_zonal_mean(u, "lat", "lon")                 # (time, lev)
            vals.append(eq.mean("time").values.astype(np.float32))   # monthly mean
            yyyymm = os.path.basename(fp).split("_")[-1].split(".")[0]
            months.append(yyyymm)
    return months, np.stack(vals)


def load_era5(months):
    """Returns U(months, lev) for the same YYYYMM list, in m/s."""
    out = np.full((len(months), len(LEV_HPA)), np.nan, dtype=np.float32)
    by_year = {}
    for i, m in enumerate(months):
        by_year.setdefault(m[:4], []).append((i, int(m[4:6])))

    for y, idx_list in sorted(by_year.items()):
        fp = os.path.join(
            ERA5_DIR, y, f"e5.moda.an.pl.128_131_u.ll025uv.{y}010100_{y}120100.nc",
        )
        if not os.path.exists(fp):
            print(f"  ERA5 missing: {fp}")
            continue
        with xr.open_dataset(fp) as ds:
            lev_hpa = ds["level"].values
            lev_idx = [int(np.argmin(np.abs(lev_hpa - L))) for L in LEV_HPA]
            u = ds["U"].isel(level=lev_idx)                     # (time, lev, lat, lon)
            eq = eq_zonal_mean(u, "latitude", "longitude")       # (time, lev)
            arr = eq.values.astype(np.float32)                  # (12, lev)
        for i, mm in idx_list:
            if 1 <= mm <= arr.shape[0]:
                out[i] = arr[mm - 1]
    return out


def main():
    print(f"Reading model rollouts from {DATA_DIR}")
    months, u_pred = load_model()
    print(f"  {len(months)} months: {months[0]} - {months[-1]}, levels: {LEV_HPA.tolist()}")

    print(f"Reading ERA5 monthly U from {ERA5_DIR}")
    u_era = load_era5(months)

    np.savez(
        os.path.join(OUT_DIR, f"qbo_eq_u_{MEMBER}.npz"),
        pred=u_pred, era5=u_era, levels=LEV_HPA,
        months=np.array(months),
    )

    dates = [dt.date(int(m[:4]), int(m[4:6]), 15) for m in months]

    vmax = float(np.nanmax(np.abs(np.concatenate([u_pred, u_era]))))
    vmax = max(20.0, np.ceil(vmax / 5) * 5)
    step = 5.0
    levels_c = np.arange(-vmax, vmax + step / 2, step)

    # RdBu_r but with the two bins straddling zero (|U| < 5 m/s) set to white.
    base_cmap = plt.get_cmap("RdBu_r")
    n_int = len(levels_c) - 1
    colors = base_cmap(np.linspace(0.0, 1.0, n_int))
    mid = 0.5 * (levels_c[:-1] + levels_c[1:])
    colors[np.abs(mid) < step] = [1.0, 1.0, 1.0, 1.0]
    cmap = ListedColormap(colors)
    cmap.set_under(base_cmap(0.0))
    cmap.set_over(base_cmap(1.0))

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, sharey=True)
    for ax, data, title in zip(
        axes, [u_era, u_pred],
        ["ERA5 monthly mean (ds633.1)",
         f"SI stochastic emulator ({MEMBER})"],
    ):
        cf = ax.contourf(dates, LEV_HPA, data.T, levels=levels_c,
                         cmap=cmap, extend="both")
        ax.set_yscale("log")
        ax.set_ylim(200, 5)
        ax.set_yticks(LEV_HPA)
        ax.set_yticklabels([str(int(v)) for v in LEV_HPA])
        ax.set_ylabel("Pressure (hPa)")
        ax.set_title(f"{title}: zonal-mean U, 5S-5N")
    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    fig.colorbar(cf, ax=axes, orientation="vertical", pad=0.02,
                 shrink=0.9, label="U (m s$^{-1}$)")
    fig.suptitle(
        f"Equatorial zonal-mean U (200-5 hPa) - QBO check  "
        f"[{months[0]} - {months[-1]}]", fontsize=13,
    )

    out_png = os.path.join(OUT_DIR, f"qbo_time_height_{MEMBER}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {os.path.join(OUT_DIR, f'qbo_eq_u_{MEMBER}.npz')}")
    print(f"Pred U range:  [{np.nanmin(u_pred):+.2f}, {np.nanmax(u_pred):+.2f}] m/s")
    print(f"ERA5 U range:  [{np.nanmin(u_era):+.2f},  {np.nanmax(u_era):+.2f}] m/s")


if __name__ == "__main__":
    main()
