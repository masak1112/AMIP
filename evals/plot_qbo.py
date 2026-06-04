"""
Time-height plot of equatorial (5S-5N), zonally-averaged zonal wind
from 100 hPa to 5 hPa, for the AMIP SI_X model predictions vs. ERA5 targets.
Used as a quick visual check for QBO-like signal.
"""
import os, glob, datetime as dt
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = "/glade/derecho/scratch/ayz/AMIP_logs/SI_X_forcings_42_2026-05-01T09-46-57/bias_logs_ep_10_forcing_1"
OUT_DIR = "/glade/derecho/scratch/pahlavan/ai-models/AMIP/qbo_check"

# Levels in the model state, in hPa, top -> bottom (26 total).
LEVELS = np.array([5,7,10,20,30,50,70,100,125,150,175,200,250,300,
                   400,500,600,700,800,850,875,900,925,950,975,1000])
# Top of model down to 100 hPa = first 8 indices.
N_TOP = 8
LEV_TOP = LEVELS[:N_TOP]                 # [5,7,10,20,30,50,70,100]

# Lat grid: N->S, 1 deg, centered. lat[i] = 89.5 - i.
LAT = 89.5 - np.arange(180)
# Equatorial band 5S - 5N, cos-lat weighted.
EQ_MASK = np.abs(LAT) <= 5.0             # rows ~85..94
EQ_W = np.cos(np.deg2rad(LAT[EQ_MASK]))
EQ_W = EQ_W / EQ_W.sum()

# Snapshot indices: idx = days past 1995-01-01, every 30 days.
INDICES = list(range(1, 1801 + 1, 30))   # 61 snapshots
BASE_DATE = dt.date(1995, 1, 1)
DATES = [BASE_DATE + dt.timedelta(days=i) for i in INDICES]


def load_eq_zm_u(path):
    """Return (n_top,) zonal-mean, equatorial-mean u from one .pt snapshot."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    u = d["u_component_of_wind"][0, :N_TOP].numpy()    # (8, 180, 360)
    zm = u.mean(axis=2)                                # (8, 180)
    eq = (zm[:, EQ_MASK] * EQ_W[None, :]).sum(axis=1)  # (8,)
    return eq


def collect(prefix):
    out = np.full((len(INDICES), N_TOP), np.nan, dtype=np.float32)
    for k, idx in enumerate(INDICES):
        p = os.path.join(RUN_DIR, f"{prefix}_{idx}.pt")
        if not os.path.exists(p):
            print(f"  missing: {p}")
            continue
        out[k] = load_eq_zm_u(p)
    return out


def main():
    print(f"Reading {len(INDICES)} snapshots from {RUN_DIR}")
    print("Predictions ...")
    u_pred = collect("predictions")
    print("Targets ...")
    u_tgt = collect("targets")

    # Save raw arrays for any follow-up analysis.
    np.savez(os.path.join(OUT_DIR, "qbo_eq_u.npz"),
             pred=u_pred, target=u_tgt,
             levels=LEV_TOP, indices=np.array(INDICES),
             dates=np.array([d.isoformat() for d in DATES]))

    # ----- plot -----
    vmax = float(np.nanmax(np.abs(np.concatenate([u_pred, u_tgt]))))
    # round to a sensible contour bound
    vmax = max(10.0, np.ceil(vmax / 5) * 5)
    levels_c = np.linspace(-vmax, vmax, 21)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, data, title in zip(
        axes, [u_tgt, u_pred],
        ["ERA5 target", "Model prediction (SI_X, ep10)"],
    ):
        # data: (time, lev), lev[0]=5 hPa (top). pcolormesh expects (y, x)
        cf = ax.contourf(DATES, LEV_TOP, data.T, levels=levels_c,
                         cmap="RdBu_r", extend="both")
        ax.set_yscale("log")
        ax.set_ylim(100, 5)   # 100 hPa at bottom, 5 hPa at top
        ax.set_ylabel("Pressure (hPa)")
        ax.set_yticks(LEV_TOP)
        ax.set_yticklabels([str(int(v)) for v in LEV_TOP])
        ax.set_title(f"{title}: zonal-mean U, 5S-5N")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    cbar = fig.colorbar(cf, ax=axes, orientation="vertical",
                        pad=0.02, shrink=0.9, label="U (m s$^{-1}$)")
    fig.suptitle("Equatorial zonal-mean U (100-5 hPa) - QBO check",
                 fontsize=13)

    out_png = os.path.join(OUT_DIR, "qbo_time_height.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {os.path.join(OUT_DIR, 'qbo_eq_u.npz')}")
    print(f"Pred U range: [{np.nanmin(u_pred):+.2f}, {np.nanmax(u_pred):+.2f}] m/s")
    print(f"Tgt  U range: [{np.nanmin(u_tgt):+.2f}, {np.nanmax(u_tgt):+.2f}] m/s")


if __name__ == "__main__":
    main()
