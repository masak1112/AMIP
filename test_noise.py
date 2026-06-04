"""
Log-Uniform Noise Schedule Demo for Climate Diffusion Models
=============================================================
Demonstrates how different noise levels affect climate-like fields
and their power spectra, following the approach in cBottle (Brenowitz et al. 2025).

Key concepts:
  - Log-uniform noise: p(σ) ∝ 1/σ for σ ∈ [σ_min, σ_max]
  - Signal leak: when σ_max is too small, large-scale modes survive noising
  - Spectral resolution: σ_min must be small enough to denoise fine-scale modes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from common.utils import get_yaml
from data.amip_new import GetDataset
import torch 

from common.plotting import zonal_averaged_power_spectrum


def load_field():
    config = get_yaml("configs/SI_Latent_DiT.yaml")
    dataconfig = config["data"]
    dataset = GetDataset(dataconfig,
                         year_start = 2010,
                         year_end = 2011)
    batch = dataset.__getitem__(0)
    surface_t, upper_air_t, diagnostic_t, surface_t1, upper_air_t1, diagnostic_t1, varying_boundary_data = batch 

    t2m = surface_t[2]
    z500 = upper_air_t[3, -10]
    precip = diagnostic_t[8]
    q850 = upper_air_t[4, -6]
    t850 = upper_air_t[0, -6]
    u250 = upper_air_t[1, -13]

    return np.array(z500), np.array(t2m), np.array(precip), np.array(q850), np.array(t850), np.array(u250)
    



# ---------------------------------------------------------------------------
# 2. Log-uniform noise sampling
# ---------------------------------------------------------------------------

def sample_log_uniform(sigma_min, sigma_max, n_samples, rng=None):
    """
    Sample σ from the log-uniform distribution: p(σ) ∝ 1/σ.

    Equivalently, log(σ) ~ Uniform(log(σ_min), log(σ_max)).
    """
    if rng is None:
        rng = np.random.default_rng()
    log_sigma = rng.uniform(np.log(sigma_min), np.log(sigma_max), size=n_samples)
    return np.exp(log_sigma)


def add_noise(x, sigma):
    """Apply Gaussian noise at level σ: x_noisy = x + σ * ε."""
    eps = np.random.randn(*x.shape)
    return x + sigma * eps

# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------

def visualize_noising(sigma_min=0.02, sigma_max=200.0):
    """Main visualization showing noised fields and power spectra."""

    field_types = ["geopotential", "temperature", "precipitation"]
    # Noise levels spanning the full range (log-spaced) plus two comparison regimes
    sigma_levels = [0.0, 0.02, 1.0, 10.0, 80.0, 200.0]

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "Log-Uniform Noise Schedule: Effect on Climate Fields & Power Spectra\n"
        f"σ_min = {sigma_min}, σ_max = {sigma_max}  |  "
        r"$p(\sigma) \propto \sigma^{-1}$",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # Layout: 3 field types × (noised images row + power spectrum plot)
    outer = GridSpec(
        len(field_types), 1, hspace=0.45, top=0.93, bottom=0.05,
        left=0.05, right=0.97,
    )

    cmap = "twilight_shifted"

    z500, t2m, precip, q850, t850, u250 = load_field()

    fields = [z500, t2m, precip]

    for row_idx, ftype in enumerate(field_types):
        
        field = fields[row_idx]
        label = field_types[row_idx].capitalize()
        inner = outer[row_idx].subgridspec(2, len(sigma_levels), height_ratios=[1, 2])

        vmin, vmax = field.min(), field.max()
        # Expand range for noised versions
        vrange = vmax - vmin
        vmin_plot = vmin - 0.3 * vrange
        vmax_plot = vmax + 0.3 * vrange

        # ---- Top sub-row: noised field images ----
        for col, sigma in enumerate(sigma_levels):
            ax = fig.add_subplot(inner[0, col])
            if sigma == 0:
                noised = field
                title = "Clean"
            else:
                noised = add_noise(field, sigma)
                title = f"σ = {sigma}"
            im = ax.imshow(noised, cmap=cmap, vmin=vmin_plot, vmax=vmax_plot, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold")

        fig.colorbar(im, ax=fig.axes[:len(sigma_levels)], orientation="horizontal", fraction=0.02, pad=0.02)

        # ---- Bottom sub-row: power spectra ----
        ax_spec = fig.add_subplot(inner[1, :])
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sigma_levels)))

        for col, sigma in enumerate(sigma_levels):
            if sigma == 0:
                noised = field
                lbl = "Clean signal"
                ls = "-"
                lw = 1
            else:
                noised = add_noise(field, sigma)
                lbl = f"σ = {sigma}"
                ls = "--"
                lw = 0.5
            k, P = zonal_averaged_power_spectrum(torch.tensor(noised))
            ax_spec.loglog(k, P, ls=ls, lw=lw, color=colors[col], label=lbl, alpha=0.5)

        # Show the flat noise floor for reference at a few sigma levels
        for sigma in [0.02, 1.0, 200.0]:
            noise_floor = sigma**2
            ax_spec.axhline(
                noise_floor, color="gray", ls=":", lw=0.8, alpha=0.5,
            )
            ax_spec.text(
                1.2, noise_floor * 1.3, f"noise floor σ²={sigma**2}",
                fontsize=7, color="gray",
            )

        ax_spec.set_xlabel("Wavenumber k", fontsize=10)
        ax_spec.set_ylabel("Power", fontsize=10)
        ax_spec.set_title(f"Power spectra of {label} at different noise levels", fontsize=10)
        ax_spec.legend(fontsize=8, ncol=3, loc="upper right")
        ax_spec.set_xlim(1, None)
        ax_spec.grid(True, alpha=0.3, which="both")

    plt.savefig("logs/noise_schedule_demo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: noise_schedule_demo.png")


def visualize_signal_leak(sigma_max_values=[1.0, 10.0, 80.0, 200.0]):
    """
    Demonstrate signal leak: at σ_max too small, you can still recover
    the large-scale mean by spatial averaging.
    """
    
    z500, t2m, precip, q850, t850, u250 = load_field()
    field = z500
    
    field_mean = field.mean()
    field_std = field.std()

    fig, axes = plt.subplots(2, len(sigma_max_values), figsize=(18, 8))
    fig.suptitle(
        "Signal Leak Demonstration: Can the seasonal cycle be recovered from the noised state?\n"
        f"Clean field: mean = {field_mean:.1f}, std = {field_std:.2f}",
        fontsize=13, fontweight="bold",
    )

    rng = np.random.default_rng(123)
    n_ensemble = 200

    for col, smax in enumerate(sigma_max_values):
        noised = add_noise(field, smax)
        snr_global = field_mean**2 / smax**2
        #snr_mode1 = (field_std * np.sqrt(field.size))**2 / (smax**2 * field.size)

        # Top row: noised field
        ax = axes[0, col]
        ax.imshow(noised, cmap="RdBu_r", aspect="auto")
        ax.set_title(f"σ_max = {smax}\nSNR(mean) = {snr_global:.1f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # Bottom row: distribution of spatial means from noised samples
        # If signal leaks, the spatial mean of noised samples clusters around field_mean
        ax2 = axes[1, col]
        means = []
        for _ in range(n_ensemble):
            noised_sample = add_noise(field, smax)
            means.append(noised_sample.mean())
        means = np.array(means)

        # Expected: mean = field_mean, std = σ/√n
        expected_std = smax / np.sqrt(field.size)
        ax2.hist(means, bins=30, density=True, alpha=0.7, color="steelblue",
                 edgecolor="white", linewidth=0.5)
        ax2.axvline(field_mean, color="red", lw=2, label=f"True mean = {field_mean:.1f}")
        ax2.axvline(0, color="black", ls="--", lw=1, label="Zero (pure noise mean)")
        ax2.set_title(
            f"Spatial mean distribution\n"
            f"noise std of mean = {expected_std:.2f}",
            fontsize=9,
        )
        ax2.legend(fontsize=7)
        ax2.set_xlabel("Spatial mean of noised field", fontsize=8)

        # Highlight if signal leaks
        if abs(field_mean) > 3 * expected_std:
            ax2.patch.set_facecolor("#ffe0e0")
            ax2.set_ylabel("SIGNAL LEAKS!", fontsize=10, color="red", fontweight="bold")
        else:
            ax2.patch.set_facecolor("#e0ffe0")
            ax2.set_ylabel("Signal hidden", fontsize=10, color="green", fontweight="bold")

    plt.tight_layout()
    plt.savefig("logs/signal_leak_demo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: signal_leak_demo.png")


def visualize_log_uniform_distribution(sigma_min=0.02, sigma_max=200.0, n=50000):
    """Show the log-uniform distribution vs alternatives."""
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Noise Schedule Distributions Compared", fontsize=14, fontweight="bold")

    # 1. Log-uniform (cBottle)
    sigmas = sample_log_uniform(sigma_min, sigma_max, n, rng)
    axes[0].hist(np.log10(sigmas), bins=80, density=True, alpha=0.8,
                 color="steelblue", edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("log₁₀(σ)", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].set_title(
        f"Log-uniform (cBottle)\nσ ∈ [{sigma_min}, {sigma_max}]\nUniform in log-space",
        fontsize=10,
    )
    axes[0].axvline(np.log10(sigma_min), color="red", ls="--", lw=1.5, label=f"σ_min={sigma_min}")
    axes[0].axvline(np.log10(sigma_max), color="red", ls="--", lw=1.5, label=f"σ_max={sigma_max}")
    axes[0].legend(fontsize=9)

    # 2. Log-normal (EDM / Karras et al.)
    log_sigmas = rng.normal(loc=-1.2, scale=1.2, size=n)  # typical EDM settings
    sigmas_edm = np.exp(log_sigmas)
    axes[1].hist(np.log10(sigmas_edm), bins=80, density=True, alpha=0.8,
                 color="coral", edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel("log₁₀(σ)", fontsize=11)
    axes[1].set_title(
        "Log-normal (EDM/Karras)\nP_mean=-1.2, P_std=1.2\nConcentrated near σ≈0.3",
        fontsize=10,
    )
    axes[1].axvline(np.log10(80), color="black", ls="--", lw=1,
                    label="σ=80 (typical image σ_max)")
    axes[1].legend(fontsize=9)

    # 3. Rectified flow (uniform t → implied σ)
    t = rng.uniform(0.001, 0.999, size=n)
    sigmas_rf = t / (1 - t)
    axes[2].hist(np.log10(sigmas_rf), bins=80, density=True, alpha=0.8,
                 color="mediumpurple", edgecolor="white", linewidth=0.3)
    axes[2].set_xlabel("log₁₀(σ)", fontsize=11)
    axes[2].set_title(
        "Rectified flow (uniform t)\nσ = t/(1-t)\nPoor coverage of high σ",
        fontsize=10,
    )
    axes[2].axvline(np.log10(sigma_max), color="red", ls="--", lw=1.5,
                    label=f"σ={sigma_max} (cBottle σ_max)")
    axes[2].legend(fontsize=9)

    for ax in axes:
        ax.set_xlim(-3, 4)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("logs/noise_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: noise_distributions.png")


# ---------------------------------------------------------------------------
# 5. Run everything
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Log-Uniform Noise Schedule Demo")
    print("=" * 60)

    print("\n1. Generating noised field visualizations & power spectra...")
    visualize_noising(sigma_min=0.02, sigma_max=200.0)

    print("\n2. Demonstrating signal leak at different σ_max values...")
    visualize_signal_leak(sigma_max_values=[1.0, 10.0, 80.0, 200.0])

    print("\n3. Comparing noise schedule distributions...")
    visualize_log_uniform_distribution(sigma_min=0.02, sigma_max=200.0)

    print("\n" + "=" * 60)
    print("All figures saved. Key takeaways:")
    print("  - Z500-like fields need large σ_max to swamp the seasonal cycle")
    print("  - Precipitation needs small σ_min to resolve fine-scale structure")
    print("  - Log-uniform gives equal weight per decade of σ")
    print("  - Rectified flow (uniform t) starves the high-σ regime")
    print("=" * 60)