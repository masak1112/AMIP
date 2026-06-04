import argparse
import torch
import os
import yaml
from datetime import timedelta
from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from common.loss import _weight_for_latitude_vector_without_poles

def plot_loss(losses, filename, key=None):
    # losses in shape [t]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(losses)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Loss')
    ax.set_title(f'{key} loss over time')
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_reconstruction(y_pred, y, filename=None, cmap='twilight_shifted'):
    # y in shape [h w], y_pred in shape [h w]

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))

    vmin = y.min()
    vmax = y.max()

    im0 = axs[0].imshow(y.numpy(), vmin=vmin, vmax=vmax,cmap=cmap, origin='lower')
    im1 = axs[1].imshow(y_pred.numpy(), vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')

    # set the title
    axs[0].set_title(f"True")
    axs[1].set_title(f"Pred")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im0, cax=cbar_ax)
    # save the figure
    if filename is not None:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_result(y_pred, y, filename=None, num_t=5, cmap='twilight_shifted'):
    # y in shape [t h w], y_pred in shape [t h w]

    t_total, h, w = y_pred.shape

    dt = 0
    if num_t != 1:
        dt = t_total // num_t
        if dt == 0:
            num_t = t_total # since t_total < num_t
            dt = 1
        y_pred = y_pred[::dt]
        y = y[::dt]

    fig, axs = plt.subplots(2, num_t, figsize=(num_t*6, 6))

    vmin = y.min()
    vmax = y.max()

    for i in range(num_t):
        if num_t == 1:
            im0 = axs[0].imshow(y[i].numpy(), vmin=vmin, vmax=vmax,cmap=cmap, origin='lower')
            im1 = axs[1].imshow(y_pred[i].numpy(), vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')

            # set the title
            axs[0].set_title(f"True t={(i+1)*dt}")
            axs[1].set_title(f"Pred t={(i+1)*dt}")
        else:
            im0 = axs[0][i].imshow(y[i].numpy(), vmin=vmin, vmax=vmax,cmap=cmap, origin='lower')
            im1 = axs[1][i].imshow(y_pred[i].numpy(), vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')

            # set the title
            axs[0][i].set_title(f"True t={(i+1)*dt}")
            axs[1][i].set_title(f"Pred t={(i+1)*dt}")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im0, cax=cbar_ax)
    # save the figure
    if filename is not None:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_bias(pred, target, save_path=None, title=""):
    # pred, target in shape nlat nlon
    bias = pred - target
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    vmin = target.min()
    vmax = target.max()
    
    bias_min = bias.min()
    bias_max = bias.max()

    bias_scale = max(abs(bias_min), abs(bias_max))

    im0 = axs[0].imshow(pred, cmap='twilight_shifted', vmin=vmin, vmax=vmax, origin='lower')
    axs[0].set_title('Predicted')
    fig.colorbar(im0, ax=axs[0], orientation='horizontal')

    im1 = axs[1].imshow(target, cmap='twilight_shifted', vmin=vmin, vmax=vmax, origin='lower')
    axs[1].set_title('Target')
    fig.colorbar(im1, ax=axs[1], orientation='horizontal')

    im2 = axs[2].imshow(bias, cmap='bwr', vmin=-bias_scale, vmax=bias_scale, origin='lower')
    axs[2].set_title('Bias (Predicted - Target)')
    fig.colorbar(im2, ax=axs[2], orientation='horizontal')

    fig.suptitle(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

def plot_crps(crps, title, t=120, save_path=None):
    plt.figure()
    plt.plot(crps)
    plt.xticks(np.arange(0, t+1, 24), np.arange(0, t//4+1, 6))
    plt.xlabel('Forecast lead time (days)')
    plt.ylabel('CRPS')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_ssr(ssr, title, t=120, save_path=None):
    plt.figure()
    plt.plot(ssr)
    plt.xticks(np.arange(0, t+1, 24), np.arange(0, t//4+1, 6))
    plt.xlabel('Forecast lead time (days)')
    plt.ylabel('SSR')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_spectrum(pred, target, path=None, num_t = 4):
    # pred and target in shape (t, nlat, nlon)
    if pred.shape[0] == 1: # assume t is trivial
        pred = pred.squeeze()
        target = target.squeeze()
        nlat = pred.shape[0]
        nlon = pred.shape[1]

        k_x_pred, power_spectrum_pred = zonal_averaged_power_spectrum(pred, nlon=nlon, nlat=nlat)
        k_x_target, power_spectrum_target = zonal_averaged_power_spectrum(target, nlon=nlon, nlat=nlat)

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(k_x_pred.numpy(), power_spectrum_pred.numpy(), label='Predicted', color='blue')
        ax.plot(k_x_target.numpy(), power_spectrum_target.numpy(), label='Target', color='orange')
        ax.set_xlabel('Zonal wavenumber')
        ax.set_ylabel('Power Spectrum')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Zonal Averaged Power Spectrum')
        ax.legend()
    else:
        t_total = pred.shape[0]

        dt = 0
        if num_t != 1:
            dt = t_total // num_t
            if dt == 0:
                num_t = t_total # since t_total < num_t
                dt = 1
            pred = pred[::dt]
            target = target[::dt]

        nlat = pred.shape[1]
        nlon = pred.shape[2]

        fig, axs = plt.subplots(1, num_t, figsize=(num_t*6, 6))

        for i in range(num_t):
            k_x_pred, power_spectrum_pred = zonal_averaged_power_spectrum(pred[i], nlon=nlon, nlat=nlat)
            k_x_target, power_spectrum_target = zonal_averaged_power_spectrum(target[i], nlon=nlon, nlat=nlat)

            axs[i].plot(k_x_pred.numpy(), power_spectrum_pred.numpy(), label='Predicted', color='blue')
            axs[i].plot(k_x_target.numpy(), power_spectrum_target.numpy(), label='Target', color='orange')
            axs[i].set_xlabel('Zonal wavenumber')
            axs[i].set_ylabel('Power Spectrum')
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].set_title(f'Zonal Averaged Power Spectrum at t={(i+1)*dt}')
            axs[i].legend()

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_spectrum_by_latitude(pred, target, latitudes=None, filename=None):
    # pred and target in shape (nlat, nlon)
    nlat, nlon = pred.shape

    if latitudes is None:
        latitudes = [-85, -60, -30, 0, 30, 60, 85]

    # Reconstruct latitude grid (same as zonal_averaged_power_spectrum)
    lat_end = (nlat - 1) * (360 / nlon) / 2
    latitude_grid = np.linspace(-lat_end, lat_end, nlat)

    # Map requested latitudes to nearest grid indices
    lat_indices = []
    actual_lats = []
    for lat in latitudes:
        idx = np.argmin(np.abs(latitude_grid - lat))
        lat_indices.append(idx)
        actual_lats.append(latitude_grid[idx])

    # Wavenumber axis
    k_x = torch.fft.fftfreq(nlon, d=1/nlon)[:nlon//2]

    # Subplot layout
    n = len(latitudes)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axs = np.atleast_2d(axs)

    for i, (idx, actual_lat) in enumerate(zip(lat_indices, actual_lats)):
        row_idx = i // ncols
        col_idx = i % ncols
        ax = axs[row_idx, col_idx]

        # Per-row spectral computation
        fft_pred = torch.fft.rfft(pred[idx, :], norm='forward')
        fft_target = torch.fft.rfft(target[idx, :], norm='forward')

        power_pred = torch.abs(fft_pred) ** 2
        power_target = torch.abs(fft_target) ** 2

        power_pred = power_pred[:nlon//2]
        power_target = power_target[:nlon//2]
        power_pred[1:] *= 2
        power_target[1:] *= 2

        ax.plot(k_x.numpy(), power_pred.numpy(), label='Predicted', color='blue')
        ax.plot(k_x.numpy(), power_target.numpy(), label='Target', color='orange')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Zonal wavenumber')
        ax.set_ylabel('Power Spectrum')
        ax.set_title(f'Lat = {actual_lat:.1f}\u00b0')
        ax.legend()

    # Hide unused subplots
    for i in range(n, nrows * ncols):
        axs[i // ncols, i % ncols].set_visible(False)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def zonal_averaged_power_spectrum(field,
                                  nlon=360,
                                  nlat=180):
    """
    This function calculates the zonal averaged power spectrum of a given field. It is designed to work with xarray DataArrays or Datasets that have 'lat', 'lon', and optionally 'time' dimensions. The function first transposes the dimensions to ensure 'lat' and 'lon' are the first two dimensions, then performs a Fast Fourier Transform (FFT) along the 'lon' axis to compute the power spectrum. The power spectrum is then averaged over 'lat' and 'time' (if present) to produce the zonal averaged power spectrum.

    Parameters:
    - field, tensor of shape (nlat, nlon)
    """

    field = torch.transpose(field, 0, 1)  # nlon, nlat

    ###########################################################################################
    field_fft = torch.fft.rfft(field, dim=0, norm='forward') # Convention used: the first Fourier coefficient is the mean of the field

    # Compute the power spectrum (squared magnitude of Fourier coefficients)
    power_spectrum = torch.abs(field_fft)**2

    # Define the zonal wavenumbers
    nx = nlon
    k_x = torch.fft.fftfreq(nx, d=1/nx)

    # Only take the positive frequencies (or the first half if using real FFT)
    k_x = k_x[:nx//2]
    power_spectrum = power_spectrum[:nx//2]
    # count the positive frequencies twice except for the first one (zero frequency), because the FFT of a real function is symmetric
    power_spectrum[1:] *= 2
    # multiply by a factor cos(pi latitude[i] / 180) in axis 1
    lat_end = (nlat-1)*(360/nlon) / 2
    latitude = np.linspace(-lat_end, lat_end, nlat)
    weights = _weight_for_latitude_vector_without_poles(latitude)
    weights = torch.from_numpy(weights)
    weights = weights.view(1, -1)
    power_spectrum *= weights 

    power_spectrum_avg = power_spectrum.mean(axis=1)

    return k_x, power_spectrum_avg


def _resolve_variable_in_dataset(datset, variable_name):
    """Locate a variable inside a GetDataset, returning (category, channel_index).

    ``category`` is one of ``"surface"``, ``"diagnostic"``, ``"upper_air"``.
    ``channel_index`` is the position inside the matching list.
    """
    if variable_name in datset.surface_variables:
        return "surface", datset.surface_variables.index(variable_name)
    if variable_name in datset.diagnostic_variables:
        return "diagnostic", datset.diagnostic_variables.index(variable_name)
    if variable_name in datset.upper_air_variables:
        return "upper_air", datset.upper_air_variables.index(variable_name)
    raise ValueError(
        f"Variable '{variable_name}' not found among surface "
        f"{datset.surface_variables}, diagnostic {datset.diagnostic_variables}, "
        f"or upper_air {datset.upper_air_variables}."
    )


def _truth_field_from_batch(datset, batch, category, channel_idx, level_idx=None):
    """Pull a single (h, w) denormalized truth field out of a dataset batch.

    Expects the batch tuple produced when ``return_calendar`` is set:
    ``(surface_t, upper_air_t, diagnostic_t, surface_t1, upper_air_t1,
       diagnostic_t1, varying_boundary_data, calendar)``.
    """
    surface_t, upper_air_t, diagnostic_t, *_ = batch
    if category == "surface":
        surface_denorm = datset.surface_inv_transform(surface_t.unsqueeze(0))
        return surface_denorm[0, channel_idx]
    if category == "diagnostic":
        diag_denorm = datset.diagnostic_inv_transform(diagnostic_t.unsqueeze(0))
        return diag_denorm[0, channel_idx]
    if category == "upper_air":
        if level_idx is None:
            raise ValueError("level_idx required for upper_air variables.")
        ua_denorm = datset.upper_air_inv_transform(upper_air_t.unsqueeze(0))
        return ua_denorm[0, channel_idx, level_idx]
    raise ValueError(f"Unknown category {category}")


# Maps the dataset's category to the on-disk rollout subdirectory.
ROLLOUT_SUBDIR = {
    "surface": "surface",
    "diagnostic": "diagnostic",
    "upper_air": "multilevel",
}


def _load_prediction(rollout_root, member, category, variable_name, year,
                     month=1, day=1, level_idx=None):
    """Load a single (h, w) prediction field from a rollout NetCDF file."""
    subdir = ROLLOUT_SUBDIR[category]
    path = os.path.join(
        rollout_root, member, subdir, variable_name,
        f"{variable_name}_{year:04d}{month:02d}.nc",
    )
    ds = xr.open_dataset(path, engine="h5netcdf")
    da = ds[variable_name]
    # find the day in the file (the file is monthly, daily-stepped)
    time_index = int(np.where(ds.time.dt.day.values == day)[0][0])
    if category == "upper_air":
        if level_idx is None:
            raise ValueError("level_idx required for upper_air variables.")
        field = da.isel(time=time_index, plev=level_idx).values
    else:
        field = da.isel(time=time_index).values
    ds.close()
    return torch.tensor(field)


def plot_pred_truth_multi_year(
    variable_name,
    dataconfig,
    dataset_cls,
    years=(1981, 1986, 1991, 1996, 2001, 2006),
    member="member_19800101",
    rollout_root="/glade/campaign/univ/uchi0014/ayz/rollouts_new",
    level_idx=None,
    month=1,
    day=1,
    cmap="twilight_shifted",
    save_path=None,
):
    """Compare model rollout predictions to the true state for one variable
    across a set of years, plotting maps and zonal power spectra side-by-side.

    Parameters
    ----------
    variable_name : str
        Name as listed in the data config (e.g. ``"PRATEsfc_24h"``,
        ``"2m_temperature"``, ``"temperature"``).
    dataconfig : dict
        Data section of the YAML config (used to build a ``GetDataset``).
    dataset_cls : class
        ``GetDataset`` (or compatible) class; called as
        ``dataset_cls(dataconfig, year_start=Y, year_end=Y+1)`` for each year.
    years : iterable of int
        Years to evaluate. The Jan-``day`` field of each year is plotted.
    member : str
        Rollout member directory under ``rollout_root``.
    rollout_root : str
        Root directory containing the model rollouts.
    level_idx : int, optional
        Required if ``variable_name`` is an upper-air variable. Index into
        ``dataconfig["levels"]``.
    month, day : int
        Calendar day to plot (default Jan 1).
    cmap : str
        Colormap for the field maps.
    save_path : str, optional
        If given, save the figure here instead of showing it.
    """
    years = list(years)
    n_years = len(years)

    fig, axs = plt.subplots(n_years, 3, figsize=(18, 4 * n_years),
                            squeeze=False)

    pretty_var = variable_name
    if level_idx is not None:
        pretty_var = f"{variable_name} (level idx {level_idx})"

    for row, year in enumerate(years):
        datset = dataset_cls(dataconfig, year_start=year, year_end=year + 1)
        category, channel_idx = _resolve_variable_in_dataset(datset, variable_name)

        # Dataset starts at year_start Jan 1 hour 0 with 6-hour steps.
        if month == 1 and day == 1:
            idx = 0
        else:
            target = datset.datetime_class(year, month, day)
            delta = target - datset.start_date
            idx = int(delta.days * 4)

        batch = datset[idx]
        truth = _truth_field_from_batch(datset, batch, category, channel_idx,
                                         level_idx=level_idx)

        pred = _load_prediction(rollout_root, member, category, variable_name,
                                year, month=month, day=day, level_idx=level_idx)

        vmin = float(truth.min())
        vmax = float(truth.max())

        ax_true, ax_pred, ax_spec = axs[row]

        im0 = ax_true.imshow(truth.numpy(), vmin=vmin, vmax=vmax, cmap=cmap,
                              origin="lower")
        ax_true.set_title(f"True {pretty_var} {year}-{month:02d}-{day:02d}")
        plt.colorbar(im0, ax=ax_true, fraction=0.025, pad=0.02)

        im1 = ax_pred.imshow(pred.numpy(), vmin=vmin, vmax=vmax, cmap=cmap,
                              origin="lower")
        ax_pred.set_title(f"Pred {pretty_var} {year}-{month:02d}-{day:02d}")
        plt.colorbar(im1, ax=ax_pred, fraction=0.025, pad=0.02)

        nlat, nlon = truth.shape
        k_p, p_pred = zonal_averaged_power_spectrum(pred, nlon=nlon, nlat=nlat)
        k_t, p_true = zonal_averaged_power_spectrum(truth, nlon=nlon, nlat=nlat)
        ax_spec.plot(k_p.numpy(), p_pred.numpy(), label="Pred", color="blue")
        ax_spec.plot(k_t.numpy(), p_true.numpy(), label="True", color="orange")
        ax_spec.set_xscale("log")
        ax_spec.set_yscale("log")
        ax_spec.set_xlabel("Zonal wavenumber")
        ax_spec.set_ylabel("Power")
        ax_spec.set_title(f"Spectrum {year}-{month:02d}-{day:02d}")
        ax_spec.legend()

    fig.suptitle(f"{pretty_var} — member {member}", y=1.001, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _load_prediction_from_cache(cache, rollout_root, member, category,
                                 variable_name, year, month, day, level_idx=None):
    """Like ``_load_prediction`` but reuses an open monthly NetCDF per (year, month).

    ``cache`` is a dict ``{(year, month): xr.Dataset}`` that the caller owns and
    is responsible for closing.
    """
    key = (year, month)
    if key not in cache:
        path = os.path.join(
            rollout_root, member, ROLLOUT_SUBDIR[category], variable_name,
            f"{variable_name}_{year:04d}{month:02d}.nc",
        )
        cache[key] = xr.open_dataset(path, engine="h5netcdf")
    ds = cache[key]
    time_index = int(np.where(ds.time.dt.day.values == day)[0][0])
    if category == "upper_air":
        if level_idx is None:
            raise ValueError("level_idx required for upper_air variables.")
        field = ds[variable_name].isel(time=time_index, plev=level_idx).values
    else:
        field = ds[variable_name].isel(time=time_index).values
    return torch.tensor(field)


def plot_avg_spectrum_range(
    variable_name,
    dataconfig,
    dataset_cls,
    year_start,
    year_end,
    n_samples=100,
    member="member_19800101",
    rollout_root="/glade/campaign/univ/uchi0014/ayz/rollouts_new",
    level_idx=None,
    hour=0,
    save_path=None,
    verbose=False,
):
    """Average the zonal power spectrum across many daily snapshots in a year
    range, separately for the rollout and the true state, then plot both.

    The function samples ``n_samples`` full days (uniformly) from
    ``[year_start, year_end)``, computes ``zonal_averaged_power_spectrum`` for
    the prediction and the truth at each day, and averages the spectra across
    snapshots.

    Parameters
    ----------
    variable_name : str
        Variable name as listed in the data config.
    dataconfig : dict
        Data section of the YAML config.
    dataset_cls : class
        ``GetDataset`` (or compatible) class.
    year_start, year_end : int
        End-exclusive year range to sample from.
    n_samples : int
        Number of daily snapshots to average over.
    member : str
        Rollout member directory under ``rollout_root``.
    rollout_root : str
        Root directory containing rollouts.
    level_idx : int, optional
        Required for upper-air variables. Index into ``dataconfig['levels']``.
    hour : int
        Hour of day at which to compare (default 0; rollouts are daily at h=0).
    save_path : str, optional
        If given, save the figure here instead of showing it.
    verbose : bool
        Print progress.
    """
    datset = dataset_cls(dataconfig, year_start=year_start, year_end=year_end)
    category, channel_idx = _resolve_variable_in_dataset(datset, variable_name)

    # Pick n_samples evenly-spaced full days within [year_start, year_end).
    total_days = (datset.end_date - datset.start_date).days
    if n_samples > total_days:
        n_samples = total_days
    day_offsets = np.linspace(0, total_days - 1, n_samples).astype(int)

    # Dataset is stepped at data_timedelta_hours; pick the right index for ``hour``.
    step_h = int(dataconfig.get("data_timedelta_hours", 6))
    steps_per_day = 24 // step_h
    hour_step = hour // step_h

    cache = {}
    pred_spectra = []
    true_spectra = []
    k_axis = None

    for i, day_offset in enumerate(day_offsets):
        target = datset.start_date + timedelta(days=int(day_offset),
                                                hours=hour)
        y, m, d = target.year, target.month, target.day

        idx = int(day_offset) * steps_per_day + hour_step
        batch = datset[idx]
        truth = _truth_field_from_batch(datset, batch, category, channel_idx,
                                         level_idx=level_idx)

        pred = _load_prediction_from_cache(
            cache, rollout_root, member, category, variable_name,
            y, m, d, level_idx=level_idx,
        )

        nlat, nlon = truth.shape
        k, p_pred = zonal_averaged_power_spectrum(pred, nlon=nlon, nlat=nlat)
        _, p_true = zonal_averaged_power_spectrum(truth, nlon=nlon, nlat=nlat)
        pred_spectra.append(p_pred)
        true_spectra.append(p_true)
        k_axis = k

        if verbose and ((i + 1) % max(1, n_samples // 10) == 0 or i == n_samples - 1):
            print(f"  [{i+1}/{n_samples}] {y:04d}-{m:02d}-{d:02d}")

    for ds in cache.values():
        ds.close()

    avg_pred = torch.stack(pred_spectra).mean(dim=0)
    avg_true = torch.stack(true_spectra).mean(dim=0)

    pretty_var = variable_name
    if level_idx is not None:
        pretty_var = f"{variable_name} (level idx {level_idx})"

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(k_axis.numpy(), avg_pred.numpy(), label="Pred", color="blue")
    ax.plot(k_axis.numpy(), avg_true.numpy(), label="True", color="orange")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Zonal wavenumber")
    ax.set_ylabel("Power (averaged)")
    ax.set_title(
        f"{pretty_var}: avg zonal spectrum, {year_start}–{year_end-1} "
        f"({n_samples} snapshots, member {member})"
    )
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return k_axis, avg_pred, avg_true