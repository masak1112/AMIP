"""
Dataset for weather/climate forecasting models.

Loads atmospheric state data from per-timestep HDF5 files, applies normalization,
and returns (input, target) pairs for training or multi-step rollout sequences
for autoregressive inference/validation.

Data layout per file: ``{year}_{index:04d}.h5`` containing an ``input`` group
with one dataset per variable (plus ``time``).  Variables are split into:

- **Upper-air** (3-D): variables on pressure levels (e.g. temperature, wind)
- **Surface** (2-D): single-level fields (e.g. 2m temperature, surface pressure)
- **Diagnostic** (2-D, output only): radiation fluxes, precipitation, etc.
- **Varying boundary** (2-D, input only): SST, sea-ice, TOA solar forcing
- **Constant boundary**: land-sea mask, surface geopotential (loaded once)

Calendar-aware date handling is provided via ``cftime`` so that non-standard
calendars (no-leap, 360-day, etc.) used by different climate models are supported.
"""

import sys

import cftime
import h5py
import numpy as np
import torch
import xarray as xr
from datetime import timedelta
from itertools import product
from os.path import join
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_data_given_path(path, variables):
    """Read selected variables from an HDF5 file and return as a stacked array.

    Parameters
    ----------
    path : str
        Path to an HDF5 file with an ``input`` group.
    variables : list[str]
        Variable names to extract from the ``input`` group.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_variables, ...)``.
    """
    with h5py.File(path, 'r') as f:
        data = {
            sub_key: np.array(value)
            for sub_key, value in f['input'].items()
            if sub_key in variables + ['time']
        }
    return np.stack([data[v] for v in variables], axis=0)


def get_out_path(root_dir, year, file_idx):
    """Build the HDF5 file path for a given year and timestep index."""
    return join(root_dir, f'{year}_{file_idx:04}.h5')


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

CALENDAR_TO_DATETIME = {
    'standard': cftime.DatetimeGregorian,
    'Gregorian': cftime.DatetimeGregorian,
    'noleap': cftime.DatetimeNoLeap,
    '365_day': cftime.DatetimeNoLeap,
    'proleptic_gregorian': cftime.DatetimeProlepticGregorian,
    'all_leap': cftime.DatetimeAllLeap,
    '366_day': cftime.DatetimeAllLeap,
    '360_day': cftime.Datetime360Day,
    'julian': cftime.DatetimeJulian,
}


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def get_data_loader(params, 
                    year_start: int, 
                    year_end: int, 
                    num_inferences: int = 0,
                    train: bool = True, 
                    validate: bool = False):
    """Create a DataLoader (and sampler) for training or evaluation.

    Parameters
    ----------
    params : dict-like
        Full dataset/config parameters forwarded to :class:`GetDataset`.
    distributed : bool
        Whether to use a ``DistributedSampler``.
    year_start, year_end : int
        Date range for the dataset (end-exclusive).
    num_inferences : int
        Number of evenly spaced inference samples to draw (0 = all).
    train : bool
        Training mode flag — controls shuffling and return values.
    validate : bool
        If *True* (and ``train`` is *False*), load multi-step targets.

    Returns
    -------
    tuple
        ``(dataloader, dataset)``.
    """
    dataset = GetDataset(params, 
                         year_start=year_start,
                         year_end=year_end,
                         num_inferences=num_inferences,
                         train=train, 
                         validate=validate)

    dataloader = DataLoader(
        dataset,
        batch_size=int(params["batch_size"]),
        num_workers=params["num_data_workers"],
        shuffle=train,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GetDataset(Dataset):
    """PyTorch Dataset for atmospheric reanalysis data.

    Each sample consists of an input atmospheric state at time *t* and
    (during training) the target state at time *t + dt*.  For validation /
    inference the dataset can return multi-step target sequences for
    autoregressive rollout evaluation.

    Parameters
    ----------
    params : dict-like
        Configuration object (typically loaded from YAML) containing at least:

        - ``data_dir``: root directory with per-timestep HDF5 files
        - ``year_start``, ``year_end``: date range (end-exclusive)
        - ``train``: bool — training vs. inference mode
        - ``calendar``: calendar type string (e.g. ``'noleap'``)
        - ``timedelta_hours``: forecast step in hours
        - ``data_timedelta_hours``: temporal resolution of files in hours
        - ``has_year_zero``: bool for cftime year-zero support
        - ``surface_variables``, ``upper_air_variables``, ``diagnostic_variables``: list of variable names
        - ``constant_boundary_variables``, ``varying_boundary_variables``: list of forcing field names
        - ``forecast_lead_times``: list of lead-time steps for evaluation
        - ``levels``: pressure levels to use
        - ``horizontal_resolution``: ``(nlat, nlon)``
        - ``num_inferences``: number of evenly-spaced inference starts (0 = all)
        - ``epsilon_factor``: input noise scale (0 disables noise)
        - ``predict_delta``: if True, targets are state increments
        - ``mean_path``, ``std_path``: paths to NetCDF files with normalization stats
    train : bool
        If True, return (input, target) pairs for training. 
    validate : bool
        If True and not training, load full target sequences.
    """

    def __init__(self, params: dict, 
                 year_start: int, 
                 year_end: int, 
                 num_inferences: int = 0,
                 train: bool = True, 
                 validate: bool = False):
        self.params = params
        self.data_dir = params['data_dir']
        self.train = train
        self.num_inferences = num_inferences
        self.has_year_zero = params['has_year_zero']
        self.epsilon_factor = params['epsilon_factor']
        self.diagnostic_input = params.get('diagnostic_input', False) # whether to use diagnostic as prognostic
        self.validate = validate if not self.train else False
        self.autoencoder = params.get('autoencoder', False)

        if not self.train and not self.params['forecast_lead_times']:
            self.params['forecast_lead_times'] = [1]

        self.mask_fill = params.get('mask_fill', {
            'land_sea_mask': 0.,
            'sea_surface_temperature': 270.,
            'sea_ice_cover': 0.,
        })

        # Calendar / time setup
        self.year_start = year_start
        self.year_end = year_end
        self.calendar = params["calendar"]
        self.timedelta_hours = params["timedelta_hours"]
        self.data_timedelta_hours = params["data_timedelta_hours"]
        self.datetime_class = CALENDAR_TO_DATETIME[self.calendar]

        days, hours = divmod(self.timedelta_hours, 24)
        self.timedelta = (
            self.datetime_class(1, 1, 1 + days, hour=hours)
            - self.datetime_class(1, 1, 1, hour=0)
        )

        # Variable lists
        self.surface_variables = params["surface_variables"]

        # disabled
        self.land_variables = []
        self.ocean_variables = []

        if self.land_variables:
            if any(v in self.surface_variables for v in self.land_variables):
                raise ValueError('land variables cannot be in surface variables.')
            self.surface_variables = self.surface_variables + self.land_variables

        if self.ocean_variables:
            if any(v in self.surface_variables for v in self.ocean_variables):
                raise ValueError('ocean variables cannot be in surface variables.')
            self.surface_variables = self.surface_variables + self.ocean_variables

        self.upper_air_variables = params["upper_air_variables"] 
        self.constant_boundary_variables = params["constant_boundary_variables"] 
        self.varying_boundary_variables = params["varying_boundary_variables"] 
        self.diagnostic_variables = params['diagnostic_variables']

        # Date range
        self.dates, self.start_date, self.end_date = self._get_dates(
            hour_step=params["data_timedelta_hours"]
        )

        # Constant boundary fields (e.g. land-sea mask, orography)
        self.constant_boundary_data, self.land_mask = self._load_constant_boundary_data()
        if torch.any(torch.isnan(self.constant_boundary_data)):
            raise ValueError('Constant boundary data contains NaN values.')

        # Inference index selection
        max_inference_idx = (
            len(self.dates)
            - max(self.params['forecast_lead_times']) * self.timedelta_hours // self.data_timedelta_hours
        )
        if self.num_inferences > 0:
            self.inference_idxs = np.linspace(0, max_inference_idx, num=self.num_inferences + 1, dtype=int)
        else:
            self.inference_idxs = np.arange(0, max_inference_idx)

        # Pressure levels
        if len(params['levels']) > 0:
            self.levels = np.array(params['levels'])
        else:
            raise ValueError('levels must be explicitly specified in config file.')

        # Load normalization statistics
        mean_path = params["mean_path"]
        std_path = params["std_path"]
        self.surface_mean, self.surface_std = self._load_mean_std(
            mean_path,
            std_path,
            self.surface_variables, upper_air=False,
        )
        self.upper_air_mean, self.upper_air_std = self._load_mean_std(
            mean_path,
            std_path,
            self.upper_air_variables,
        )

        if self.params['predict_delta']:
            _, self.surface_delta_std = self._load_mean_std(
                mean_path,
                std_path,
                self.surface_variables, upper_air=False,
            )
            _, self.upper_air_delta_std = self._load_mean_std(
                mean_path,
                std_path,
                self.upper_air_variables,
            )

        self.varying_boundary_mean, self.varying_boundary_std = self._load_mean_std(
            mean_path,
            std_path,
            self.varying_boundary_variables, upper_air=False,
        )

        if self.diagnostic_variables:
            self.diagnostic_mean, self.diagnostic_std = self._load_mean_std(
                mean_path,
                std_path,
                self.diagnostic_variables, upper_air=False,
            )

        self._build_variable_lists()

        if self.epsilon_factor > 0.:
            torch.manual_seed(0)
            
        self.print_info()

    def print_info(self):
        print(f"Dataset info:")
        print(f"  Date range: {self.start_date} to {self.end_date} ({len(self.dates)} total hours)")
        print(f"  Number of inference samples: {len(self.inference_idxs)}")
        print(f"  Upper-air variables: {self.upper_air_variables}")
        print(f"  Surface variables: {self.surface_variables}")
        print(f"  Diagnostic variables: {self.diagnostic_variables}")
        print(f"  Varying boundary variables: {self.varying_boundary_variables}")
        print(f"  Constant boundary variables: {self.constant_boundary_variables}")
        print(f"  Pressure levels: {self.levels}")
        print(f"  Horizontal resolution: {self.params['horizontal_resolution']}")
        print(f"  Forecast lead times (hours): {self.params['forecast_lead_times']}")
        print(f"  Diagnostic input: {self.diagnostic_input}")

    # ------------------------------------------------------------------
    # Variable list bookkeeping
    # ------------------------------------------------------------------

    def _build_variable_lists(self, level_units='.0'):
        """Build ordered variable name lists for input and output tensors.

        Sets ``self.variable_list_in`` and ``self.variable_list_out`` as well
        as ``self.upper_air_len`` (the number of upper-air channels after
        flattening variables x levels).
        """
        self.variable_list_out = []
        for variable, level in product(self.upper_air_variables, self.levels):
            self.variable_list_out.append(f'{variable}_{int(level)}{level_units}')
        self.upper_air_len = len(self.variable_list_out)
        self.variable_list_out.extend(self.surface_variables)
        self.variable_list_in = self.variable_list_out.copy()
        self.variable_list_out.extend(self.diagnostic_variables)
        self.variable_list_in.extend(self.varying_boundary_variables)

        if self.diagnostic_input: # add diagnostic variables to input list if configured
            self.variable_list_in.extend(self.diagnostic_variables)

    # ------------------------------------------------------------------
    # Reshaping / masking
    # ------------------------------------------------------------------

    def _reshape_and_mask_variables(self, data_array, out=False):
        """Reshape a flat channel array into (upper_air, surface, [extra]) tensors.

        For **input** (``out=False``), the extra tensor is ``varying_boundary``.
        For **output** (``out=True``), the extra tensor is ``diagnostic``.

        NaN values in surface / boundary / diagnostic fields are filled using
        ``self.mask_fill``.

        Parameters
        ----------
        data_array : np.ndarray
            Shape ``(n_channels, nlat, nlon)`` in the order defined by
            ``variable_list_in`` (input) or ``variable_list_out`` (output).
        out : bool
            Whether this is an output (target) array.

        Returns
        -------
        tuple of torch.Tensor
            ``(upper_air, surface)`` or ``(upper_air, surface, extra)``
        """
        nlat, nlon = self.params['horizontal_resolution']
        n_ua = len(self.upper_air_variables)
        n_lev = len(self.levels)
        n_sfc = len(self.surface_variables)

        upper_air = torch.tensor(
            data_array[:self.upper_air_len].reshape(n_ua, n_lev, nlat, nlon)
        ).to(torch.float32)

        surface = torch.tensor(
            data_array[self.upper_air_len:self.upper_air_len + n_sfc].reshape(n_sfc, nlat, nlon)
        ).to(torch.float32)
        surface = self._fill_mask(surface, self.surface_variables,
                                  self.land_variables + self.ocean_variables)

        offset = self.upper_air_len + n_sfc

        if out:
            if self.diagnostic_variables:
                n_diag = len(self.diagnostic_variables)
                diagnostic = torch.tensor(
                    data_array[offset:offset + n_diag].reshape(n_diag, nlat, nlon)
                ).to(torch.float32)
                diagnostic = self._fill_mask(diagnostic, self.diagnostic_variables)
                return upper_air, surface, diagnostic
            return upper_air, surface
        else:
            if self.varying_boundary_variables:
                n_bnd = len(self.varying_boundary_variables)
                varying_boundary = torch.tensor(
                    data_array[offset:offset + n_bnd].reshape(n_bnd, nlat, nlon)
                ).to(torch.float32)
                varying_boundary = self._fill_mask(varying_boundary, self.varying_boundary_variables)

                if self.diagnostic_input:
                    offset += n_bnd
                    n_diag = len(self.diagnostic_variables)
                    diagnostic = torch.tensor(
                        data_array[offset:offset + n_diag].reshape(n_diag, nlat, nlon)
                    ).to(torch.float32)
                    diagnostic = self._fill_mask(diagnostic, self.diagnostic_variables)
                    return upper_air, surface, diagnostic, varying_boundary
                else:
                    return upper_air, surface, varying_boundary
            return upper_air, surface

    def _fill_mask(self, data, variables, optional_variables=None):
        """Replace NaN values with predefined fill values from ``self.mask_fill``.

        Parameters
        ----------
        data : torch.Tensor
            Shape ``(n_vars, nlat, nlon)``.
        variables : list[str]
            Variable names corresponding to the first dimension.
        optional_variables : list[str] or None
            If provided, only fill NaNs for variables in this subset.
        """
        for i, var in enumerate(variables):
            if optional_variables and var not in optional_variables:
                continue
            nans = torch.isnan(data[i])
            if torch.any(nans):
                data[i] = data[i].masked_fill(nans, self.mask_fill[var])
        return data

    # ------------------------------------------------------------------
    # Date handling
    # ------------------------------------------------------------------

    def _get_dates(self, hour_step=6.):
        """Generate an array of hour-offsets from ``year_start`` to ``year_end``.

        Returns
        -------
        tuple
            ``(date_offsets, start_date, end_date)`` where *date_offsets* is a
            1-D numpy array of hours since *start_date*.
        """
        start_date = self.datetime_class(self.year_start, 1, 1)
        end_date = self.datetime_class(self.year_end, 1, 1)
        hours = (end_date - start_date).days * 24.
        date_range = np.arange(0., hours, hour_step)
        return date_range, start_date, end_date

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def _get_data(self, data_datetime, out=False, variable_list=None):
        """Load raw data for a single datetime from disk.

        Parameters
        ----------
        data_datetime : cftime datetime
            Timestamp to load.
        out : bool
            If True and *variable_list* is None, use output variable list.
        variable_list : list[str] or None
            Explicit variable list override.

        Returns
        -------
        np.ndarray
            Shape ``(n_channels, ...)``.
        """
        data_year = data_datetime.year
        seconds_into_year = int(
            (data_datetime - self.datetime_class(data_year, 1, 1, hour=0,
                                                  has_year_zero=self.has_year_zero)).total_seconds()
        )
        data_idx = seconds_into_year // 3600 // self.data_timedelta_hours
        data_file_path = get_out_path(self.data_dir, data_year, data_idx)

        if variable_list:
            return get_data_given_path(data_file_path, variable_list)
        if out:
            return get_data_given_path(data_file_path, self.variable_list_out)
        return get_data_given_path(data_file_path, self.variable_list_in)

    def _load_constant_boundary_data(self):
        """Load and normalize constant boundary fields (e.g. land-sea mask).

        Returns
        -------
        tuple
            ``(constant_boundary_data, land_mask)`` both as float32 tensors.
        """
        raw = torch.tensor(
            self._get_data(self.start_date, variable_list=self.constant_boundary_variables)
        ).to(torch.float32)
        raw = self._fill_mask(raw, self.constant_boundary_variables)
        land_mask = raw[(np.array(self.constant_boundary_variables) == 'land_sea_mask').tolist()].clone().detach()
        mean = torch.mean(raw, dim=(1, 2))
        std = torch.std(raw, dim=(1, 2))
        normalized = (raw - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
        return normalized, land_mask

    # ------------------------------------------------------------------
    # Normalization statistics
    # ------------------------------------------------------------------

    def _load_mean_std(self, mean_file, std_file, datavars, upper_air=True):
        """Load mean and standard deviation tensors from NetCDF files.

        Parameters
        ----------
        mean_file, std_file : str
            Paths to NetCDF files containing per-variable statistics.
        datavars : list[str]
            Variable names to extract.
        upper_air : bool
            If True, select only the configured pressure levels along the
            vertical (``level``) dimension.

        Returns
        -------
        tuple
            ``(mean, std)`` tensors.
        """
        if upper_air:
            with xr.open_dataset(mean_file, engine = "h5netcdf") as ds:
                level_mask = xr.DataArray(
                    data=[lev in self.levels for lev in ds['level'].values], dims=['level']
                )
                mean = torch.stack([
                    torch.tensor(ds[var].where(level_mask, drop=True).values).to(torch.float32)
                    for var in datavars
                ], dim=0)
            with xr.open_dataset(std_file, engine = "h5netcdf") as ds:
                level_mask = xr.DataArray(
                    data=[lev in self.levels for lev in ds['level'].values], dims=['level']
                )
                std = torch.stack([
                    torch.tensor(ds[var].where(level_mask, drop=True).values).to(torch.float32)
                    for var in datavars
                ], dim=0)
        else:
            with xr.open_dataset(mean_file, engine = "h5netcdf") as ds:
                mean = torch.stack([
                    torch.tensor(ds[var].values).to(torch.float32) for var in datavars
                ], dim=0)
            with xr.open_dataset(std_file, engine = "h5netcdf") as ds:
                std = torch.stack([
                    torch.tensor(ds[var].values).to(torch.float32) for var in datavars
                ], dim=0)
        return mean, std

    # ------------------------------------------------------------------
    # Transforms (normalize / denormalize)
    # ------------------------------------------------------------------

    def surface_transform(self, data):
        """Normalize surface fields: ``(x - mean) / std``."""
        return (data - self.surface_mean.reshape(-1, 1, 1)) / self.surface_std.reshape(-1, 1, 1)

    def diagnostic_transform(self, data):
        """Normalize diagnostic fields."""
        return (data - self.diagnostic_mean.reshape(-1, 1, 1)) / self.diagnostic_std.reshape(-1, 1, 1)

    def boundary_transform(self, data):
        """Normalize varying boundary fields."""
        return (data - self.varying_boundary_mean.reshape(-1, 1, 1)) / self.varying_boundary_std.reshape(-1, 1, 1)

    def upper_air_transform(self, data):
        """Normalize upper-air fields (shape: ``(n_vars, n_levels, nlat, nlon)``)."""
        n = len(self.upper_air_variables)
        return (data - self.upper_air_mean.reshape(n, -1, 1, 1)) / self.upper_air_std.reshape(n, -1, 1, 1)

    def surface_inv_transform(self, data):
        """Denormalize surface fields (expects leading batch dim)."""
        return data * self.surface_std.reshape(1, -1, 1, 1) + self.surface_mean.reshape(1, -1, 1, 1)

    def upper_air_inv_transform(self, data):
        """Denormalize upper-air fields (expects leading batch dim)."""
        n = len(self.upper_air_variables)
        return data * self.upper_air_std.reshape(1, n, -1, 1, 1) + self.upper_air_mean.reshape(1, n, -1, 1, 1)

    def diagnostic_inv_transform(self, data):
        """Denormalize diagnostic fields (expects leading batch dim)."""
        return data * self.diagnostic_std.reshape(1, -1, 1, 1) + self.diagnostic_mean.reshape(1, -1, 1, 1)

    def surface_delta_transform(self, data):
        """Normalize surface increments (zero-mean assumed)."""
        return data / self.surface_delta_std.reshape(-1, 1, 1)

    def upper_air_delta_transform(self, data):
        """Normalize upper-air increments (zero-mean assumed)."""
        n = len(self.upper_air_variables)
        return data / self.upper_air_delta_std.reshape(n, -1, 1, 1)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.inference_idxs)

    def __getitem__(self, index):
        """Return a sample for training, validation, or inference.

        Returns
        -------
        tuple of torch.Tensor
            The exact contents depend on mode:

            **Training** (``self.train``):
              ``(surface_t, upper_air_t, surface_t1, upper_air_t1,
              [diagnostic_t1,] varying_boundary)``

            **Validation** (``self.validate`` and ``forecast_lead_times``):
              ``(surface_t, upper_air_t, targets_surface, targets_upper_air,
              [targets_diagnostic,] [targets_delta_surface, targets_delta_upper_air,]
              varying_boundary, start_time_tensor)``

            **Inference** (``forecast_lead_times`` without validate):
              ``(surface_t, upper_air_t, varying_boundary)``

            **Single-step eval** (no ``forecast_lead_times``):
              Same as training format.
        """
        lead_times = self.params['forecast_lead_times']
        has_boundary = len(self.varying_boundary_variables) > 0
        has_diagnostic = len(self.diagnostic_variables) > 0

        # ---- Training ----
        if self.train:
            return self._getitem_train(index, has_boundary, has_diagnostic)

        # ---- Autoregressive inference / validation ----
        if lead_times:
            return self._getitem_autoregressive(index, lead_times, has_boundary, has_diagnostic)

        # ---- Single-step evaluation ----
        return self._getitem_single_step(index, has_boundary)

    def _getitem_train(self, index, has_boundary, has_diagnostic):
        """Build a single training sample (input at t, target at t+dt)."""
        start_time = self.start_date + timedelta(hours=self.dates[index])
        end_time = self.start_date + timedelta(hours=self.dates[index] + self.timedelta_hours)

        data_in = self._get_data(start_time, out=False)

        if has_boundary:
            if self.diagnostic_input:
                upper_air_t, surface_t, diagnostic_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out=False)
            else:
                upper_air_t, surface_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out=False)
        else:
            upper_air_t, surface_t = self._reshape_and_mask_variables(data_in, out=False)

        if self.autoencoder:
            if self.diagnostic_input:
                return surface_t, upper_air_t
            else:
                return surface_t, upper_air_t, diagnostic_t

        data_out = self._get_data(end_time, out=True)

        if has_diagnostic:
            upper_air_t1, surface_t1, diagnostic_t1 = self._reshape_and_mask_variables(data_out, out=True)
        else:
            upper_air_t1, surface_t1 = self._reshape_and_mask_variables(data_out, out=True)

        # Normalize
        if self.params['predict_delta']:
            surface_t1 = self.surface_delta_transform(surface_t1 - surface_t)
            upper_air_t1 = self.upper_air_delta_transform(upper_air_t1 - upper_air_t)
            surface_t = self.surface_transform(surface_t)
            upper_air_t = self.upper_air_transform(upper_air_t)
        else:
            surface_t = self.surface_transform(surface_t)
            surface_t1 = self.surface_transform(surface_t1)
            upper_air_t = self.upper_air_transform(upper_air_t)
            upper_air_t1 = self.upper_air_transform(upper_air_t1)

        if has_diagnostic:
            diagnostic_t1 = self.diagnostic_transform(diagnostic_t1)
        if has_boundary:
            varying_boundary_data = self.boundary_transform(varying_boundary_data)
        if self.diagnostic_input:
            diagnostic_t = self.diagnostic_transform(diagnostic_t)

        # Optional input noise
        if self.epsilon_factor > 0.:
            surface_t = self._add_input_noise(surface_t)
            upper_air_t = self._add_input_noise(upper_air_t)

        self._check_nans(surface_t=surface_t, upper_air_t=upper_air_t,
                         varying_boundary_data=varying_boundary_data if has_boundary else None,
                         surface_t1=surface_t1, upper_air_t1=upper_air_t1,
                         diagnostic_t1=diagnostic_t1 if has_diagnostic else None,
                         diagnostic_t=diagnostic_t if self.diagnostic_input else None)

        if self.diagnostic_input:
            return surface_t, upper_air_t, diagnostic_t, surface_t1, upper_air_t1, diagnostic_t1, varying_boundary_data
        if has_diagnostic:
            return surface_t, upper_air_t, surface_t1, upper_air_t1, diagnostic_t1, varying_boundary_data
        return surface_t, upper_air_t, surface_t1, upper_air_t1, varying_boundary_data

    def _getitem_autoregressive(self, index, lead_times, has_boundary, has_diagnostic):
        """Build an autoregressive sample with multi-step boundary forcing."""
        start_time = self.start_date + timedelta(hours=self.dates[index])
        data_in = self._get_data(start_time, out=False)

        if has_boundary:
            if self.diagnostic_input:
                upper_air_t, surface_t, diagnostic_t, varying_boundary_t = self._reshape_and_mask_variables(data_in, out=False)
            else:
                upper_air_t, surface_t, varying_boundary_t = self._reshape_and_mask_variables(data_in, out=False)
                diagnostic_t = None
        else:
            upper_air_t, surface_t = self._reshape_and_mask_variables(data_in, out=False)

        # Load boundary forcing for all lead times
        max_lead_time = lead_times[-1]
        start_time_tensor = torch.tensor([start_time.year, start_time.month, start_time.day, start_time.hour])

        varying_boundary_data = [varying_boundary_t]
        for step in range(max_lead_time):
            bnd_time = start_time + timedelta(hours=self.timedelta_hours * step)
            bnd_raw = torch.tensor(
                self._get_data(bnd_time, variable_list=self.varying_boundary_variables)
            ).to(torch.float32)
            varying_boundary_data.append(self._fill_mask(bnd_raw, self.varying_boundary_variables))
        varying_boundary_data = torch.stack(
            [self.boundary_transform(b) for b in varying_boundary_data], dim=0
        )

        if self.validate:
            return self._getitem_validate(
                start_time, max_lead_time, surface_t, upper_air_t, diagnostic_t,
                varying_boundary_data, start_time_tensor, has_diagnostic,
            )

        # Inference only — return input + boundary
        surface_t = self.surface_transform(surface_t)
        upper_air_t = self.upper_air_transform(upper_air_t)
        if self.diagnostic_input:
            diagnostic_t = self.diagnostic_transform(diagnostic_t)

        self._check_nans(surface_t=surface_t, upper_air_t=upper_air_t,
                         varying_boundary_data=varying_boundary_data)

        if self.diagnostic_input:
            return surface_t, upper_air_t, diagnostic_t, varying_boundary_data

        return surface_t, upper_air_t, varying_boundary_data

    def _getitem_validate(self, start_time, max_lead_time, surface_t, upper_air_t, diagnostic_t,
                          varying_boundary_data, start_time_tensor, has_diagnostic):
        """Load multi-step targets for validation scoring."""
        targets_surface = []
        targets_upper_air = []
        targets_diagnostic = [] if has_diagnostic else None
        targets_delta_surface = [] if self.params['predict_delta'] else None
        targets_delta_upper_air = [] if self.params['predict_delta'] else None

        for step in range(1, max_lead_time + 1):
            target_time = start_time + timedelta(hours=self.timedelta_hours * step)
            raw_target = self._get_data(target_time, out=True)

            if has_diagnostic:
                ua_target, sfc_target, diag_target = self._reshape_and_mask_variables(raw_target, out=True)
                targets_diagnostic.append(diag_target)
            else:
                ua_target, sfc_target = self._reshape_and_mask_variables(raw_target, out=True)

            targets_surface.append(sfc_target)
            targets_upper_air.append(ua_target)

            if self.params['predict_delta']:
                if step == 1:
                    sfc_delta = targets_surface[-1] - surface_t
                    ua_delta = targets_upper_air[-1] - upper_air_t
                else:
                    sfc_delta = targets_surface[-1] - targets_surface[-2]
                    ua_delta = targets_upper_air[-1] - targets_upper_air[-2]
                targets_delta_surface.append(self.surface_delta_transform(sfc_delta))
                targets_delta_upper_air.append(self.upper_air_delta_transform(ua_delta))

        # Normalize all targets
        targets_surface = torch.stack([self.surface_transform(s) for s in targets_surface], dim=0)
        targets_upper_air = torch.stack([self.upper_air_transform(u) for u in targets_upper_air], dim=0)
        if has_diagnostic:
            targets_diagnostic = torch.stack([self.diagnostic_transform(d) for d in targets_diagnostic], dim=0)

        surface_t = self.surface_transform(surface_t)
        upper_air_t = self.upper_air_transform(upper_air_t)
        diagnostic_t = self.diagnostic_transform(diagnostic_t) if self.diagnostic_input else None

        self._check_nans(surface_t=surface_t, upper_air_t=upper_air_t,
                         varying_boundary_data=varying_boundary_data)

        # Build return tuple
        if diagnostic_t is not None:
            result = [surface_t, upper_air_t, diagnostic_t, targets_surface, targets_upper_air]
        else:
            result = [surface_t, upper_air_t, targets_surface, targets_upper_air]
        if has_diagnostic:
            result.append(targets_diagnostic)
        if self.params['predict_delta']:
            targets_delta_surface = torch.stack(targets_delta_surface, dim=0)
            targets_delta_upper_air = torch.stack(targets_delta_upper_air, dim=0)
            result.extend([targets_delta_surface, targets_delta_upper_air])
        result.extend([varying_boundary_data, start_time_tensor])
        return tuple(result)

    def _getitem_single_step(self, index, has_boundary):
        """Single-step evaluation without lead times."""
        start_time = self.start_date + timedelta(hours=self.dates[index])
        data_in = self._get_data(start_time, out=False)

        if has_boundary:
            if self.diagnostic_input:
                upper_air_t, surface_t, diagnostic_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out=False)
                diagnostic_t = self.diagnostic_transform(diagnostic_t)
            else:
                upper_air_t, surface_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out=False)
            varying_boundary_data = self.boundary_transform(varying_boundary_data).unsqueeze(0)
        else:
            upper_air_t, surface_t = self._reshape_and_mask_variables(data_in, out=False)

        surface_t = self.surface_transform(surface_t)
        upper_air_t = self.upper_air_transform(upper_air_t)

        self._check_nans(surface_t=surface_t, upper_air_t=upper_air_t,
                         varying_boundary_data=varying_boundary_data if has_boundary else None)

        if self.diagnostic_input:
            return surface_t, upper_air_t, diagnostic_t, varying_boundary_data
        return surface_t, upper_air_t, upper_air_t, varying_boundary_data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_input_noise(self, data):
        """Add scaled Gaussian noise to input for regularization.

        Parameters
        ----------
        data : torch.Tensor
        field_type : str
            ``'surface'`` or ``'upper_air'``.
        """
        scale = self.epsilon_factor
        return data + torch.randn_like(data) * scale

    @staticmethod
    def _check_nans(**tensors):
        """Raise ValueError if any provided tensor contains NaN."""
        for name, tensor in tensors.items():
            if tensor is not None and torch.any(torch.isnan(tensor)):
                raise ValueError(f'{name} contains NaN values.')
