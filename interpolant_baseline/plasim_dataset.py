"""PlaSim per-timestep H5 dataset for the diffusion baseline.

Each .h5 file stores one 6-hourly snapshot under group ``input`` on a 64x128
equiangular grid (see /glade/campaign/univ/uchi0018/weidong/PLASIM/sim52/).
File names are ``{year}_{step:04d}.h5`` where ``step = day*4 + sub_step`` with
day in [0, 365) and sub_step in [0, 4); PlaSim simulates a 365-day year.

The dataset returns tensors shaped for the 2D UNet in
DiffusionDate_noDropout_latlon.py: state is a single [C, H, W] tensor where
the multi-level vars are flattened over (var, level).
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


LEVS = [
    100000.0, 92500.0, 85000.0, 70000.0, 60000.0, 50000.0,
    40000.0, 30000.0, 25000.0, 20000.0, 15000.0, 10000.0, 5000.0,
]

# Variables predicted by the diffusion model (state channels)
UPPER_AIR_VARS = ['ta', 'ua', 'va', 'hus', 'zg']
SURFACE_VARS = ['tas', 'pl', 'ts', 'mrso']

# Conditioning (forcing) channels
VARYING_BOUNDARY_VARS = ['rsdt', 'sst', 'sic']
CONSTANT_BOUNDARY_VARS = ['lsm', 'sg']

N_UPPER = len(UPPER_AIR_VARS) * len(LEVS)   # 5 * 13 = 65
N_SURFACE = len(SURFACE_VARS)               # 4
N_STATE = N_UPPER + N_SURFACE               # 69
N_VARYING_BOUND = len(VARYING_BOUNDARY_VARS)  # 3
N_CONST_BOUND = len(CONSTANT_BOUNDARY_VARS)   # 2

H, W = 64, 128
STEPS_PER_YEAR = 365 * 4  # PlaSim uses idealised 365-day calendar, 6h steps


def generate_file_names(start_year, end_year):
    return [
        f"{year}_{step:04d}"
        for year in range(start_year, end_year + 1)
        for step in range(STEPS_PER_YEAR)
    ]


def load_norm_stats(plasim_root):
    """Load normalize_mean.npz / normalize_std.npz into plain dicts.

    Returned values are numpy arrays (per-pixel for upper-air keys like
    ``ta_50000.0``; scalar-broadcasting works because of trailing ``[:]``
    in the apply step).
    """
    with np.load(os.path.join(plasim_root, 'normalize_mean.npz')) as f:
        normalize_mean = {k: f[k] for k in f.files}
    with np.load(os.path.join(plasim_root, 'normalize_std.npz')) as f:
        normalize_std = {k: f[k] for k in f.files}
    return normalize_mean, normalize_std


def load_constants(plev_data_dir, normalize_mean=None, normalize_std=None):
    """Read lsm + standardised sg from any one h5 (they are time-invariant).

    Returns a [N_CONST_BOUND, H, W] float32 tensor.
    """
    sample = sorted(os.listdir(plev_data_dir))[0]
    with h5py.File(os.path.join(plev_data_dir, sample), 'r') as f:
        lsm = np.nan_to_num(f['input/lsm'][:])
        sg = f['input/sg'][:]
    sg = (sg - sg.mean()) / sg.std()
    return torch.from_numpy(np.stack([lsm, sg]).astype(np.float32))


class PlaSimDiffusionDataset(Dataset):
    """Pairs (state_t, state_{t+interval}, boundary_t, scalar_params).

    Args:
        date_list: list of ``"{year}_{step:04d}"`` strings.
        data_root_path: directory containing the per-timestep h5 files.
        normalize_mean / normalize_std: dicts from :func:`load_norm_stats`.
        interval: number of 6h steps between input and target (4 = 1 day).
    """

    def __init__(self,
                 date_list,
                 data_root_path,
                 normalize_mean,
                 normalize_std,
                 interval=4):
        self.date_list = date_list
        self.data_root_path = data_root_path
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.interval = interval

    def __len__(self):
        return len(self.date_list) - self.interval

    def _read_state(self, f):
        """Read normalised state from an already-open h5 handle."""
        surface = []
        for var in SURFACE_VARS:
            data = f[f'input/{var}'][:]
            data = (data - self.normalize_mean[var][:]) / self.normalize_std[var][:]
            surface.append(np.nan_to_num(data))
        surface = np.stack(surface)  # [N_SURFACE, H, W]

        upper = []
        for var in UPPER_AIR_VARS:
            for lev in LEVS:
                key = f'{var}_{lev}'
                data = f[f'input/{key}'][:]
                data = (data - self.normalize_mean[key][:]) / self.normalize_std[key][:]
                upper.append(data)
        upper = np.stack(upper)  # [N_UPPER, H, W]
        return np.concatenate([surface, upper], axis=0).astype(np.float32)

    def _read_boundary(self, f):
        """Read normalised varying boundary from an already-open h5 handle."""
        chans = []
        for var in VARYING_BOUNDARY_VARS:
            data = f[f'input/{var}'][:]
            data = (data - self.normalize_mean[var][:]) / self.normalize_std[var][:]
            chans.append(np.nan_to_num(data))  # sst/sic NaN over land/ocean
        return np.stack(chans).astype(np.float32)

    @staticmethod
    def _scalar_params(date_str):
        """Length-2 [day_of_year_frac, hour_of_day_frac] in [0, 1).

        Both feed ClimaDiT's TimestepEmbedder (Fourier-based), which expects
        scalar cond inputs roughly in O(1) — fractions are fine.
        """
        _, step = date_str.split('_')
        step = int(step)
        day = step // 4
        sub = step % 4
        return np.array([day / 365.0, sub / 4.0], dtype=np.float32)

    def __getitem__(self, idx):
        d_in = self.date_list[idx]
        d_out = self.date_list[idx + self.interval]

        # Open d_in once: state_in and boundary_in share the same file. Then
        # open d_out once for state_out. This cuts h5 opens per sample from
        # 3 (state_in + state_out + boundary_in) to 2.
        path_in = os.path.join(self.data_root_path, f"{d_in}.h5")
        with h5py.File(path_in, 'r') as f_in:
            state_in_np = self._read_state(f_in)
            boundary_in_np = self._read_boundary(f_in)

        path_out = os.path.join(self.data_root_path, f"{d_out}.h5")
        with h5py.File(path_out, 'r') as f_out:
            state_out_np = self._read_state(f_out)

        state_in = torch.from_numpy(state_in_np)
        state_out = torch.from_numpy(state_out_np)
        boundary_in = torch.from_numpy(boundary_in_np)
        scalar_params = torch.from_numpy(self._scalar_params(d_in))
        return state_in, state_out, boundary_in, scalar_params
