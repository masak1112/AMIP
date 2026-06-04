"""Compute observational climatology (running mean of unnormalized GetDataset
samples) for 1996-2001, stride-4 starting at index 0.

Saves three .pt files:
  - climatology_surface.pt      shape (c, nlat, nlon)
  - climatology_diagnostic.pt   shape (c, nlat, nlon)
  - climatology_multilevel.pt   shape (c, nlevel, nlat, nlon)
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from common.utils import get_yaml
from data.amip_new import GetDataset


def main(args):
    config = get_yaml(args.config)
    dataconfig = config['data']
    dataconfig['batch_size'] = 1

    year_start = 1996
    year_end = 2001

    dataset = GetDataset(dataconfig,
                         year_start=year_start,
                         year_end=year_end)

    # Stride = forecast step / data step (e.g. 24h / 6h = 4). The user asked for
    # every 4 steps starting at index 0.
    stride = 4
    start = 0
    num_steps = len(dataset) // stride
    strided_indices = list(range(start, num_steps * stride, stride))
    strided_dataset = Subset(dataset, strided_indices)

    num_workers = int(dataconfig.get('num_data_workers', 4))
    loader = DataLoader(
        strided_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_sfc = len(dataset.surface_variables)
    n_ua = len(dataset.upper_air_variables)
    n_lev = len(dataset.levels)
    n_diag = len(dataset.diagnostic_variables)
    nlat, nlon = dataconfig['horizontal_resolution']

    clim_surface = torch.zeros((n_sfc, nlat, nlon), device=device)
    clim_multilevel = torch.zeros((n_ua, n_lev, nlat, nlon), device=device)
    clim_diagnostic = torch.zeros((n_diag, nlat, nlon), device=device)

    os.makedirs(args.output_dir, exist_ok=True)

    total = len(loader)
    print(f"Computing climatology over {total} strided samples "
          f"({year_start}-{year_end}, stride={stride})", flush=True)

    log_every = 50
    start_time = time.time()

    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            # GetDataset in training mode with diagnostic_input=True returns a
            # 7-tuple. We accumulate the current state (t), not the target (t+dt).
            surface_t_b, upper_air_t_b, diagnostic_t_b = batch[0], batch[1], batch[2]

            surface_t = surface_t_b.to(device, non_blocking=True)        # (1, c, h, w)
            upper_air_t = upper_air_t_b.to(device, non_blocking=True)    # (1, c, l, h, w)
            diagnostic_t = diagnostic_t_b.to(device, non_blocking=True)  # (1, c, h, w)

            surface_denorm = dataset.surface_inv_transform(surface_t).squeeze(0)
            multilevel_denorm = dataset.upper_air_inv_transform(upper_air_t).squeeze(0)
            diagnostic_denorm = dataset.diagnostic_inv_transform(diagnostic_t).squeeze(0)

            n = step_idx + 1
            clim_surface += (surface_denorm - clim_surface) / n
            clim_multilevel += (multilevel_denorm - clim_multilevel) / n
            clim_diagnostic += (diagnostic_denorm - clim_diagnostic) / n

            if (step_idx + 1) % log_every == 0 or step_idx == total - 1:
                elapsed = time.time() - start_time
                avg = elapsed / (step_idx + 1)
                remaining = avg * (total - step_idx - 1)
                print(
                    f"Step {step_idx + 1}/{total} | "
                    f"elapsed {elapsed:.1f}s | "
                    f"remaining {remaining:.1f}s | "
                    f"{avg:.2f}s/step",
                    flush=True,
                )

    torch.save(clim_surface.cpu(),
               os.path.join(args.output_dir, 'climatology_surface.pt'))
    torch.save(clim_diagnostic.cpu(),
               os.path.join(args.output_dir, 'climatology_diagnostic.pt'))
    torch.save(clim_multilevel.cpu(),
               os.path.join(args.output_dir, 'climatology_multilevel.pt'))

    print(f"Saved climatologies to {args.output_dir}")
    print(f"  surface:    {tuple(clim_surface.shape)}")
    print(f"  diagnostic: {tuple(clim_diagnostic.shape)}")
    print(f"  multilevel: {tuple(clim_multilevel.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute strided observational climatology.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    parser.add_argument('--output_dir', default='/glade/derecho/scratch/ayz/climatologies_1996_2001',
                        help='Directory to write .pt files.')
    args = parser.parse_args()
    main(args)
