import argparse
import glob
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# --- Local + interpolant_pdes imports ---
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# interpolant_pdes/ ships as a sibling of this folder under benchmarkGM/.
SI_REPO = os.path.normpath(os.path.join(HERE, os.pardir, 'interpolant_pdes'))
sys.path.insert(0, SI_REPO)

from plasim_dataset import (
    H, W, N_STATE, N_VARYING_BOUND, N_CONST_BOUND,
    PlaSimDiffusionDataset, generate_file_names,
    load_constants, load_norm_stats,
)
# From the upstream interpolant_pdes repo:
from modules.diffusion.interpolant import DriftScheduler  # noqa: E402
# Vendored, patch_size-fixed ClimaDiT (see climadit.py docstring):
from climadit import ClimaDiT  # noqa: E402


# ---- Default paths ----
PLASIM_ROOT = '/glade/campaign/univ/uchi0018/weidong/PLASIM/sim52'
PLEV_DATA_DIR = os.path.join(PLASIM_ROOT, 'h5', 'plev_data')

WORK_DIR = HERE
DEFAULT_CKPT_DIR = os.path.join(WORK_DIR, 'checkpoints')
DEFAULT_LOG_DIR = os.path.join(WORK_DIR, 'logs')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--interval', type=int, default=4,
                   help='Forecast lead in 6h steps (4 = 1 day).')
    p.add_argument('--train_start', type=int, default=7)
    p.add_argument('--train_end', type=int, default=47)
    p.add_argument('--val_year', type=int, default=101)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--min_lr', type=float, default=5e-7)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--grad_clip', type=float, default=1.0,
                   help='Global L2-norm grad clip (DiT convention).')
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--mixed_precision', type=str, default='bf16',
                   choices=['no', 'fp16', 'bf16'])

    # ---- ClimaDiT knobs ----
    p.add_argument('--dit_dim', type=int, default=768)
    p.add_argument('--dit_cond_dim', type=int, default=768)
    p.add_argument('--dit_heads', type=int, default=12)
    p.add_argument('--dit_sa_blocks', type=int, default=12)
    p.add_argument('--dit_ca_blocks', type=int, default=6)
    p.add_argument('--dit_fa_blocks', type=int, default=0)
    p.add_argument('--dit_patch_size', type=int, default=2)
    p.add_argument('--dit_l_max', type=int, default=20)
    p.add_argument('--dit_depth_dropout', type=float, default=0.02)
    p.add_argument('--dit_proj_bottleneck_dim', type=int, default=768)
    p.add_argument('--dit_kernel_expansion_ratio', type=float, default=1.0)
    p.add_argument('--dit_scale_by_sigma', action='store_true', default=True)

    # ---- SI scheduler knobs (mirror configs/climate.yaml > interpolant) ----
    p.add_argument('--si_num_train_steps', type=int, default=101)
    p.add_argument('--si_num_refinement_steps', type=int, default=5)
    p.add_argument('--si_sigma_coef', type=float, default=0.5)
    p.add_argument('--si_integrator', type=str, default='em',
                   choices=['em', 'euler'])

    p.add_argument('--ckpt_dir', type=str, default=DEFAULT_CKPT_DIR)
    p.add_argument('--ckpt_tag', type=str, default='si_climadit_plasim')
    return p.parse_args()


# ---- Model wrapper ----
class ClimaDiTSIModel(nn.Module):
    """Adapt ``ClimaDiT`` to the DriftScheduler's call contract.

    The scheduler calls
        model(cat([x, I_noised], dim=-1), t.float().view(-1, 1), **kwargs)
    with NHWC tensors. ClimaDiT.forward is
        forward(u, sigma_t, scalar_params, grid_params),
    so we just rename the positional arg and forward the kwargs through.
    """

    def __init__(self, climadit):
        super().__init__()
        self.net = climadit

    def forward(self, x, t, scalar_params=None, grid_params=None):
        return self.net(x, t, scalar_params, grid_params)


# ---- DataLoader ----
def make_loaders(args, normalize_mean, normalize_std):
    train_dates = generate_file_names(args.train_start, args.train_end)
    val_dates = generate_file_names(args.val_year, args.val_year)
    train_ds = PlaSimDiffusionDataset(
        train_dates, PLEV_DATA_DIR, normalize_mean, normalize_std,
        interval=args.interval,
    )
    val_ds = PlaSimDiffusionDataset(
        val_dates, PLEV_DATA_DIR, normalize_mean, normalize_std,
        interval=args.interval,
    )
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, train_ds, val_ds


def build_model(args):
    grid_chans = N_VARYING_BOUND + N_CONST_BOUND       # rsdt/sst/sic + lsm/sg
    in_dim = 2 * N_STATE                               # scheduler cats x + I_noised
    out_dim = N_STATE

    config = {
        'dit': {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'dim': args.dit_dim,
            'cond_dim': args.dit_cond_dim,
            'num_heads': args.dit_heads,
            'num_sa_blocks': args.dit_sa_blocks,
            'num_ca_blocks': args.dit_ca_blocks,
            'num_fa_blocks': args.dit_fa_blocks,
            'num_cond': 2,                              # day_frac, hour_frac
            'patch_size': args.dit_patch_size,
            'num_constants': grid_chans,
            'l_max': args.dit_l_max,
            'proj_bottleneck_dim': args.dit_proj_bottleneck_dim,
            'kernel_expansion_ratio': args.dit_kernel_expansion_ratio,
            'depth_dropout': args.dit_depth_dropout,
            'scale_by_sigma': args.dit_scale_by_sigma,
        },
        'data': {'nlat': H, 'nlon': W},
    }
    backbone = ClimaDiT(config)
    model = ClimaDiTSIModel(backbone)
    return model, in_dim, out_dim, grid_chans


def build_scheduler(optimizer, total_epochs, warmup_epochs, min_lr):
    """Linear warmup -> cosine anneal to min_lr, stepped per epoch."""
    warmup_epochs = max(1, warmup_epochs)
    warmup = LinearLR(
        optimizer, start_factor=1.0 / warmup_epochs, end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs),
        eta_min=min_lr,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


def model_load(model, accelerator, ckpt_dir, scheduler=None):
    """Load model weights + scheduler/meta from ``{ckpt_dir}`` if present."""
    if not os.path.isdir(ckpt_dir):
        return 0, 0, float('inf')
    sf_files = sorted(glob.glob(os.path.join(ckpt_dir, '*.safetensors')))
    if not sf_files:
        return 0, 0, float('inf')
    from safetensors.torch import load_model
    raw = accelerator.unwrap_model(model)
    device = accelerator.device
    for sf in sf_files:
        missing, unexpected = load_model(raw, sf, strict=False, device=str(device))
        if missing:
            print(f'  [load] missing keys (kept fresh): {missing[:5]} ...')
        if unexpected:
            print(f'  [load] unexpected keys (ignored): {unexpected[:5]} ...')
    meta_path = ckpt_dir + '_meta.pt'
    best_loss = float('inf')
    global_step = 0
    start_epoch = 0
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location='cpu', weights_only=False)
        best_loss = meta.get('best_loss', best_loss)
        global_step = meta.get('global_step', 0)
        start_epoch = meta.get('epoch', 0) + 1
        if scheduler is not None and 'scheduler' in meta:
            try:
                scheduler.load_state_dict(meta['scheduler'])
            except Exception as e:
                print(f'  [load] scheduler restore failed ({e}); keeping fresh schedule')
    print(f'Loaded checkpoint from {ckpt_dir} '
          f'(best_loss={best_loss:.6f}, step={global_step}, start_epoch={start_epoch})')
    return start_epoch, global_step, best_loss


def _to_nhwc(t):
    return t.permute(0, 2, 3, 1).contiguous()


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_tag)
    ckpt_path_best = os.path.join(args.ckpt_dir, f'{args.ckpt_tag}_best')

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f'Device: {device}, world_size: {accelerator.num_processes}')

    normalize_mean, normalize_std = load_norm_stats(PLASIM_ROOT)
    constants = load_constants(PLEV_DATA_DIR)  # [N_CONST_BOUND, H, W]

    train_loader, val_loader, train_ds, val_ds = make_loaders(
        args, normalize_mean, normalize_std,
    )
    if is_main:
        print(f'Train samples: {len(train_ds)}, Val samples: {len(val_ds)}')

    model, in_dim, out_dim, grid_chans = build_model(args)
    if is_main:
        n_param = sum(p.numel() for p in model.parameters())
        print(f'ClimaDiT in_dim={in_dim} out_dim={out_dim}'
              f' grid_chans={grid_chans} dim={args.dit_dim}'
              f' patch={args.dit_patch_size}'
              f' sa/ca/fa={args.dit_sa_blocks}/{args.dit_ca_blocks}/{args.dit_fa_blocks}'
              f' params={n_param/1e6:.2f}M')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    lr_sched = build_scheduler(
        optimizer, args.epochs, args.warmup_epochs, args.min_lr,
    )

    si_scheduler = DriftScheduler(
        num_refinement_steps=args.si_num_refinement_steps,
        num_train_steps=args.si_num_train_steps,
        integrator=args.si_integrator,
        sigma_coef=args.si_sigma_coef,
        ndim=2,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader,
    )
    start_epoch, global_step, best_loss = model_load(
        model, accelerator, ckpt_path, scheduler=lr_sched,
    )

    constants_dev = constants.to(device=device).unsqueeze(0)  # [1, C_const, H, W]

    for epoch in range(start_epoch, args.epochs):
        if is_main:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{args.epochs}  lr={cur_lr:.3e}')

        # --- Train ---
        model.train()
        loss_running = 0.0
        pbar = tqdm(train_loader, disable=not is_main)
        for i, (state_in, state_out, boundary_in, scalar_params) in enumerate(pbar):
            with accelerator.accumulate(model):
                state_in = state_in.to(device, non_blocking=True)
                state_out = state_out.to(device, non_blocking=True)
                boundary_in = boundary_in.to(device, non_blocking=True)
                scalar_params = scalar_params.to(device, non_blocking=True)
                B = state_in.shape[0]
                consts = constants_dev.expand(B, -1, -1, -1)

                x_nhwc = _to_nhwc(state_in)
                y_nhwc = _to_nhwc(state_out)
                grid_nhwc = _to_nhwc(torch.cat([boundary_in, consts], dim=1))

                loss = si_scheduler.compute_loss(
                    x_nhwc, y_nhwc, model,
                    scalar_params=scalar_params,
                    grid_params=grid_nhwc,
                )
                if not torch.isfinite(loss):
                    continue
                accelerator.backward(loss)
                if args.grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(),
                                                args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            loss_val = loss.detach().item()
            loss_running = (loss_running * i + loss_val) / (i + 1)
            global_step += 1
            if is_main:
                pbar.set_postfix(epoch=epoch, loss=loss_val,
                                 running=loss_running, step=global_step)

        # --- Val ---
        # SI training loss on val for cheap monitoring; full ensemble-sampling
        # diagnostics (CRPS/spread) live in val.py upstream.
        model.eval()
        val_loss_sum, val_cnt = 0.0, 0
        with torch.no_grad():
            for state_in, state_out, boundary_in, scalar_params in val_loader:
                state_in = state_in.to(device, non_blocking=True)
                state_out = state_out.to(device, non_blocking=True)
                boundary_in = boundary_in.to(device, non_blocking=True)
                scalar_params = scalar_params.to(device, non_blocking=True)
                B = state_in.shape[0]
                consts = constants_dev.expand(B, -1, -1, -1)

                x_nhwc = _to_nhwc(state_in)
                y_nhwc = _to_nhwc(state_out)
                grid_nhwc = _to_nhwc(torch.cat([boundary_in, consts], dim=1))

                vloss = si_scheduler.compute_loss(
                    x_nhwc, y_nhwc, model,
                    scalar_params=scalar_params,
                    grid_params=grid_nhwc,
                )
                val_loss_sum += vloss.item()
                val_cnt += 1

        val_loss = val_loss_sum / max(1, val_cnt)
        if is_main:
            print(f'Epoch {epoch+1} | train_running {loss_running:.6f} | val {val_loss:.6f}')

        lr_sched.step()

        # --- Save ---
        accelerator.wait_for_everyone()
        if is_main:
            meta = {
                'best_loss': best_loss,
                'global_step': global_step,
                'epoch': epoch,
                'scheduler': lr_sched.state_dict(),
            }
            accelerator.save_state(ckpt_path)
            torch.save(meta, ckpt_path + '_meta.pt')
            if loss_running < best_loss:
                best_loss = loss_running
                meta['best_loss'] = best_loss
                accelerator.save_state(ckpt_path_best)
                torch.save(meta, ckpt_path_best + '_meta.pt')
                print(f'  best running loss {best_loss:.6f} -> {ckpt_path_best}')


if __name__ == '__main__':
    main()
