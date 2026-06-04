# Default imports
import argparse
import time
import torch
import os
from collections import deque

# Custom imports
from common.utils import get_yaml, save_yaml, assemble_forcing, disassemble_input
from common.plotting import plot_reconstruction, plot_spectrum
from modules.train_module import TrainModule
from modules.combined_module import CombinedModule
from data.amip_new import GetDataset
from torch.utils.data import DataLoader, Subset

import torch_harmonics as th

# Lightning imports
import lightning as L
from lightning.pytorch import seed_everything


# ---------------------------------------------------------------------------
# Spectral monitoring (2m_temperature)
# ---------------------------------------------------------------------------
#
# Mirrors eval_lagged.py: compares the spherical-harmonic power spectrum of the
# predicted 2m_temperature field against an ERA5 reference taken from
# ``spectral_lag_days`` strided steps earlier. When the per-degree relative
# spectral RMSE exceeds a threshold, the rollout rewinds to the ERA5 state
# ``spectral_lag_days`` steps back and continues from a fresh IC. Recent
# predictions live in a ``pending`` deque so they can be discarded on rewind
# without contaminating the climatology accumulator.


class _SphericalSpectrum:
    """Caches a RealSHT and returns the total power at each spherical-harmonic
    degree l: P(l) = sum_m w_m * |a_{l,m}|^2, with w_m = 1 for m=0 and 2 for
    m>0 (real-field convention). Output is on CPU as a (lmax,) tensor.
    """

    def __init__(self, nlat: int, nlon: int, device, grid: str = "equiangular"):
        self.sht = th.RealSHT(nlat, nlon, grid=grid).float().to(device)
        m_w = torch.ones(self.sht.mmax, device=device)
        m_w[1:] = 2.0
        self.m_w = m_w  # (mmax,)

    def __call__(self, field_2d: torch.Tensor) -> torch.Tensor:
        f = field_2d.to(self.m_w.device, dtype=torch.float32).unsqueeze(0)
        coeffs = self.sht(f).squeeze(0)  # (lmax, mmax), complex
        power = (coeffs.real ** 2 + coeffs.imag ** 2) * self.m_w.unsqueeze(0)
        return power.sum(dim=-1).detach().cpu()  # (lmax,)


def normalized_spectral_rmse(pred_spec: torch.Tensor,
                             ref_spec: torch.Tensor,
                             eps: float = 1e-30) -> float:
    """Per-degree relative RMSE between two spherical-harmonic power spectra.
    Skips l=0 (the global mean) so the metric reflects drift in the
    variability spectrum rather than the global mean.
    """
    rel = (pred_spec - ref_spec) / (ref_spec + eps)
    return float(torch.sqrt(torch.mean(rel[1:] ** 2)).item())


def process_args(args, config):
    modelconfig = config['model']
    trainconfig = config['training']
    dataconfig = config['data']

    if len(args.devices) > 0:
        trainconfig["devices"] = [int(device) for device in args.devices]
    if args.seed is not None:
        trainconfig["seed"] = args.seed
    if args.wandb_mode is not None:
        trainconfig["wandb_mode"] = args.wandb_mode
    if args.model_name is not None:
        modelconfig["model_name"] = args.model_name
    if args.checkpoint is not None:
        trainconfig["checkpoint"] = args.checkpoint
        trainconfig["forecaster_checkpoint"] = args.checkpoint
    if args.description is not None:
        trainconfig["description"] = args.description

    return config, modelconfig, trainconfig, dataconfig

def plot_predictions(pred_feat_dict, target_feat_dict, log_dir, step):

    t2m_pred = pred_feat_dict['2m_temperature'][0].cpu() #b h w -> h w
    t2m_target = target_feat_dict['2m_temperature'][0].cpu()
    pr_6h_pred = pred_feat_dict['PRATEsfc_24h'][0].cpu()
    pr_6h_target = target_feat_dict['PRATEsfc_24h'][0].cpu()

    z500_pred = pred_feat_dict['geopotential'][0, -6, ...].cpu() # b l h w -> h w
    z500_target = target_feat_dict['geopotential'][0, -6, ...].cpu()
    u250_pred = pred_feat_dict['u_component_of_wind'][0, -9, ...].cpu()
    u250_target = target_feat_dict['u_component_of_wind'][0, -9, ...].cpu()
    t850_pred = pred_feat_dict['temperature'][0, -3, ...].cpu()
    t850_target = target_feat_dict['temperature'][0, -3, ...].cpu()
    q850_pred = pred_feat_dict['specific_total_water'][0, -3, ...].cpu()
    q850_target = target_feat_dict['specific_total_water'][0, -3, ...].cpu()

    plot_reconstruction(t2m_pred, # h w
                t2m_target,
                f'{log_dir}/t2m_{step}.png')
    plot_reconstruction(z500_pred,
                z500_target,
                f'{log_dir}/z500_{step}.png')
    plot_reconstruction(pr_6h_pred,
                pr_6h_target,
                f'{log_dir}/PRATEsfc_{step}.png')
    plot_reconstruction(u250_pred,
                u250_target,
                f'{log_dir}/u250_{step}.png')
    plot_reconstruction(t850_pred,
                t850_target,
                f'{log_dir}/t850_{step}.png')
    plot_reconstruction(q850_pred,
                q850_target,
                f'{log_dir}/q850_{step}.png')


    plot_spectrum(t2m_pred.unsqueeze(0),
                    t2m_target.unsqueeze(0),
                    f'{log_dir}/t2m_spectrum_{step}.png',
                    num_t=1)
    plot_spectrum(z500_pred.unsqueeze(0),
                    z500_target.unsqueeze(0),
                    f'{log_dir}/z500_spectrum_{step}.png',
                    num_t=1)
    plot_spectrum(pr_6h_pred.unsqueeze(0),
                    pr_6h_target.unsqueeze(0),
                    f'{log_dir}/PRATEsfc_spectrum_{step}.png',
                    num_t=1)
    plot_spectrum(u250_pred.unsqueeze(0),
                    u250_target.unsqueeze(0),
                    f'{log_dir}/u250_spectrum_{step}.png',
                    num_t=1)
    plot_spectrum(t850_pred.unsqueeze(0),
                    t850_target.unsqueeze(0),
                    f'{log_dir}/t850_spectrum_{step}.png',
                    num_t=1)
    plot_spectrum(q850_pred.unsqueeze(0),
                    q850_target.unsqueeze(0),
                    f'{log_dir}/q850_spectrum_{step}.png',
                    num_t=1)

def save_predictions(pred_feat_dict, target_feat_dict, log_dir, step):
    torch.save(pred_feat_dict, f'{log_dir}/predictions_{step}.pt')
    torch.save(target_feat_dict, f'{log_dir}/targets_{step}.pt')

def main(args):
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    ID = 0
    seed = trainconfig["seed"] + ID
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    is_combined = modelconfig.get("model_name", "") == "Combined"

    description = trainconfig.get("description", "")
    # Combined module bundles two checkpoints; anchor the log dir on the forecaster's.
    if is_combined:
        anchor_ckpt = trainconfig.get("checkpoint") or trainconfig["forecaster_checkpoint"]
    else:
        anchor_ckpt = trainconfig["checkpoint"]

    directory_path = os.path.dirname(anchor_ckpt)
    path = os.path.join(directory_path, f"bias_logs_{description}_{ID}/")

    os.makedirs(path, exist_ok=True)
    print(f"Logging to: {path}")
    save_yaml(config, path + "config.yml")

    dataconfig['batch_size'] = 1

    dataset = GetDataset(dataconfig,
                         year_start=1996,
                         year_end=2001)

    return_calendar = dataconfig.get('return_calendar', False)

    # Step through dataset at forecast intervals (e.g. every 4th sample for 24h steps with 6h data)
    stride = dataconfig['timedelta_hours'] // dataconfig['data_timedelta_hours']

    device_index = torch.cuda.current_device()
    device = f"cuda:{device_index}"

    if is_combined:
        # CombinedModule loads forecaster + downscaler checkpoints internally.
        model = CombinedModule(config, normalizer=dataset).to(device)
    else:
        model = TrainModule(config, normalizer=dataset).to(device)
        state_dict = torch.load(anchor_ckpt, map_location=device, weights_only=False)['state_dict']
        model.load_state_dict(state_dict)
    model.eval()

    ensemble_size = 1
    invariant = model.invariant_input.to(device) # 1 c nlat nlon
    invariant = invariant.expand(ensemble_size, -1, -1, -1) # e c nlat nlon

    # Climatology resolution matches the prediction resolution:
    # - CombinedModule outputs at full (downscaler) resolution.
    # - Forecaster-only: low-res if a downsample is configured, else full-res.
    if (not is_combined) and model.downsample is not None:
        downsample_factor = model.downsample.downsample_factor
        clim_nlat, clim_nlon = model.nlat // downsample_factor, model.nlon // downsample_factor
    else:
        clim_nlat, clim_nlon = model.nlat, model.nlon

    num_steps = len(dataset) // stride
    print(f"Processing {num_steps} timesteps (stride={stride}) with ensemble size {ensemble_size}...")

    plot_every = 30
    plot_val = trainconfig.get("plot_val", False)
    #num_steps = 500

    # Strided subset preserves date ordering for the autoregressive rollout while
    # letting a multi-worker DataLoader prefetch HDF5 reads in parallel. The model
    # forward is still sequential, but I/O + host->device copies overlap with compute.
    start = 0
    strided_indices = list(range(start, num_steps * stride, stride))
    num_workers = int(dataconfig.get("num_data_workers", 4))

    # ---- Spectral monitor setup ----
    spectral_monitor = args.spectral_monitor
    spectral_lag_days = args.spectral_lag_days
    base_threshold = args.spectral_threshold
    current_threshold = base_threshold
    spectral_max_restarts = args.spectral_max_restarts

    t2m_index = None
    sht_power = None
    if spectral_monitor:
        try:
            t2m_index = list(model.surface_variables).index("2m_temperature")
        except ValueError:
            print("  [spectral_monitor] '2m_temperature' not in surface_variables — disabling monitor.",
                  flush=True)
            spectral_monitor = False
        else:
            sht_power = _SphericalSpectrum(clim_nlat, clim_nlon, device=device)

    # Cached ERA5 reference spectra keyed by strided step index.
    ref_spectrum_cache: dict = {}

    # Adaptive threshold: same-target rewinds bump the threshold to let the
    # rollout squeeze past a sticky spot; once we have advanced lag_days past
    # the last rewind target the bump is reverted.
    last_rewind_step = None
    same_target_rewinds = 0
    MAX_THRESHOLD = 0.9
    THRESHOLD_BUMP = 0.05

    def fetch_state_at(step: int):
        """Fetch the normalized ERA5 state at a strided step. Returns tensors
        on device with a leading ensemble dim."""
        sample = dataset[strided_indices[step]]
        surface_t = sample[0].unsqueeze(0).to(device).expand(ensemble_size, -1, -1, -1)
        upper_air_t = sample[1].unsqueeze(0).to(device).expand(ensemble_size, -1, -1, -1, -1)
        diagnostic_t = sample[2].unsqueeze(0).to(device).expand(ensemble_size, -1, -1, -1)
        return surface_t, upper_air_t, diagnostic_t

    def downsampled_denorm(surface_t, upper_air_t, diagnostic_t):
        """Match prediction resolution, then denormalize."""
        if (not is_combined) and model.downsample is not None:
            surface_t, upper_air_t, diagnostic_t = model.downsample(
                surface_t, upper_air_t, diagnostic_t)
        return (
            model.n.surface_inv_transform(surface_t),
            model.n.upper_air_inv_transform(upper_air_t),
            model.n.diagnostic_inv_transform(diagnostic_t),
        )

    def reference_spectrum(ref_step: int):
        cached = ref_spectrum_cache.get(ref_step)
        if cached is not None:
            return cached
        try:
            surface_t, upper_air_t, diagnostic_t = fetch_state_at(ref_step)
        except (OSError, FileNotFoundError, IndexError):
            return None
        surface_denorm, _, _ = downsampled_denorm(surface_t, upper_air_t, diagnostic_t)
        spec = sht_power(surface_denorm[0, t2m_index])
        ref_spectrum_cache[ref_step] = spec
        return spec

    # per-member running mean accumulators: e c h w / e c l h w
    climatology_surface = torch.zeros((ensemble_size, len(model.surface_variables), clim_nlat, clim_nlon), device=device)
    climatology_multilevel = torch.zeros((ensemble_size, len(model.multilevel_variables), model.nlevels, clim_nlat, clim_nlon), device=device)
    climatology_diagnostic = torch.zeros((ensemble_size, len(model.diagnostic_variables), clim_nlat, clim_nlon), device=device)

    n_committed = 0
    n_restarts = 0

    def commit_to_clim(entry):
        """Welford-style running mean update. Tensors come from pending; the
        climatology buffers are updated in place."""
        nonlocal n_committed
        _, surf_pred, multi_pred, diag_pred = entry
        n_committed += 1
        climatology_surface.add_((surf_pred - climatology_surface) / n_committed)
        climatology_multilevel.add_((multi_pred - climatology_multilevel) / n_committed)
        climatology_diagnostic.add_((diag_pred - climatology_diagnostic) / n_committed)

    # Pending queue of the most-recent predictions, deferred from the climatology
    # so the last lag_days+1 days are reversible. Without the spectral monitor
    # capacity=0, so each prediction is committed immediately (original behavior).
    pending: deque = deque()
    pending_capacity = (spectral_lag_days + 1) if spectral_monitor else 0

    log_every = 10
    start_time = time.time()

    # Persistent rollout state carried across (potential) loader recreations.
    x = None
    surface_pred_denorm = multilevel_pred_denorm = diagnostic_pred_denorm = None

    # Outer loop owns the data loader; on a spectral rewind we break out, set
    # cur_idx = rewind_step, and recreate the loader from the new starting
    # point. Without rewinds it iterates exactly once.
    cur_idx = 0

    with torch.no_grad():
        while cur_idx < num_steps:
            sub_dataset = Subset(dataset, strided_indices[cur_idx:])
            loader = DataLoader(
                sub_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else None,
            )

            rewind_triggered = False

            for local_idx, batch in enumerate(loader):
                step_idx = cur_idx + local_idx

                # DataLoader has already added the leading batch dim (size 1).
                if return_calendar:
                    surface_t_b, upper_air_t_b, diagnostic_t_b, surface_t1_b, upper_air_t1_b, diagnostic_t1_b, varying_boundary_data_b, calendar_b = batch
                    calendar = calendar_b.to(device, non_blocking=True).expand(ensemble_size, -1)
                else:
                    surface_t_b, upper_air_t_b, diagnostic_t_b, surface_t1_b, upper_air_t1_b, diagnostic_t1_b, varying_boundary_data_b = batch
                    calendar = None

                varying_boundary_data = varying_boundary_data_b.to(device, non_blocking=True).expand(ensemble_size, -1, -1, -1)

                # Initialize x from the IC only on the very first step. After a
                # rewind, x has already been preprocessed from the rewound ERA5
                # state, so we must not overwrite it here.
                if step_idx == 0:
                    surface_t = surface_t_b.to(device, non_blocking=True).expand(ensemble_size, -1, -1, -1)
                    upper_air_t = upper_air_t_b.to(device, non_blocking=True).expand(ensemble_size, -1, -1, -1, -1)
                    diagnostic_t = diagnostic_t_b.to(device, non_blocking=True).expand(ensemble_size, -1, -1, -1)

                    x = model.preprocess(surface_t, upper_air_t, diagnostic_t) # e c h w / e c l h w / e c h w

                c_grid = assemble_forcing(varying_boundary_data, invariant) # e c h w

                # TrainModule: y and y_last both low-res.
                # CombinedModule: y is low-res rollout state, y_last is full-res downscaled prediction.
                fwd_kwargs = {'return_model_last': True}
                if calendar is not None:
                    fwd_kwargs['c_scalar'] = calendar
                y, y_last = model.forward(x, c_grid, **fwd_kwargs)

                surface_pred, multilevel_pred, diagnostic_pred = disassemble_input(y_last, nlevels=model.nlevels)

                surface_pred_denorm = model.n.surface_inv_transform(surface_pred)
                multilevel_pred_denorm = model.n.upper_air_inv_transform(multilevel_pred)
                diagnostic_pred_denorm = model.n.diagnostic_inv_transform(diagnostic_pred)

                # Buffer the prediction; commit the oldest once buffer overflows.
                pending.append((step_idx,
                                surface_pred_denorm.detach().clone(),
                                multilevel_pred_denorm.detach().clone(),
                                diagnostic_pred_denorm.detach().clone()))
                while len(pending) > pending_capacity:
                    commit_to_clim(pending.popleft())

                x = y

                # ---- Spectral monitor: trigger rewind on blowup ----
                # Fires only once pending has its full lag_days+1 entries — that
                # guarantees the rewind-target prediction is still in pending
                # and has not been committed to the climatology.
                if (spectral_monitor
                        and len(pending) >= spectral_lag_days + 1
                        and n_restarts < spectral_max_restarts):
                    rewind_step = step_idx + 1 - spectral_lag_days
                    ref_spec = reference_spectrum(rewind_step)
                    if ref_spec is not None:
                        pred_spec = sht_power(surface_pred_denorm[0, t2m_index])
                        rmse = normalized_spectral_rmse(pred_spec, ref_spec)
                        if rmse > current_threshold:
                            n_restarts += 1

                            if rewind_step == last_rewind_step:
                                same_target_rewinds += 1
                            else:
                                same_target_rewinds = 1
                                current_threshold = base_threshold
                                last_rewind_step = rewind_step

                            print(f"  [spectral_monitor] step {step_idx} "
                                  f"2m_T spectral rmse={rmse:.3f} > {current_threshold:.3f} -- "
                                  f"rewinding {spectral_lag_days} steps to step {rewind_step} "
                                  f"(restart #{n_restarts}, same-target #{same_target_rewinds}).",
                                  flush=True)

                            if same_target_rewinds > 1 and current_threshold < MAX_THRESHOLD:
                                new_threshold = min(current_threshold + THRESHOLD_BUMP, MAX_THRESHOLD)
                                print(f"  [spectral_monitor] {same_target_rewinds} rewinds at step "
                                      f"{rewind_step} -- bumping threshold "
                                      f"{current_threshold:.3f} -> {new_threshold:.3f}.", flush=True)
                                current_threshold = new_threshold

                            # Discard all pending entries (none have been committed
                            # yet) and replace with the ERA5 ground-truth snapshot
                            # at the rewind target — so the climatology has the
                            # same effective sample count and the next spectral
                            # check fires against the same target, enabling the
                            # adaptive-threshold ratchet.
                            pending.clear()
                            surface_t, upper_air_t, diagnostic_t = fetch_state_at(rewind_step)
                            x = model.preprocess(surface_t, upper_air_t, diagnostic_t)
                            gt_surf, gt_multi, gt_diag = downsampled_denorm(
                                surface_t, upper_air_t, diagnostic_t)
                            pending.append((rewind_step - 1,
                                            gt_surf.detach().clone(),
                                            gt_multi.detach().clone(),
                                            gt_diag.detach().clone()))

                            cur_idx = rewind_step
                            rewind_triggered = True
                            break

                # ---- Reset bumped threshold once we've cleared the window ----
                # Once step_idx is lag_days past the last rewind target, the
                # whole pending window holds predictions made after that date,
                # so the sticky spot is behind us.
                if (spectral_monitor
                        and last_rewind_step is not None
                        and step_idx >= last_rewind_step + spectral_lag_days):
                    if current_threshold != base_threshold:
                        print(f"  [spectral_monitor] cleared prediction window past step "
                              f"{last_rewind_step} -- resetting threshold "
                              f"{current_threshold:.3f} -> {base_threshold:.3f}.", flush=True)
                    current_threshold = base_threshold
                    last_rewind_step = None
                    same_target_rewinds = 0

                if (step_idx + 1) % log_every == 0 or step_idx == num_steps - 1:
                    elapsed = time.time() - start_time
                    steps_done = step_idx + 1
                    avg_per_step = elapsed / steps_done
                    remaining = avg_per_step * (num_steps - steps_done)
                    restart_tag = f" | restarts {n_restarts}" if spectral_monitor else ""
                    print(
                        f"Step {steps_done}/{num_steps} | "
                        f"elapsed {elapsed:.1f}s | "
                        f"remaining {remaining:.1f}s | "
                        f"{avg_per_step:.2f}s/step{restart_tag}",
                        flush=True,
                    )

                if step_idx % plot_every == 0:
                    print(f"Step {step_idx}/{num_steps}")
                    # save intermediate climatology (ensemble mean)
                    torch.save(climatology_surface.mean(dim=0).cpu(), path + f"climatology_surface_{step_idx + 1}.pt")
                    torch.save(climatology_multilevel.mean(dim=0).cpu(), path + f"climatology_multilevel_{step_idx + 1}.pt")
                    torch.save(climatology_diagnostic.mean(dim=0).cpu(), path + f"climatology_diagnostic_{step_idx + 1}.pt")

                    # Targets stay at full resolution
                    surface_t1_dev = surface_t1_b.to(device, non_blocking=True)
                    upper_air_t1_dev = upper_air_t1_b.to(device, non_blocking=True)
                    diagnostic_t1_dev = diagnostic_t1_b.to(device, non_blocking=True)

                    # Match target resolution to prediction resolution.
                    if (not is_combined) and model.downsample is not None:
                        surface_t1_dev, upper_air_t1_dev, diagnostic_t1_dev = model.downsample(surface_t1_dev, upper_air_t1_dev, diagnostic_t1_dev)

                    surface_true_denorm = model.n.surface_inv_transform(surface_t1_dev)
                    multilevel_true_denorm = model.n.upper_air_inv_transform(upper_air_t1_dev)
                    diagnostic_true_denorm = model.n.diagnostic_inv_transform(diagnostic_t1_dev)

                    # use first ensemble member for plotting
                    pred_feat_dict = {}
                    target_feat_dict = {}

                    for c, surface_feat_name in enumerate(model.surface_variables):
                        pred_feat_dict[surface_feat_name] = surface_pred_denorm[:1, c] # 1 nlat nlon
                        target_feat_dict[surface_feat_name] = surface_true_denorm[:, c]

                    for c, multilevel_feat_name in enumerate(model.multilevel_variables):
                        pred_feat_dict[multilevel_feat_name] = multilevel_pred_denorm[:1, c] # 1 nlevel nlat nlon
                        target_feat_dict[multilevel_feat_name] = multilevel_true_denorm[:, c]

                    for c, diagnostic_feat_name in enumerate(model.diagnostic_variables):
                        pred_feat_dict[diagnostic_feat_name] = diagnostic_pred_denorm[:1, c] # 1 nlat nlon
                        target_feat_dict[diagnostic_feat_name] = diagnostic_true_denorm[:, c]

                    if plot_val:
                        plot_predictions(pred_feat_dict, target_feat_dict, path, step_idx + 1)

                    save_predictions(pred_feat_dict, target_feat_dict, path, step_idx + 1)

            if not rewind_triggered:
                # Loader exhausted naturally — exit the outer loop.
                break

    # Drain any remaining pending entries into the climatology accumulator.
    while pending:
        commit_to_clim(pending.popleft())

    # save ensemble climatologies
    torch.save(climatology_surface.cpu(), path + "climatology_surface_ensemble.pt")
    torch.save(climatology_multilevel.cpu(), path + "climatology_multilevel_ensemble.pt")
    torch.save(climatology_diagnostic.cpu(), path + "climatology_diagnostic_ensemble.pt")

    # average across ensemble members
    torch.save(climatology_surface.mean(dim=0).cpu(), path + "climatology_surface.pt")
    torch.save(climatology_multilevel.mean(dim=0).cpu(), path + "climatology_multilevel.pt")
    torch.save(climatology_diagnostic.mean(dim=0).cpu(), path + "climatology_diagnostic.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--config", default=None)
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--devices', nargs='+', help='<Required> Set flag', default=[])
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--wandb_mode', default=None)
    parser.add_argument('--description', default=None)
    parser.add_argument('--checkpoint', default=None, help='Path to the checkpoint to resume training')
    parser.add_argument('--spectral_monitor', action='store_true',
                        help="Monitor 2m_temperature zonal power spectrum and rewind the rollout "
                             "when its normalized spectral RMSE vs. an ERA5 reference exceeds "
                             "--spectral_threshold.")
    parser.add_argument('--spectral_lag_days', type=int, default=10,
                        help="Look-back window (strided steps) for both the reference spectrum "
                             "and the rewind target. Default 10.")
    parser.add_argument('--spectral_threshold', type=float, default=0.5,
                        help="Per-wavenumber relative spectral RMSE above which a rewind is "
                             "triggered (l=0 excluded). Default 0.5.")
    parser.add_argument('--spectral_max_restarts', type=int, default=100,
                        help="Maximum number of spectral rewinds before monitoring is suppressed "
                             "and the rollout is allowed to continue unchecked. Default 100.")
    args = parser.parse_args()

    main(args)
