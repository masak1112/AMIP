import argparse
import os

import matplotlib.pyplot as plt
import torch

from common.utils import assemble_forcing, disassemble_input, get_yaml
from data.amip_new import GetDataset
from modules.train_module import TrainModule


PLOT_KEYS = [
    "2m_temperature",
    "geopotential",
    "PRATEsfc_24h",
    "u_component_of_wind",
    "temperature",
    "specific_total_water",
]
VAR_LEVEL_MAP = {
    "geopotential": -10,
    "u_component_of_wind": -13,
    "temperature": -6,
    "specific_total_water": -6,
}
T_PLOT = [0, 2, 4, 9]         # indices into the 10-day rollout
T_LABELS = [1, 3, 5, 10]      # day labels for those indices


def init_feat_dict(b, nlat, nlon, device):
    return {k: torch.zeros((b, len(T_PLOT), nlat, nlon), device=device) for k in PLOT_KEYS}


@torch.no_grad()
def rollout(model, batch, device, num_steps, inference_sampler, inference_rho, state_mode="y"):
    """Run a full 10-day autoregressive rollout and capture y / y_last / target at plot days.

    state_mode: "y" advances the state with the last Euler step (standard); "y_last"
    advances with the last model x-prediction.
    """
    assert state_mode in ("y", "y_last")
    model.scheduler.num_steps = num_steps
    #model.scheduler.inference_sampler = inference_sampler # timestep scheduler
    model.scheduler.integrator = inference_sampler
    model.scheduler.inference_rho = inference_rho

    (
        surface_t, upper_air_t, diagnostic_t,
        targets_surface, targets_upper_air, targets_diagnostic,
        varying_boundary_data, _start_time_tensor,
    ) = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

    b = surface_t.shape[0]
    nt = targets_surface.shape[1]

    nlat = model.nlat
    nlon = model.nlon
    if model.downsample is not None:
        nlat = nlat // model.downsample.downsample_factor
        nlon = nlon // model.downsample.downsample_factor

    invariant = model.invariant_input.expand(b, -1, -1, -1).to(device)

    x = model.preprocess(surface_t, upper_air_t, diagnostic_t)

    y_dict = init_feat_dict(b, nlat, nlon, device)
    y_last_dict = init_feat_dict(b, nlat, nlon, device)
    target_dict = init_feat_dict(b, nlat, nlon, device)

    i_plot = 0
    for t in range(nt):
        c_grid = assemble_forcing(varying_boundary_data[:, t], invariant)
        y, y_last = model.forward(x, c_grid, return_model_last=True)

        if t in T_PLOT:
            y_sfc, y_ml, y_diag = disassemble_input(y, nlevels=model.nlevels)
            yl_sfc, yl_ml, yl_diag = disassemble_input(y_last, nlevels=model.nlevels)

            y_sfc = model.n.surface_inv_transform(y_sfc)
            y_ml = model.n.upper_air_inv_transform(y_ml)
            y_diag = model.n.diagnostic_inv_transform(y_diag)
            yl_sfc = model.n.surface_inv_transform(yl_sfc)
            yl_ml = model.n.upper_air_inv_transform(yl_ml)
            yl_diag = model.n.diagnostic_inv_transform(yl_diag)

            sfc_tgt = targets_surface[:, t]
            ml_tgt = targets_upper_air[:, t]
            diag_tgt = targets_diagnostic[:, t]
            if model.downsample is not None:
                sfc_tgt, ml_tgt, diag_tgt = model.downsample(sfc_tgt, ml_tgt, diag_tgt)
            sfc_tgt = model.n.surface_inv_transform(sfc_tgt)
            ml_tgt = model.n.upper_air_inv_transform(ml_tgt)
            diag_tgt = model.n.diagnostic_inv_transform(diag_tgt)

            for c, name in enumerate(model.surface_variables):
                if name in PLOT_KEYS:
                    y_dict[name][:, i_plot] = y_sfc[:, c]
                    y_last_dict[name][:, i_plot] = yl_sfc[:, c]
                    target_dict[name][:, i_plot] = sfc_tgt[:, c]
            for c, name in enumerate(model.multilevel_variables):
                if name in PLOT_KEYS:
                    l = VAR_LEVEL_MAP[name]
                    y_dict[name][:, i_plot] = y_ml[:, c, l]
                    y_last_dict[name][:, i_plot] = yl_ml[:, c, l]
                    target_dict[name][:, i_plot] = ml_tgt[:, c, l]
            for c, name in enumerate(model.diagnostic_variables):
                if name in PLOT_KEYS:
                    y_dict[name][:, i_plot] = y_diag[:, c]
                    y_last_dict[name][:, i_plot] = yl_diag[:, c]
                    target_dict[name][:, i_plot] = diag_tgt[:, c]

            i_plot += 1

        x = y if state_mode == "y" else y_last

    return y_dict, y_last_dict, target_dict


def plot_rollout(y, y_last, target, var_name, out_path, cmap="twilight_shifted"):
    num_t = len(T_LABELS)
    fig, axs = plt.subplots(3, num_t, figsize=(num_t * 4.5, 10))

    vmin = target.min().item()
    vmax = target.max().item()

    for i, lbl in enumerate(T_LABELS):
        axs[0, i].imshow(target[i].numpy(), vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
        axs[0, i].set_title(f"Target — day {lbl}")
        axs[1, i].imshow(y_last[i].numpy(), vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
        axs[1, i].set_title(f"y_last (model) — day {lbl}")
        axs[2, i].imshow(y[i].numpy(), vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
        axs[2, i].set_title(f"y (euler) — day {lbl}")
        for r in range(3):
            axs[r, i].set_xticks([])
            axs[r, i].set_yticks([])

    fig.suptitle(var_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(args):
    config = get_yaml(args.config)
    checkpoint = args.checkpoint or config["training"]["checkpoint"]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device_index = torch.cuda.current_device() if torch.cuda.is_available() else None
    device = torch.device(f"cuda:{device_index}" if device_index is not None else "cpu")
    torch.set_float32_matmul_precision("high")

    dataconfig = config["data"]
    dataconfig["batch_size"] = 1

    dataset = GetDataset(
        dataconfig,
        year_start=dataconfig["val_year_start"],
        year_end=dataconfig["val_year_end"],
        num_inferences=1,
        train=False,
        validate=True,
    )

    sample = dataset[args.sample_idx]
    batch = tuple(s.unsqueeze(0) if isinstance(s, torch.Tensor) else s for s in sample)

    model = TrainModule(config, normalizer=dataset).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    num_steps_list = [5, 10, 20]
    inference_samplers = ["AB3"]
    state_modes = ["y"]
    inference_rho = 1.0

    for ns in num_steps_list:
        for sampler in inference_samplers:
            for state_mode in state_modes:
                tag = f"ns{ns}_{sampler}_rho{inference_rho}_state-{state_mode}"
                print(f"Running rollout: num_steps={ns}, sampler={sampler}, rho={inference_rho}, state_mode={state_mode}")

                y_dict, y_last_dict, target_dict = rollout(
                    model, batch, device, ns, sampler, inference_rho, state_mode=state_mode
                )

                subdir = os.path.join(output_dir, tag)
                os.makedirs(subdir, exist_ok=True)

                # for var in PLOT_KEYS:
                #     plot_rollout(
                #         y_dict[var][0].cpu(),
                #         y_last_dict[var][0].cpu(),
                #         target_dict[var][0].cpu(),
                #         var_name=f"{var}  [{tag}]",
                #         out_path=os.path.join(subdir, f"{var}.png"),
                #     )

                torch.save(
                    {
                        "y": {k: v.cpu() for k, v in y_dict.items()},
                        "y_last": {k: v.cpu() for k, v in y_last_dict.items()},
                        "target": {k: v.cpu() for k, v in target_dict.items()},
                        "num_steps": ns,
                        "inference_sampler": sampler,
                        "inference_rho": inference_rho,
                        "state_mode": state_mode,
                    },
                    os.path.join(subdir, "rollout.pt"),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/SI_midway.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Override the checkpoint path from config.")
    parser.add_argument("--output_dir", default="/project/pedramh/ayz/AMIP_logs/SI_X_large_42_2026-04-20T15-12-22/eval")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Which validation sample index to rollout.")
    args = parser.parse_args()
    main(args)
