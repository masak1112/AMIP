# Default imports
import argparse
import torch
import os 

# Custom imports
from common.utils import get_yaml, save_yaml, assemble_forcing, assemble_input
from modules.train_module import TrainModule
from data.amip_new import GetDataset
from tqdm import tqdm

# Lightning imports
import lightning as L
from lightning.pytorch import seed_everything

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
    if args.description is not None:
        trainconfig["description"] = args.description
    
    return config, modelconfig, trainconfig, dataconfig

def main(args):
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    seed = trainconfig["seed"]
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    
    checkpoint = trainconfig['checkpoint']
    directory_path = os.path.dirname(checkpoint)
    path = os.path.join(directory_path, "bias_logs/")

    os.makedirs(path, exist_ok=True) 
    print(f"Logging to: {path}")
    save_yaml(config, path + "config.yml")

    dataconfig["data_timedelta_hours"] = dataconfig['timedelta_hours'] # set timedelta to 24h
    dataconfig['batch_size'] = 1

    dataset = GetDataset(dataconfig,
                         year_start=1990,
                         year_end=1997)
    device_index = torch.cuda.current_device()
    device = f"cuda:{device_index}"

    model = TrainModule(config,
                        normalizer=dataset).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=False)['state_dict']
    model.load_state_dict(state_dict)

    #ensemble_size = 8
    invariant = model.invariant_input.to(device) # 1 c nlat nlon

    print(f"Processing {len(dataset)} timesteps...")

    # get start

    all_surface_preds = torch.empty((len(dataset), len(model.surface_variables), 45, 90), device=device)
    all_multilevel_preds = torch.empty((len(dataset), len(model.multilevel_variables), model.nlevels, 45, 90), device=device)
    all_diagnostic_preds = torch.empty((len(dataset), len(model.diagnostic_variables), 45, 90), device=device)

    for batch_idx in tqdm(range(len(dataset))):
        if batch_idx == 0:
            surface_t, upper_air_t, diagnostic_t, surface_t1, upper_air_t1, diagnostic_t1, varying_boundary_data = dataset.__getitem__(batch_idx)
            surface_t = surface_t.unsqueeze(0).to(device)
            upper_air_t = upper_air_t.unsqueeze(0).to(device)
            diagnostic_t = diagnostic_t.unsqueeze(0).to(device)

            varying_boundary_data = varying_boundary_data.unsqueeze(0).to(device)

            x = assemble_input(surface_t, upper_air_t, diagnostic_t) # b c h w
            c_grid = assemble_forcing(varying_boundary_data, invariant) # b c h w

            if model.latent:
                x = model.encoder(x)
                c_grid = model.encoder(c_grid) # destroys some information in the forcing/invariants. Can use a learnable encoder?
        

        else:
            _, _, _, _, _, _, varying_boundary_data = dataset.__getitem__(batch_idx)
            c_grid = assemble_forcing(varying_boundary_data.unsqueeze(0).to(device), invariant) # b c h w

            if model.latent:
                c_grid = model.encoder(c_grid)

        surface_pred, multilevel_pred, diagnostic_pred = model.forward(x, c_grid)

        # save preds
        all_surface_preds[batch_idx] = surface_pred.squeeze(0)
        all_multilevel_preds[batch_idx] = multilevel_pred.squeeze(0)
        all_diagnostic_preds[batch_idx] = diagnostic_pred.squeeze(0)

        # update x
        x = assemble_input(surface_pred, multilevel_pred, diagnostic_pred)

    torch.save(all_surface_preds.cpu(), path + "surface_preds.pt")
    torch.save(all_multilevel_preds.cpu(), path + "multilevel_preds.pt")
    torch.save(all_diagnostic_preds.cpu(), path + "diagnostic_preds.pt")

    climatology_surface = torch.mean(all_surface_preds, dim=0)
    climatology_multilevel = torch.mean(all_multilevel_preds, dim=0)
    climatology_diagnostic = torch.mean(all_diagnostic_preds, dim=0)

    torch.save(climatology_surface.cpu(), path + "climatology_surface.pt")
    torch.save(climatology_multilevel.cpu(), path + "climatology_multilevel.pt")
    torch.save(climatology_diagnostic.cpu(), path + "climatology_diagnostic.pt")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--config", default=None)
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--devices', nargs='+', help='<Required> Set flag', default=[])
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--wandb_mode', default=None)
    parser.add_argument('--description', default=None)
    parser.add_argument('--checkpoint', default=None, help='Path to the checkpoint to resume training')
    args = parser.parse_args()

    main(args)
