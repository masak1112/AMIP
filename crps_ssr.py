# Default imports
import argparse
import torch
import os 

# Custom imports
from common.utils import get_yaml, save_yaml, assemble_forcing, assemble_input
from modules.train_module import TrainModule
from data.datamodule import ClimateDataModule
from tqdm import tqdm
import pickle 

# Lightning imports
import lightning as L
from lightning.pytorch import seed_everything

import xarray as xr 
import weatherbench2.metrics as wb_metrics
import matplotlib.pyplot as plt 
import numpy as np

ensemble_mean = wb_metrics.EnsembleMeanMSE()
ensemble_variance = wb_metrics.EnsembleVariance()
crps = wb_metrics.CRPS()

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

def plot_result_climate(y_pred, y, filename, num_t=6, cmap='twilight_shifted'):
    # y in shape [t h w], y_pred in shape [t h w]

    t_total, h, w = y_pred.shape

    dt = 0
    if num_t != 1:
        dt = t_total // num_t
        y_pred = y_pred[::dt]
        y = y[::dt]

    fig, axs = plt.subplots(2, num_t, figsize=(num_t*6, 6))

    vmin = y.min()
    vmax = y.max()

    for i in range(num_t):
        if num_t == 1:
            im0 = axs[0].imshow(y[i], vmin=vmin, vmax=vmax,cmap=cmap)
            im1 = axs[1].imshow(y_pred[i], vmin=vmin, vmax=vmax, cmap=cmap)

            # set the title
            axs[0].set_title(f"True t={(i+1)*dt}")
            axs[1].set_title(f"Pred t={(i+1)*dt}")
        else:
            im0 = axs[0][i].imshow(y[i], vmin=vmin, vmax=vmax,cmap=cmap)
            im1 = axs[1][i].imshow(y_pred[i], vmin=vmin, vmax=vmax, cmap=cmap)

            # set the title
            axs[0][i].set_title(f"True t={(i+1)*dt}")
            axs[1][i].set_title(f"Pred t={(i+1)*dt}")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im0, cax=cbar_ax)
    # save the figure
    plt.savefig(filename, dpi=300)
    plt.close()

def get_latitude(longitude_resolution=128, latitude_resolution=64):
    lat_end = (latitude_resolution-1)*(360/longitude_resolution) / 2
    latitude = np.linspace(-lat_end, lat_end, latitude_resolution)
    return latitude

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

def get_ssr_chunk(pred, target, latitude):

    forecast = xr.Dataset(
        {
            "var": (["realization", "time", "latitude", "longitude"], pred),
        },
        coords={"latitude": latitude,}
    )

    truth = xr.Dataset(
        {
            "var": (["time", "latitude", "longitude"], target),
        },
        coords={"latitude": latitude,}
    )

    mse = ensemble_mean.compute_chunk(forecast, truth)
    variance = ensemble_variance.compute_chunk(forecast, truth)

    mse_value = mse["var"].values
    variance_value = variance["var"].values

    ssr = np.sqrt(variance_value)/np.sqrt(mse_value)

    return ssr 

def get_crps_chunk(pred, target, latitude):

    forecast = xr.Dataset(
        {
            "var": (["realization", "time", "latitude", "longitude"], pred),
        },
        coords={"latitude": latitude,}
    )

    truth = xr.Dataset(
        {
            "var": (["time", "latitude", "longitude"], target),
        },
        coords={"latitude": latitude,}
    )
    out = crps.compute_chunk(forecast, truth)
    return out["var"]

def main(args, model_path, save_path, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)
    torch.set_float32_matmul_precision('high') # to use tensor cores if available
    seed = config["training"]["seed"]
    seed_everything(seed)

    checkpoint_path = model_path
    log_dir = save_path
    config['data']['batch_size'] = 1
    config["data"]["val_num_inferences"]= 36
    config["data"]["val_num_inferences"]= 36
    config['data']["forecast_lead_times"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    ensemble_size = 51
    plot_interval = 2
    num_t = 6

    latitude = get_latitude(longitude_resolution=128, latitude_resolution=64)

    os.makedirs(log_dir, exist_ok=True) 

    datamodule = ClimateDataModule(config["data"])

    model = TrainModule(config=config,
                        normalizer=datamodule.train_dataset)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    loader = datamodule.val_dataloader()
    model.to(device)
    
    idx_dict = {'2m_temperature': -1,
                'temperature': -6,
                'geopotential': -10,
                'u_component_of_wind': -13,
                'PRATEsfc_24h': -1,
                'specific_total_water': -6}


    crps_dict = {"2m_temperature": [], "temperature": [], "geopotential": [], "u_component_of_wind": [], "PRATEsfc_24h": [], 'specific_total_water': []}
    ssr_dict = {"2m_temperature": [], "temperature": [], "geopotential": [], "u_component_of_wind": [], "PRATEsfc_24h": [], 'specific_total_water': []}

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader)):

            batch = [item.to(device) for item in batch]
            
            pred_feat_dict, target_feat_dict = model.validation_step(batch, batch_idx=idx, eval=True, ensemble_size=ensemble_size, return_ens=True)
            
            for key in idx_dict.keys():
                # shape of crps, ssr is (t,)
                crps_channel = get_crps_chunk(pred_feat_dict[key][0], target_feat_dict[key][0], latitude=latitude)
                ssr_channel = get_ssr_chunk(pred_feat_dict[key][0], target_feat_dict[key][0], latitude=latitude)
                crps_dict[key].append(crps_channel)
                ssr_dict[key].append(ssr_channel)

                with open(os.path.join(log_dir, f"crps_{key}_{idx}.pkl"), "wb") as f:
                    pickle.dump(crps_channel, f)
                with open(os.path.join(log_dir, f"ssr_{key}_{idx}.pkl"), "wb") as f:
                    pickle.dump(ssr_channel, f)

                if (idx+1) % plot_interval == 0:
                    plot_result_climate(pred_feat_dict[key][0].cpu().numpy(),
                                        target_feat_dict[key][0].cpu().numpy(),
                                        os.path.join(log_dir, f"{key}_{idx}.png"),
                                        num_t=num_t,
                                        cmap='twilight_shifted')
                    
                    plot_crps(crps_channel,
                            title=key,
                            save_path= os.path.join(log_dir, f"crps_{key}_{idx}.png"))
                    plot_ssr(ssr_channel,
                            title=key,
                            save_path= os.path.join(log_dir, f"ssr_{key}_{idx}.png"))
        
    # save the crps_dict and ssr_dict
    for key in crps_dict.keys():
        crps_dict[key] = np.array(crps_dict[key])
        ssr_dict[key] = np.array(ssr_dict[key])

    with open(os.path.join(log_dir, "crps_dict.pkl"), "wb") as f:
        pickle.dump(crps_dict, f)
    with open(os.path.join(log_dir, "ssr_dict.pkl"), "wb") as f:
        pickle.dump(ssr_dict, f)
    
    # plot the crps_dict and ssr_dict
    for key in crps_dict.keys():
        time_averaged_crps = crps_dict[key].mean(axis=0)
        time_averaged_ssr = ssr_dict[key].mean(axis=0)
        plot_crps(time_averaged_crps,
                  title=key,
                  save_path=os.path.join(log_dir, f"crps_{key}_time_averaged.png"))
        plot_ssr(time_averaged_ssr,
                    title=key,
                    save_path=os.path.join(log_dir, f"ssr_{key}_time_averaged.png"))


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

    model_path = "/glade/derecho/scratch/ayz/AMIP_logs/SI_Latent_DiT__42_2026-03-17T15-05-45/last.ckpt"
    save_path = "/glade/derecho/scratch/ayz/AMIP_logs/SI_Latent_DiT__42_2026-03-17T15-05-45/CRPS_SSR"

    main(args, model_path, save_path)
