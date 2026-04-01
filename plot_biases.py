import os 
import xarray as xr
import torch 
from common.plotting import plot_bias
from common.loss import latitude_weighted_rmse

BASE_PATH = "/glade/derecho/scratch/awikner/ERA5/AMIP/"
BIAS_LOGS = "/glade/derecho/scratch/ayz/AMIP_logs/SI_Latent_DiT__42_2026-03-17T15-05-45/bias_logs/"
downsample = True 

def plot_biases(base_path, variable_type, variable_name, variable_type2, variable_index, bias_logs, level = -1):
    if variable_type2 == "multilevel":
        path = os.path.join(base_path, variable_type, "3D_PL", variable_name + "/climo_1996_2000_180x360.nc")
    else:
        path = os.path.join(base_path, variable_type, variable_name + "/climo_1996_2000_180x360.nc")
    
    with xr.open_dataset(path) as ds:
        climatology = ds[variable_name].values
        climatology = climatology.mean(axis=0) # nlat nlon
        if level > 0:
            climatology = climatology[-1*level - 1]
    
    if downsample:
        climatology = climatology[::4, ::4]

    bias_path = os.path.join(bias_logs, f"climatology_{variable_type2}.pt")

    model_climatology = torch.load(bias_path).numpy()
    model_climatology = model_climatology[variable_index] # nlat nlon

    if level > 0:
        model_climatology = model_climatology[level]

    bias = latitude_weighted_rmse(torch.tensor(model_climatology).unsqueeze(0), torch.tensor(climatology).unsqueeze(0),
                                  nlon = climatology.shape[1], nlat = climatology.shape[0], with_time=False)
    
    print(f"Bias for {variable_name}: {bias.item()}")

    save_path = save_path = os.path.join(bias_logs, f"bias_{variable_name}.png")
    plot_bias(model_climatology, climatology, save_path)

plot_biases(BASE_PATH, "diagnostic", "PRATEsfc", "diagnostic", 8, BIAS_LOGS)
plot_biases(BASE_PATH, "prognostic", "2m_temperature", "surface", 2, BIAS_LOGS)
plot_bias(BASE_PATH, "prognostic", "geopotential", "multilevel", 3, BIAS_LOGS, -10)
plot_bias(BASE_PATH, "prognostic", "temperature", "multilevel", 0, BIAS_LOGS, -6)
plot_bias(BASE_PATH, "prognostic", "specific_total_water", "multilevel", 4, BIAS_LOGS, -6)
plot_bias(BASE_PATH, "prognostic", "u_component_of_wind", "multilevel", 2, BIAS_LOGS, -13)
