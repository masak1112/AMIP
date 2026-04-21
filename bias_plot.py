import os
import xarray as xr
import torch
from common.plotting import plot_bias
from common.loss import latitude_weighted_rmse

BASE_PATH = "/glade/derecho/scratch/awikner/ERA5/AMIP/"
BIAS_LOGS = "/glade/derecho/scratch/ayz/AMIP_logs/xInterpolant/climatologies/"
downsample = False

# Climatology levels are stored in this order (increasing pressure)
CLIMO_LEVELS = list(reversed([5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600, 700, 800, 850, 875, 900, 925, 950, 975, 1000]))

# Model output levels (decreasing pressure, i.e. increasing altitude)
MODEL_LEVELS_26 = [5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600, 700, 800, 850, 875, 900, 925, 950, 975, 1000]
MODEL_LEVELS_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

def plot_biases(base_path, variable_type, variable_name, variable_type2, variable_index, bias_logs, pressure_level=None):
    if variable_type2 == "multilevel":
        path = os.path.join(base_path, variable_type, "3D_PL", variable_name + "/climo_1996_2000_180x360.nc")
    else:
        path = os.path.join(base_path, variable_type, variable_name + "/climo_1996_2000_180x360.nc")

    with xr.open_dataset(path) as ds:
        climatology = ds[variable_name].values
        climatology = climatology.mean(axis=0) # nlat nlon
        if variable_type2 == "multilevel":
            climo_idx = CLIMO_LEVELS.index(pressure_level)
            climatology = climatology[climo_idx]

    if downsample:
        climatology = climatology[::4, ::4]

    bias_path = os.path.join(bias_logs, f"climatology_{variable_type2}.pt")

    model_climatology = torch.load(bias_path).numpy()
    model_climatology = model_climatology[variable_index] # nlat nlon

    if variable_type2 == "multilevel":
        num_levels = model_climatology.shape[0]
        model_levels = MODEL_LEVELS_13 if num_levels == 13 else MODEL_LEVELS_26
        model_idx = model_levels.index(pressure_level)
        model_climatology = model_climatology[model_idx]

    bias = latitude_weighted_rmse(torch.tensor(model_climatology).unsqueeze(0), torch.tensor(climatology).unsqueeze(0),
                                  nlon = climatology.shape[1], nlat = climatology.shape[0], with_time=False)
    
    print(f"Bias for {variable_name}: {bias.item()}")

    save_path = save_path = os.path.join(bias_logs, f"bias_{variable_name}.png")
    plot_bias(model_climatology, climatology, save_path, title = f"Bias for {variable_name} (RMSE: {bias.item():.4f})")

plot_biases(BASE_PATH, "diagnostic", "PRATEsfc", "diagnostic", 8, BIAS_LOGS)
plot_biases(BASE_PATH, "prognostic", "2m_temperature", "surface", 2, BIAS_LOGS)
plot_biases(BASE_PATH, "prognostic", "geopotential", "multilevel", 3, BIAS_LOGS, 500)
plot_biases(BASE_PATH, "prognostic", "temperature", "multilevel", 0, BIAS_LOGS, 850)
plot_biases(BASE_PATH, "prognostic", "specific_total_water", "multilevel", 4, BIAS_LOGS, 850)
plot_biases(BASE_PATH, "prognostic", "u_component_of_wind", "multilevel", 1, BIAS_LOGS, 250)
