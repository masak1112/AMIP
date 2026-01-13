import torch
import numpy as np
from torch.utils.data import Dataset
import h5py as h5f
import pickle 
from data.normalizer import Normalizer

SURFACE_VARIABLES = ["skin_temperature",
                     "surface_pressure",
                     "2m_temperature",
                     "2m_specific_humidity", 
                     "10m_u_component_of_wind", 
                     "10m_v_component_of_wind"]

MULTILEVEL_VARIABLES = ["temperature",
                  "u_component_of_wind",
                  "v_component_of_wind",
                  "geopotential",
                  "specific_humidity",
                  "specific_cloud_liquid_water_content",
                  "specific_cloud_ice_water_content",
                  "fraction_of_cloud_cover"]

FORCING_VARIABLES = ["DSWRFtoa", 
                     "sea_surface_temperature", # has nans
                     "sea_ice_cover"]  # has nans

INVARIANT_VARIABLES = ["geopotential_at_surface", 
                       "land_sea_mask"]

DIAGNOSTIC_VARIABLES = ["USWRFtoa",
                        "ULWRFtoa", 
                       "USWRFsfc",
                       "ULWRFsfc",
                       "DSWRFsfc",
                       "DLWRFsfc",
                       "PRATEsfc",
                       "LHTFLsfc",
                       "SHTFLsfc"]

class AMIPData(Dataset):
    def __init__(self,
                 data_path,
                 norm_stats_path,
                 split="train",
                 normalize=True,
                 nsteps=1,   # how many steps to load
                 ):

        self.data_path = data_path 
        self.nsteps = nsteps
        self.norm_stats_path = norm_stats_path
        self.normalize = normalize
        self.split = split 

        self.file = h5f.File(self.data_path, 'r') # has keys of 'split'
        self.data = self.file[split] # has keys of 'surface', 'multilevel', 'forcing', 'forcing_invariant', 'diagnostic', lat', 'lon', 'hour', 'day'
        self.n = Normalizer(norm_stats_path)

        # load refs
        self.surface = self.data['surface'] # t nlat nlon nsurface_channels
        self.multilevel = self.data['multilevel'] # t nlat nlon nlevels nmulti_channels
        self.diagnostic = self.data['diagnostic'] # t nlat nlon ndiagnostic_channels
        self.forcing = self.data['forcing'] # t nlat nlon nforcing_channels

        # directly load into memory
        self.invariants = torch.tensor(np.array(self.data['invariant'][:]), dtype=torch.float32) # nlat nlon n_invariant
        self.hour = torch.from_numpy(self.data['hour'][:]) # t
        self.day = torch.from_numpy(self.data['day'][:]) # t
        self.scalars = torch.concat([self.day.unsqueeze(-1), self.hour.unsqueeze(-1)], dim=-1) # t 2

        if self.normalize:
            self.invariants = self.n.normalize_invariant(self.invariants)

        self.horizon = len(self.hour) # t
        self.num_samples = self.horizon - nsteps + 1
        print(f"Loaded {self.horizon} snapshots for {split} split")

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        surface = torch.tensor(np.array(self.surface[idx:idx+self.nsteps]), dtype=torch.float32) # nsteps nlat nlon nsurface_channels
        multilevel = torch.tensor(np.array(self.multilevel[idx:idx+self.nsteps]), dtype=torch.float32) # nsteps nlat nlon nlevels nmulti_channels
        diagnostic = torch.tensor(np.array(self.diagnostic[idx:idx+self.nsteps]), dtype=torch.float32) # nsteps nlat nlon ndiagnostic_channels
        forcing = torch.tensor(np.array(self.forcing[idx:idx+self.nsteps]), dtype=torch.float32) # nsteps nlat nlon nforcing_channels
        scalars = self.scalars[idx:idx+self.nsteps] # nsteps 2

        if self.normalize:
            surface = self.n.normalize_surface(surface)
            multilevel = self.n.normalize_multilevel(multilevel)
            diagnostic = self.n.normalize_diagnostic(diagnostic)
            forcing = self.n.normalize_forcing(forcing)
            invariants = self.n.normalize_invariant(self.invariants)

        return_dict = {"surface": surface,
                       "multilevel": multilevel,
                       "diagnostic": diagnostic,
                       "forcing": forcing,
                       "invariants": invariants,
                       "scalars": scalars
                       }
        
        return return_dict

class ClimatologyLoader:
    def __init__(self,
                 data_path,
                 norm_stats_path,
                 climatology_path,
                 horizon=7308,
                 start_time = 32120,
                 split="train",
                 normalize=True,
                 ):

        # loads one initial frame and {horizon} timesteps of forcing data. Also returns true biases

        self.split = split 
        self.data_path = data_path 
        self.climatology_path = climatology_path
        self.norm_stats_path = norm_stats_path
        self.normalize = normalize 
        self.horizon = horizon
        self.start_time = start_time # 0 is Jan 1st, 1979. 32120 is ~Jan 1st, 2020

        self.file = h5f.File(self.data_path, 'r') 
        self.data = self.file[split] 

        self.n = Normalizer(norm_stats_path)

        # can load everything into memory
        self.surface = torch.tensor(np.array(self.data['surface'][start_time]), dtype=torch.float32).unsqueeze(0) # 1 nlat nlon nsurface_channels
        self.multilevel = torch.tensor(np.array(self.data['multilevel'][start_time]), dtype=torch.float32).unsqueeze(0) # 1 nlat nlon nlevels nmulti_channels
        self.forcing = torch.tensor(np.array(self.data['forcing'][start_time:start_time + horizon]), dtype=torch.float32).unsqueeze(0) # 1 horizon nlat nlon nforcing_channels
        self.invariants = torch.tensor(np.array(self.data['forcing_invariant'][:]), dtype=torch.float32).unsqueeze(0) # 1 nlat nlon n_invariant
        self.diagnostic = torch.tensor(np.array(self.data['diagnostic'][start_time]), dtype=torch.float32).unsqueeze(0) # 1 nlat nlon ndiagnostic_channels
        self.hour = torch.from_numpy(self.data['hour'][start_time:start_time + horizon]) # horizon
        self.day = torch.from_numpy(self.data['day'][start_time:start_time + horizon]) # horizon
        self.scalars = torch.concat([self.day.unsqueeze(-1), self.hour.unsqueeze(-1)], dim=-1).unsqueeze(0) # 1 horizon 2

        if self.normalize:
            self.invariants = self.n.normalize_invariant(self.invariants)
            self.surface = self.n.normalize_surface(self.surface)
            self.multilevel = self.n.normalize_multilevel(self.multilevel)
            self.forcing = self.n.normalize_forcing(self.forcing)
            self.diagnostic = self.n.normalize_diagnostic(self.diagnostic)

        with open(climatology_path, 'rb') as file:
            self.climatology_dict = pickle.load(file) # unnormalized

        print(f"Loaded {horizon} time stamps for {split} split")
    
    def get_data(self, device='cpu'):
        
        return_dict = {
            "surface": self.surface.to(device),
            "multilevel": self.multilevel.to(device),
            "forcing": self.forcing.to(device),
            "invariants": self.invariants.to(device),
            "diagnostic": self.diagnostic.to(device),
            "scalars": self.scalars.to(device),
            "biases": {k: v.to(device) for k, v in self.bias_dict.items()}
        }

        return return_dict


