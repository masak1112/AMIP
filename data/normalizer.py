import torch
import numpy as np
from einops import rearrange

class Normalizer:
    def __init__(self, stat_path):
        
        self.stat_dict = np.load(stat_path, allow_pickle=True).item()

        # load stats
        self.surface_means = torch.tensor(self.stat_dict['surface']['mean'], dtype=torch.float32)  # shape (surface_channels,)
        self.surface_stds = torch.tensor(self.stat_dict['surface']['std'], dtype=torch.float32)    # shape (surface_channels,)
        
        self.multilevel_means = torch.tensor(self.stat_dict['multilevel']['mean'], dtype=torch.float32)  # shape (nlevels, multi_level_channels)
        self.multilevel_stds = torch.tensor(self.stat_dict['multilevel']['std'], dtype=torch.float32)    # shape (nlevels, multi_level_channels)

        self.forcing_means = torch.tensor(self.stat_dict['forcing']['mean'], dtype=torch.float32)  # shape (forcing_channels,)
        self.forcing_stds = torch.tensor(self.stat_dict['forcing']['std'], dtype=torch.float32)    # shape (forcing_channels,)

        self.invariant_means = torch.tensor(self.stat_dict['invariant']['mean'], dtype=torch.float32)  # shape (invariant_channels,)
        self.invariant_stds = torch.tensor(self.stat_dict['invariant']['std'], dtype=torch.float32)    # shape (invariant_channels,)

        self.diagnostic_means = torch.tensor(self.stat_dict['diagnostic']['mean'], dtype=torch.float32)  # shape (diagnostic_channels,)
        self.diagnostic_stds = torch.tensor(self.stat_dict['diagnostic']['std'], dtype=torch.float32)    # shape (diagnostic_channels,)

        # reshape stats
        self.surface_means = rearrange(self.surface_means, 'c -> 1 1 1 c') # nt nlat nlon c
        self.surface_stds = rearrange(self.surface_stds, 'c -> 1 1 1 c') 

        self.multilevel_means = rearrange(self.multilevel_means, 'n c -> 1 1 1 n c')
        self.multilevel_stds = rearrange(self.multilevel_stds, 'n c -> 1 1 1 n c')

        self.forcing_means = rearrange(self.forcing_means, 'c -> 1 1 1 c')
        self.forcing_stds = rearrange(self.forcing_stds, 'c -> 1 1 1 c')

        self.invariant_means = rearrange(self.invariant_means, 'c -> 1 1 c')
        self.invariant_stds = rearrange(self.invariant_stds, 'c -> 1 1 c')

        self.diagnostic_means = rearrange(self.diagnostic_means, 'c -> 1 1 1 c')
        self.diagnostic_stds = rearrange(self.diagnostic_stds, 'c -> 1 1 1 c')

    def normalize_surface(self, x):
        # x in shape (nt, nlat, nlon, surface_channels) or (b, nt, nlat, nlon, surface_channels)
        if len(x.shape) == 5:
            x = (x - self.surface_means.unsqueeze(0).to(x.device)) / self.surface_stds.unsqueeze(0).to(x.device)
        else:
            x = (x - self.surface_means.to(x.device)) / self.surface_stds.to(x.device)
        return x
    
    def normalize_multilevel(self, x):
        # x in shape (nt, nlat, nlon, nlevels, multi_level_channels) or (b, nt, nlat, nlon, nlevels, multi_level_channels)
        if len(x.shape) == 6:
            x = (x - self.multilevel_means.unsqueeze(0).to(x.device)) / self.multilevel_stds.unsqueeze(0).to(x.device)
        else:
            x = (x - self.multilevel_means.to(x.device)) / self.multilevel_stds.to(x.device)
        return x
    
    def normalize_forcing(self, x):
        # x in shape (nt, nlat, nlon, forcing_channels) or (b, nt, nlat, nlon, forcing_channels)
        if len(x.shape) == 5:
            x = (x - self.forcing_means.unsqueeze(0).to(x.device)) / self.forcing_stds.unsqueeze(0).to(x.device)
        else:
            x = (x - self.forcing_means.to(x.device)) / self.forcing_stds.to(x.device)
        return x
    
    def normalize_invariant(self, x):
        # x in shape (nlat, nlon, invariant_channels) or (b, nlat, nlon, invariant_channels)
        if len(x.shape) == 4:
            x = (x - self.invariant_means.unsqueeze(0).to(x.device)) / self.invariant_stds.unsqueeze(0).to(x.device)
        else:
            x = (x - self.invariant_means.to(x.device)) / self.invariant_stds.to(x.device)
        return x
    
    def normalize_diagnostic(self, x):
        # x in shape (nt, nlat, nlon, diagnostic_channels) or (b, nt, nlat, nlon, diagnostic_channels)
        if len(x.shape) == 5:
            x = (x - self.diagnostic_means.unsqueeze(0).to(x.device)) / self.diagnostic_stds.unsqueeze(0).to(x.device)
        else:
            x = (x - self.diagnostic_means.to(x.device)) / self.diagnostic_stds.to(x.device)
        return x
    
    def denormalize_surface(self, x):
        # x in shape (nt, nlat, nlon, surface_channels) or (b, nt, nlat, nlon, surface_channels)
        if len(x.shape) == 5:
            x = x * self.surface_stds.unsqueeze(0).to(x.device) + self.surface_means.unsqueeze(0).to(x.device)
        else:
            x = x * self.surface_stds.to(x.device) + self.surface_means.to(x.device)
        return x
    
    def denormalize_multilevel(self, x):
        # x in shape (nt, nlat, nlon, nlevels, multi_level_channels) or (b, nt, nlat, nlon, nlevels, multi_level_channels)
        if len(x.shape) == 6:
            x = x * self.multilevel_stds.unsqueeze(0).to(x.device) + self.multilevel_means.unsqueeze(0).to(x.device)
        else:
            x = x * self.multilevel_stds.to(x.device) + self.multilevel_means.to(x.device)
        return x

    def denormalize_forcing(self, x):
        # x in shape (nt, nlat, nlon, forcing_channels) or (b, nt, nlat, nlon, forcing_channels)
        if len(x.shape) == 5:
            x = x * self.forcing_stds.unsqueeze(0).to(x.device) + self.forcing_means.unsqueeze(0).to(x.device)
        else:
            x = x * self.forcing_stds.to(x.device) + self.forcing_means.to(x.device)
        return x

    def denormalize_invariant(self, x):
        # x in shape (nlat, nlon, invariant_channels) or (b, nlat, nlon, invariant_channels)
        if len(x.shape) == 4:
            x = x * self.invariant_stds.unsqueeze(0).to(x.device) + self.invariant_means.unsqueeze(0).to(x.device)
        else:
            x = x * self.invariant_stds.to(x.device) + self.invariant_means.to(x.device)
        return x
    
    def denormalize_diagnostic(self, x):
        # x in shape (nt, nlat, nlon, diagnostic_channels) or (b, nt, nlat, nlon, diagnostic_channels)
        if len(x.shape) == 5:
            x = x * self.diagnostic_stds.unsqueeze(0).to(x.device) + self.diagnostic_means.unsqueeze(0).to(x.device)
        else:
            x = x * self.diagnostic_stds.to(x.device) + self.diagnostic_means.to(x.device)
        return x
