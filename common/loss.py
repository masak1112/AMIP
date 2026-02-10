import torch
import numpy as np
import torch.nn as nn
from einops import repeat, rearrange
import math 

try:
    import torch_harmonics as th
except ImportError:
    print("Warning: torch.distributed could not be imported. Distributed losses will not work.")

# base on the code from graphcast
def _check_uniform_spacing_and_get_delta(vector):
    diff = np.diff(vector)
    if not np.all(np.isclose(diff[0], diff)):
        raise ValueError(f'Vector {diff} is not uniformly spaced.')
    return diff[0]


def _weight_for_latitude_vector_without_poles(latitude):
    """Weights for uniform latitudes of the form [+-90-+d/2, ..., -+90+-d/2]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90 - delta_latitude/2) or
        not np.isclose(np.min(latitude), -90 + delta_latitude/2)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at '
            '+- (90 - delta_latitude/2) degrees.')
    return np.cos(np.deg2rad(latitude))


def _weight_for_latitude_vector_with_poles(latitude):
    """Weights for uniform latitudes of the form [+- 90, ..., -+90]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90.) or
        not np.isclose(np.min(latitude), -90.)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
    weights = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(delta_latitude/2))
    # The two checks above enough to guarantee that latitudes are sorted, so
    # the extremes are the poles
    weights[[0, -1]] = np.sin(np.deg2rad(delta_latitude/4)) ** 2
    return weights


class WeightedLoss(nn.Module):
    def __init__(self,
                 latitude_resolution = 180,
                 longitude_resolution = 360,
                 with_poles=False,
                 latitude_weight='equal',
                 level_weight='equal',
                 multi_level_variable_weight=None,
                 surface_variable_weight=None,
                 diag_variable_weight=None,
                 nlevels=26,
                 nsurface=6,
                 nmulti=9,
                 ndiag = 9,
                 normalize = True
                 ):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='none')
        if latitude_weight == 'cosine':
            if with_poles:
                latitude = np.linspace(-90, 90, latitude_resolution)
                weights = _weight_for_latitude_vector_with_poles(latitude)
            else:
                # assume equiangular grid
                lat_end = (latitude_resolution-1)*(360/longitude_resolution) / 2
                latitude = np.linspace(-lat_end, lat_end, latitude_resolution)
                weights = _weight_for_latitude_vector_without_poles(latitude)
            weights = torch.from_numpy(weights)
            latitude_weight = weights / weights.mean()
        else:
            weights = torch.ones(latitude_resolution)   # all latitudes weight the same
            latitude_weight = weights / weights.mean() # shape (nlat, )
        self.register_buffer('latitude_weight', latitude_weight)

        if level_weight == 'linear':     # weighs the lower levels
            level_weight = torch.linspace(0.065, 0.05, nlevels)
        elif level_weight == 'exp':
            level_weight = torch.exp(torch.linspace(0, -3, nlevels))
            level_weight = level_weight / level_weight.sum()
        else:
            level_weight = torch.ones(nlevels)
            level_weight = level_weight / level_weight.sum()
        self.register_buffer('level_weight', level_weight)

        if surface_variable_weight is None:
            surface_variable_weight = torch.ones(nsurface) # default equal weight
        else:
            surface_variable_weight = torch.tensor(surface_variable_weight, dtype=torch.float32)

        if multi_level_variable_weight is None:
            multi_level_variable_weight = torch.ones(nmulti) # default equal weight
        else:
            multi_level_variable_weight = torch.tensor(multi_level_variable_weight, dtype=torch.float32)

        if diag_variable_weight is None:
            diag_variable_weight = torch.ones(ndiag)
        else:
            diag_variable_weight = torch.tensor(diag_variable_weight, dtype=torch.float32)
        
        self.register_buffer('diag_variable_weight', diag_variable_weight)
        self.register_buffer('surface_variable_weight', surface_variable_weight)
        self.register_buffer('multi_level_variable_weight', multi_level_variable_weight)

        self.normalize = normalize

    def forward(self,
                surface_pred, surface_target,
                multilevel_pred, multilevel_target,
                diagnostic_pred, diagnostic_target
                ):

        
        surface_loss = self.loss_fn(surface_pred, surface_target) # b nlat nlon nsurface
        surface_loss = surface_loss * self.surface_variable_weight.view(1, 1, 1, -1) # b nlat nlon nsurface
        surface_loss = surface_loss.sum(dim=-1) # b nlat nlon

        diag_loss = self.loss_fn(diagnostic_pred, diagnostic_target) # b nlat nlon ndiag
        diag_loss = diag_loss * self.diag_variable_weight.view(1, 1, 1, -1) # b nlat nlon ndiag
        diag_loss = diag_loss.sum(dim=-1) # b nlat nlon

        multi_level_loss = self.loss_fn(multilevel_pred, multilevel_target) # b nlevel nlat nlon nmulti
        multi_level_loss = multi_level_loss * self.level_weight.view(1, -1, 1, 1, 1) # b nlevel nlat nlon nmulti
        multi_level_loss = multi_level_loss.sum(dim=1) # b nlat nlon nmulti
        multi_level_loss = multi_level_loss * self.multi_level_variable_weight.view(1, 1, 1, -1) # b nlat nlon nmulti
        multi_level_loss = multi_level_loss.sum(dim=-1) # b nlat nlon

        if self.normalize:
            surface_loss = surface_loss / torch.norm(surface_target, p=2, keepdim=True)
            multi_level_loss = multi_level_loss / torch.norm(multilevel_target, p=2, keepdim=True)
            diag_loss = diag_loss / torch.norm(diagnostic_target, p=2, keepdim=True)

        loss = surface_loss + multi_level_loss + diag_loss # b nlat nlon
        latitude_weight = self.latitude_weight.view(1, -1, 1) # b nlat nlon 
        loss = loss * latitude_weight

        return loss.mean()   # reduce over batch/lat/lon


class LatitudeWeightedMSE(nn.Module):
    def __init__(self, nlat, nlon, loss_module=nn.MSELoss(), with_poles=False):
        super().__init__()
        self.loss_module = loss_module
        self.with_poles = with_poles
        # print(nlat, nlon)

        if not with_poles:
            longitude_resolution = nlon
            lat_end = (nlat - 1) * (360 / longitude_resolution) / 2
            lat_weight = _weight_for_latitude_vector_without_poles(np.linspace(-lat_end, lat_end, nlat))
        else:
            lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))

        lat_weight = torch.from_numpy(lat_weight)
        lat_weight = lat_weight / lat_weight.mean()
        self.register_buffer('lat_weight', lat_weight)

    def forward(self, pred, target):
        # pred, target in shape [b, nlat, nlon, c]
        lat_weight = repeat(self.lat_weight, 'nlat -> b nlat nlon', b=pred.shape[0], nlon=pred.shape[2])
        return (self.loss_module(pred, target).mean(-1) * lat_weight).mean()


def latitude_weighted_rmse(pred, 
                           target,
                           with_poles=False, 
                           nlon=None,
                           nlat=None,
                           with_time=True):
    # if with_time, pred/target in shape: b t nlat nlon or b t l nlat nlon
    # else, pred/target in shape: b nlat nlon or b l nlat nlon

    if nlat is None:
        nlat = target.shape[2]
    if not with_poles:
        lat_end = (nlat-1)*(360/nlon) / 2
        lat_weight = _weight_for_latitude_vector_without_poles(np.linspace(-lat_end, lat_end, nlat))
    else:
        lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))

    lat_weight = torch.from_numpy(lat_weight).to(target.device)
    lat_weight = lat_weight / lat_weight.mean()
    if with_time:
        if len(pred.shape) == 5:
            lat_weight = lat_weight.view(1, 1, nlat, 1, 1)
            pred = rearrange(pred, 'b t l nlat nlon -> b t nlat nlon l')
            target = rearrange(target, 'b t l nlat nlon -> b t nlat nlon l')
        else:
            lat_weight = lat_weight.view(1, 1, nlat, 1)
        return torch.sqrt((((pred - target)**2) * lat_weight).mean(dim=(2, 3)))   # spatial averaging
    else:
        if len(pred.shape) == 4:
            lat_weight = lat_weight.view(1, nlat, 1, 1)
            pred = rearrange(pred, 'b l nlat nlon -> b nlat nlon l')
            target = rearrange(target, 'b l nlat nlon -> b nlat nlon l')
        else:
            lat_weight = lat_weight.view(1, nlat, 1)
        return torch.sqrt((((pred - target)**2) * lat_weight).mean(dim=(1, 2)))   # spatial averaging

def rmse(pred, target):
    # directly infer latitude from target: b t nface nside nside or b t nface nside nside l
    return torch.sqrt(((pred - target)**2).mean(dim=(2, 3, 4)))   # spatial averaging

def latitude_weighted_l1(pred, target):
    # directly infer latitude from target: b t nlat nlon or b t nlat nlon l
    nlat = target.shape[2]
    lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))
    lat_weight = torch.from_numpy(lat_weight).to(target.device)
    lat_weight = lat_weight / lat_weight.mean()
    if len(pred.shape) == 5:
        lat_weight = lat_weight.view(1, 1, nlat, 1, 1)
    else:
        lat_weight = lat_weight.view(1, 1, nlat, 1)

    return ((pred - target).abs() * lat_weight).mean(dim=(2, 3))   # spatial averaging


class SpectralBaseLoss(nn.Module):
    """
    Geometric base loss class used by all geometric losses
    """

    def __init__(
        self,
        img_shape = (180, 360),
        grid_type = 'equiangular',
    ):
        super().__init__()

        self.img_shape = img_shape

        self.sht = th.RealSHT(*img_shape, grid=grid_type).float()

        # get the local l weights
        lmax = self.sht.lmax
        # l_weights = 1 / (2*ls+1)
        l_weights = torch.ones(lmax)

        # get the local m weights
        mmax = self.sht.mmax
        m_weights = 2 * torch.ones(mmax)#.reshape(1, -1)
        m_weights[0] = 1.0

        # get meshgrid of weights:
        l_weights, m_weights = torch.meshgrid(l_weights, m_weights, indexing="ij")

        # use the product weights
        lm_weights = l_weights * m_weights

        # register
        self.register_buffer("lm_weights", lm_weights, persistent=False)

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:

        forecasts = self.sht(forecasts) / 4.0 / math.pi
        observations = self.sht(observations) / 4.0 / math.pi

        forecasts = torch.abs(forecasts)
        observations = torch.abs(observations)

        # we assume the following shapes:
        # forecasts: batch, channels, mmax, lmax
        # observations: batch, channels, mmax, lmax
        B, C, H, W = forecasts.shape

        spectral_weights = self.lm_weights

        # crps w/ E = 1
        crps = torch.abs(observations - forecasts).reshape(B, C, H * W)
        norm = torch.abs(observations).reshape(B, C, H * W)
        spectral_weights_split = spectral_weights.reshape(1, 1, H * W)
       
        # perform spatial average of crps score
        crps = torch.sum(crps * spectral_weights_split, dim=-1)
        norm = torch.sum(norm * spectral_weights_split, dim=-1)

        return crps.mean() / norm.mean() # dimension 