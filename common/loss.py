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
                 normalize = True,
                 eps = 1e-3,
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
        self.eps = eps

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
            surface_loss = surface_loss / (torch.norm(surface_target, p=2, keepdim=True) + self.eps)
            multi_level_loss = multi_level_loss / (torch.norm(multilevel_target, p=2, keepdim=True) + self.eps)
            diag_loss = diag_loss / (torch.norm(diagnostic_target, p=2, keepdim=True) + self.eps)

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


class FairCRPSLoss(nn.Module):
    """
    Almost-fair CRPS estimator for N=2 ensemble members (ACE2S, Appendix A):

    afCRPS_{α,M}(F,y) = E[|X-y|] - (1 - (1-α)/M) * (1/2) * E[|X-X'|]

    When alpha=1.0, this reduces to the standard fair CRPS (FGN Eq. 5).
    When alpha<1.0 (e.g. 0.95), the spread penalty is slightly reduced,
    encouraging more ensemble diversity.

    Supports latitude weighting, level weighting, and per-variable weighting
    matching the WeightedLoss interface.
    """
    def __init__(self,
                 latitude_resolution=180,
                 longitude_resolution=360,
                 with_poles=False,
                 latitude_weight='equal',
                 level_weight='equal',
                 multi_level_variable_weight=None,
                 surface_variable_weight=None,
                 diag_variable_weight=None,
                 nlevels=26,
                 nsurface=6,
                 nmulti=9,
                 ndiag=9,
                 alpha=1.0,
                 n_ensemble=2,
                 ):
        super().__init__()
        # Latitude weighting (same as WeightedLoss)
        if latitude_weight == 'cosine':
            if with_poles:
                latitude = np.linspace(-90, 90, latitude_resolution)
                weights = _weight_for_latitude_vector_with_poles(latitude)
            else:
                lat_end = (latitude_resolution - 1) * (360 / longitude_resolution) / 2
                latitude = np.linspace(-lat_end, lat_end, latitude_resolution)
                weights = _weight_for_latitude_vector_without_poles(latitude)
            weights = torch.from_numpy(weights)
            latitude_weight = weights / weights.mean()
        else:
            weights = torch.ones(latitude_resolution)
            latitude_weight = weights / weights.mean()
        self.register_buffer('latitude_weight', latitude_weight)

        # Level weighting
        if level_weight == 'linear':
            level_weight = torch.linspace(0.065, 0.05, nlevels)
        elif level_weight == 'exp':
            level_weight = torch.exp(torch.linspace(0, -3, nlevels))
            level_weight = level_weight / level_weight.sum()
        elif level_weight == 'cosine':
            level_weight = torch.cos(torch.linspace(0, math.pi / 2, nlevels))
            level_weight = level_weight / level_weight.sum()
        else:
            level_weight = torch.ones(nlevels)
            level_weight = level_weight / level_weight.sum()
        self.register_buffer('level_weight', level_weight)

        # Variable weighting
        if surface_variable_weight is None:
            surface_variable_weight = torch.ones(nsurface)
        else:
            surface_variable_weight = torch.tensor(surface_variable_weight, dtype=torch.float32)

        if multi_level_variable_weight is None:
            multi_level_variable_weight = torch.ones(nmulti)
        else:
            multi_level_variable_weight = torch.tensor(multi_level_variable_weight, dtype=torch.float32)

        if diag_variable_weight is None:
            diag_variable_weight = torch.ones(ndiag)
        else:
            diag_variable_weight = torch.tensor(diag_variable_weight, dtype=torch.float32)

        self.register_buffer('diag_variable_weight', diag_variable_weight)
        self.register_buffer('surface_variable_weight', surface_variable_weight)
        self.register_buffer('multi_level_variable_weight', multi_level_variable_weight)

        # Almost-fair CRPS spread factor: (1 - (1-alpha)/M)
        # alpha=1.0 gives standard fair CRPS, alpha=0.95 gives almost-fair
        self.spread_factor = 1.0 - (1.0 - alpha) / n_ensemble
        self.N = n_ensemble

    def _fair_crps_gridpoint(self, pred1, pred2, target):
        """Compute per-gridpoint almost-fair CRPS for N=2 ensemble members."""

        return (1 / self.N) * (torch.abs(pred1 - target) + torch.abs(pred2 - target)) \
               - self.spread_factor * 0.5 * (1 / self.N) * torch.abs(pred1 - pred2)

    def forward(self,
                surface_pred1, surface_pred2, surface_target,
                multilevel_pred1, multilevel_pred2, multilevel_target,
                diagnostic_pred1, diagnostic_pred2, diagnostic_target):
        """
        All surface/diagnostic tensors: (B, nlat, nlon, C)
        All multilevel tensors: (B, nlevel, nlat, nlon, C)
        """
        # Surface CRPS: (B, nlat, nlon, nsurface)
        surface_crps = self._fair_crps_gridpoint(surface_pred1, surface_pred2, surface_target)
        surface_crps = surface_crps * self.surface_variable_weight.view(1, 1, 1, -1)
        surface_crps = surface_crps.sum(dim=-1)  # (B, nlat, nlon)

        # Diagnostic CRPS: (B, nlat, nlon, ndiag)
        diag_crps = self._fair_crps_gridpoint(diagnostic_pred1, diagnostic_pred2, diagnostic_target)
        diag_crps = diag_crps * self.diag_variable_weight.view(1, 1, 1, -1)
        diag_crps = diag_crps.sum(dim=-1)  # (B, nlat, nlon)

        # Multilevel CRPS: (B, nlevel, nlat, nlon, nmulti)
        multi_crps = self._fair_crps_gridpoint(multilevel_pred1, multilevel_pred2, multilevel_target)
        multi_crps = multi_crps * self.level_weight.view(1, -1, 1, 1, 1)
        multi_crps = multi_crps.sum(dim=1)  # (B, nlat, nlon, nmulti)
        multi_crps = multi_crps * self.multi_level_variable_weight.view(1, 1, 1, -1)
        multi_crps = multi_crps.sum(dim=-1)  # (B, nlat, nlon)

        loss = surface_crps + multi_crps + diag_crps  # (B, nlat, nlon)
        latitude_weight = self.latitude_weight.view(1, -1, 1)
        loss = loss * latitude_weight

        return loss.mean()


class SpectralBaseLoss(nn.Module):
    """
    Geometric base loss class used by all geometric losses
    """

    def __init__(
        self,
        img_shape = (180, 360),
        grid_type = 'equiangular',
        eps = 1e-3,
        absolute = False
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

        self.eps = eps
        self.absolute = absolute

        # register
        self.register_buffer("lm_weights", lm_weights, persistent=False)

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:

        forecasts = self.sht(forecasts) / 4.0 / math.pi
        observations = self.sht(observations) / 4.0 / math.pi

        if self.absolute:
            forecasts = torch.abs(forecasts)
            observations = torch.abs(observations)
        else:
            forecasts = torch.view_as_real(forecasts)
            observations = torch.view_as_real(observations)

            # merge complex dimension after channel dimension and flatten
            # this needs to be undone at the end
            forecasts = torch.movedim(forecasts, 4, 2).flatten(1, 2)
            observations = torch.movedim(observations, 4, 2).flatten(1, 2)

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

        return crps.mean() / (norm.mean() + self.eps) # dimension 

def rankdata(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    ordinal ranking along dimension dim
    """
    ndim = x.dim()
    perm = torch.argsort(x, dim=dim, descending=False, stable=True)

    idx = torch.arange(x.shape[dim], device=x.device).reshape([-1 if i == dim else 1 for i in range(ndim)])
    rank = torch.empty_like(x, dtype=torch.long).scatter_(dim=dim, index=perm, src=idx.expand_as(perm)) + 1
    return rank

def _crps_skillspread_kernel(observation: torch.Tensor, forecasts: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    alternative CRPS variant that uses spread and skill
    """

    observation = observation.unsqueeze(0)

    # get the ranks for the spread computation
    rank = rankdata(forecasts, dim=0)

    #  ensemble size
    num_ensemble = forecasts.shape[0]

    # get the ensemble spread (total_weight is ensemble size here)
    espread = 2 * torch.mean((2 * rank - num_ensemble - 1) * forecasts, dim=0) * (float(num_ensemble) - 1.0 + alpha) / float(num_ensemble * (num_ensemble - 1))
    eskill = (observation - forecasts).abs().mean(dim=0)

    # crps = torch.where(nanmasks.sum(dim=0) != 0, torch.nan, eskill - 0.5 * espread)
    crps = eskill - 0.5 * espread

    return crps

class SpectralCRPSLoss(nn.Module):
    """
    Geometric base loss class used by all geometric losses
    """

    def __init__(
        self,
        img_shape = (180, 360),
        grid_type = 'equiangular',
        eps = 1e-3,
        absolute = False,
        alpha=0.95,
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

        self.eps = eps
        self.absolute = absolute
        self.alpha = alpha

        # register
        self.register_buffer("lm_weights", lm_weights, persistent=False)

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:

        # get the data type before stripping amp types
        dtype = forecasts.dtype

        forecasts = self.sht(forecasts) / 4.0 / math.pi
        observations = self.sht(observations) / 4.0 / math.pi

        if self.absolute:
            forecasts = torch.abs(forecasts).to(dtype)
            observations = torch.abs(observations).to(dtype)
        else:
            forecasts = torch.view_as_real(forecasts).to(dtype)
            observations = torch.view_as_real(observations).to(dtype)

            # merge complex dimension after channel dimension and flatten
            # this needs to be undone at the end
            forecasts = torch.movedim(forecasts, 5, 3).flatten(2, 3)
            observations = torch.movedim(observations, 4, 2).flatten(1, 2)

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, mmax, lmax
        # observations: batch, channels, mmax, lmax
        B, E, C, H, W = forecasts.shape

        spectral_weights = self.lm_weights

        # transpose forecasts: ensemble, batch, channels, lat, lon
        forecasts = torch.movedim(forecasts, 1, 0)

        # now we need to transpose the forecasts into ensemble direction.
        # ideally we split spatial dims
        forecasts = forecasts.reshape(E, B, C, H * W)

        # observations does not need a transpose, but just a split
        observations = observations.reshape(B, C, H * W)

        # tile in complex dim, then flatten last 3 dims
        spectral_weights_split = spectral_weights.reshape(1, 1, H * W)


        crps = _crps_skillspread_kernel(observations, forecasts, self.alpha)

        # perform spatial average of crps score
        crps = torch.sum(crps * spectral_weights_split, dim=-1)

        # finally undo the folding of the complex dimension into the channel dimension
        if not self.absolute:
            crps = crps.reshape(B, -1, 2).sum(dim=-1)

        # the resulting tensor should have dimension B, C, which is what we return
        return torch.mean(crps)