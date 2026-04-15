import torch 
from einops import rearrange
from torch_harmonics import InverseRealSHT
import torch.nn as nn

def get_log_uniform_t(t_final = 0.999, scale=1.3, n_t = 10, device = "cpu"):
    t_s = []
    t_0 = 0.0

    t_s.append(t_0)

    r = scale * (1-t_final)**(1/n_t)

    assert r < 1.0, "scale is too large for given t_final and n_t, resulting in r >= 1.0"

    for _ in range(n_t):
        delta_t = (1-r) * (1-t_0)
        t_s.append(t_0 + delta_t)
        t_0 = t_0 + delta_t

    return torch.tensor(t_s, device = device), torch.tensor(r, device = device)


def sample_logit_normal(shape, m=0.0, s=1.0, device='cpu', dtype=torch.float32):
    """
    Samples from a logit-normal distribution.
    
    Args:
        shape (tuple or int): The shape of the desired output tensor (e.g., batch size).
        m (float or torch.Tensor): Location parameter (mean of the underlying normal distribution).
                                   Negative biases towards data (p0), positive towards noise (p1).
        s (float or torch.Tensor): Scale parameter (standard deviation of the normal distribution).
        device (str or torch.device): Device to place the tensor on.
        dtype (torch.dtype): Data type of the tensor.
        
    Returns:
        torch.Tensor: Timestep samples 't' in the range (0, 1).
    """
    # 1. Sample u ~ N(m, s)
    # torch.randn generates samples from N(0, 1)
    u = torch.randn(shape, device=device, dtype=dtype)
    u = u * s + m
    
    # 2. Map it through the standard logistic function (sigmoid)
    # sigmoid(u) = 1 / (1 + exp(-u))
    t = torch.sigmoid(u)
    return t

def sample_power_law(n_steps, rho, device = 'cpu'):
    """
    Sample timesteps according to a power-law distribution.
    
    Args:
        n_steps (int): Number of timesteps to sample.
        rho (float): Power-law exponent. Higher values concentrate samples near 0.
    
    Returns:
        torch.Tensor: Timesteps sampled from the power-law distribution, in the range (0, 1).
    """
    n = torch.arange(0, n_steps, device=device, dtype=torch.float32)
    t = (1 -n / (n_steps-1)) ** rho
    
    # returns n_steps values from 1 to 0, with more concentration near 0 for higher rho
    return t

def power_sampler(batch_size, p=2.0, device = "cpu"):
    t = torch.rand(batch_size, device=device)
    return t ** p

class SphereNoiseGenerator(nn.Module):
    def __init__(self, l_max):
        super(SphereNoiseGenerator, self).__init__()
        self.l_max = l_max
        self.isht = InverseRealSHT(l_max, l_max*2, grid="equiangular")

    def forward(self, b, c, device, dtype=torch.complex64, l_max=None):
        # sample coefficient in the frequency domain
        # b: batch size, l_max: maximum degree
        # return: [b, l_max, l_max + 1] # coefficient for real harmonics
        if l_max is None:
            l_max = self.l_max
            coeffs = torch.randn(b*c, l_max, l_max + 1, device=device, dtype=dtype)
        else:
            assert l_max <= self.l_max
            coeffs = torch.randn(b*c, self.l_max, self.l_max + 1, device=device, dtype=dtype)
            # fill with zeros
            coeffs[:, l_max:, :] = 0

        noise = self.isht(coeffs)
        noise = rearrange(noise, '(b c) h w -> b c h w ', b=b, c=c)
        noise_means = torch.mean(noise, dim=(2, 3), keepdim=True)
        noise_stds = torch.std(noise, dim=(2, 3), keepdim=True)
        noise = (noise - noise_means) / noise_stds

        return noise