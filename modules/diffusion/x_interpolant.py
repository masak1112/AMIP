import torch
import torch.nn as nn

from einops import rearrange
from torch_harmonics import InverseRealSHT
from modules.diffusion.stochastic_interpolant import sample_logit_normal

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

class DynamicInterpolant(nn.Module):
    def __init__(self,
                 num_steps,  # this corresponds to physical time steps
                 sigma_coef=1.0,
                 train_sampler='uniform',
                 l_max = 180
                 ):
        super(DynamicInterpolant, self).__init__()

        self.num_steps = num_steps
        self.sigma_coef = sigma_coef
        self.train_sampler = train_sampler

        self.generator = SphereNoiseGenerator(l_max=l_max)

        print(f"sigma_coef: {self.sigma_coef}, train_sampler: {self.train_sampler}")

    def wide(self, t):
        return t[:, None, None, None]
    
    def get_noise(self, x):
        return self.generator(x.shape[0], x.shape[1], device=x.device)

    def compute_loss(self, model, x, c_grid, y):
        # x contains current prognostic state
        # c_grid contains current forcing state
        # y contains next prognostic state 

        device = x.device

        noise = self.get_noise(x)
        # sample timestep
        if self.train_sampler == 'logit_normal':
            t = sample_logit_normal(x.shape[0], device=device)
        elif self.train_sampler == 'uniform':
            t = torch.rand(x.shape[0], device=device)

        t = self.wide(t) 
        W_t = torch.sqrt(t) * noise
        X_t = (1-t) * x + t * y + (1-t) * self.sigma_coef * W_t

        pred_y = model(X_t, x, t.squeeze(dim=[1, 2, 3]), c_grid)

        loss = ((pred_y - y) ** 2).sum(dim=[1, 2, 3]).mean() 

        return loss

    def sample(self, model, x, c_grid, num_steps=None):
        # x contains current prognostic state (latent space)
        # c_grid contains current forcing state (original resolution)

        if num_steps is None:
            num_steps = self.num_steps

        timesteps = torch.linspace(0, 1, num_steps + 1, device=x.device)

        # start y at source distribution, which is current state
        y = x.clone()
        W_t = torch.zeros_like(x)

        for i in range(num_steps - 1):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_current  

            y_pred = model(y, x, t_current.expand(x.shape[0]), c_grid)

            drift = (y_pred - x) - self.sigma_coef * W_t # associated drift from y_pred
            noise = self.get_noise(x) # noise term

            dW = torch.sqrt(dt) * noise

            y = y + drift * dt + self.sigma_coef * (1-self.wide(t_current.expand(y.shape[0]))) * dW

            W_t = W_t + dW

        # take last step without drift/noise
        y = model(y, x, timesteps[-2].expand(x.shape[0]), c_grid)

        return y

    def forward(self, model, x, c_grid, num_steps=None):
        return self.sample(model, x, c_grid, num_steps)
