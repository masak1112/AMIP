import torch
import torch.nn as nn
from modules.diffusion.utils import sample_logit_normal, power_sampler

class DynamicInterpolant(nn.Module):
    def __init__(self,
                 num_steps,  # this corresponds to physical time steps
                 sigma_coef=1.0,
                 train_sampler='uniform',
                 l_max = 180,
                 spectral_weight = 0.01,
                 noise = "spherical"
                 ):
        super(DynamicInterpolant, self).__init__()

        self.num_steps = num_steps
        self.sigma_coef = sigma_coef
        self.train_sampler = train_sampler 

        if noise == "spherical":
            from modules.diffusion.utils import SphereNoiseGenerator
            self.generator = SphereNoiseGenerator(l_max=l_max)
        else:
            self.generator = None

        self.spectral_weight = spectral_weight

        if self.spectral_weight > 0: # apply  spectral regularization to model outputs
            from common.loss import SpectralScalarLoss
            self.spectral_criterion = SpectralScalarLoss(img_shape=(l_max, l_max*2))

        print(f"sigma_coef: {self.sigma_coef}, train_sampler: {self.train_sampler}")

    def wide(self, t):
        return t[:, None, None, None]
    
    def get_noise(self, x):
        if self.generator is None:
            return torch.randn_like(x, device=x.device)
        else:
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
        elif self.train_sampler == 'power':
            t = power_sampler(x.shape[0], p=2.0, device=device)
        elif self.train_sampler == 'uniform':
            t = torch.rand(x.shape[0], device=device)

        t = self.wide(t) 
        W_t = torch.sqrt(t) * noise
        X_t = (1-t) * x + t * y + (1-t) * self.sigma_coef * W_t

        pred_y = model(X_t, x, t.squeeze(dim=[1, 2, 3]), c_grid)

        loss = ((pred_y - y) ** 2).sum(dim=[1, 2, 3]).mean() 

        if self.spectral_weight > 0:
            spectral_loss = self.spectral_weight * self.spectral_criterion(pred_y, y)
        else:
            spectral_loss = 0

        loss = loss + spectral_loss

        return loss, spectral_loss

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
