import torch
import torch.nn as nn
from modules.diffusion.utils import sample_logit_normal, power_sampler, sample_power_law

class DynamicInterpolant(nn.Module):
    def __init__(self,
                 num_steps,  # this corresponds to physical time steps
                 sigma_coef=1.0,
                 train_sampler='uniform',
                 inference_sampler='uniform',
                 integrator='euler',
                 inference_rho=1.0,
                 l_max = 180,
                 spectral_weight = 0.01,
                 noise = "spherical",
                 model_last = False,
                 loss_form = "x",
                 noise_scale_path = None
                 ):
        super(DynamicInterpolant, self).__init__()

        self.num_steps = num_steps
        self.sigma_coef = sigma_coef
        self.train_sampler = train_sampler
        self.inference_sampler = inference_sampler
        self.inference_rho = inference_rho
        self.model_last = model_last
        self.loss_form = loss_form
        self.integrator = integrator

        if noise == "spherical":
            from modules.diffusion.utils import SphereNoiseGenerator
            self.generator = SphereNoiseGenerator(l_max=l_max)
        else:
            self.generator = None

        self.spectral_weight = spectral_weight

        if self.spectral_weight > 0: # apply  spectral regularization to model outputs
            from common.loss import SpectralScalarLoss
            self.spectral_criterion = SpectralScalarLoss(img_shape=(l_max, l_max*2))

        if noise_scale_path is not None:
            noise_scales = torch.load(noise_scale_path)
            self.register_buffer("noise_scales", noise_scales)
        else:
            self.noise_scales = None

        print(f"sigma_coef: {self.sigma_coef}, train_sampler: {self.train_sampler}")

    def wide(self, t):
        return t[:, None, None, None]
    
    def get_noise(self, x):
        if self.generator is None:
            return torch.randn_like(x, device=x.device)
        else:
            return self.generator(x.shape[0], x.shape[1], device=x.device)

    def compute_multistep_loss(self, model, x, c_grids, y, num_sample_steps=None):
        # x contains initial prognostic state, shape b c h w
        # c_grids contains the forcing state for each rollout step, shape b rollout c h w
        # y contains the final prognostic state after rollout steps, shape b c h w
        # num_sample_steps: truncated schedule length for intermediate rollout samples

        rollout = c_grids.shape[1]

        x_current = x
        if rollout > 1:
            with torch.no_grad():
                for step in range(rollout - 1):
                    x_current = self.sample(model, x_current, c_grids[:, step], num_steps=num_sample_steps)

        return self.compute_loss(model, x_current, c_grids[:, -1], y)

    def compute_loss(self, model, x, c_grid, y):
        # x contains current prognostic state
        # c_grid contains current forcing state
        # y contains next prognostic state

        device = x.device

        noise = self.get_noise(x)

        if self.noise_scales is not None:
            noise = noise * self.noise_scales

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

        if self.loss_form == 'x':
            loss = ((pred_y - y) ** 2).sum(dim=[1, 2, 3]).mean() 
        elif self.loss_form == 'v': # this is trivial, since additive terms cancel before gradient computation
            # construct target
            target = (y - x) - self.sigma_coef * W_t
            pred = (pred_y - x) - self.sigma_coef * W_t
            loss = ((pred - target) ** 2).sum(dim=[1, 2, 3]).mean()

        if self.spectral_weight > 0:
            spectral_loss = self.spectral_weight * self.spectral_criterion(pred_y, y)
        else:
            spectral_loss = 0

        loss = loss + spectral_loss

        return loss, spectral_loss

    def sample(self, model, x, c_grid, num_steps=None, return_model_last=False):
        # x contains current prognostic state (latent space)
        # c_grid contains current forcing state (original resolution)

        if num_steps is None:
            num_steps = self.num_steps

        timesteps = torch.linspace(0, 1, num_steps + 1, device=x.device)

        if self.inference_sampler == 'power':
            timesteps = timesteps**self.inference_rho

        # start y at source distribution, which is current state
        y = x.clone()
        W_t = torch.zeros_like(x)

        if self.model_last:
            num_steps_drift = num_steps - 1
        else:
            num_steps_drift = num_steps

        for i in range(num_steps_drift):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_current  
            
            t_curr_batch = t_current.expand(x.shape[0])
            t_next_batch = t_next.expand(x.shape[0])

            # --- 1. Predictor Step (Standard Euler-Maruyama) ---
            y_pred = model(y, x, t_curr_batch, c_grid) # Predict x_1
            drift_curr = (y_pred - x) - self.sigma_coef * W_t # v_theta(t)
            
            noise = self.get_noise(x)
            if self.noise_scales is not None:
                noise = noise * self.noise_scales
            
            dW = torch.sqrt(dt) * noise
            diffusion_scale = self.sigma_coef * (1 - self.wide(t_curr_batch))
            
            # Temporary next state (Euler predictor)
            y_next_euler = y + drift_curr * dt + diffusion_scale * dW
            W_next = W_t + dW # Advanced accumulated noise

            # --- 2. Corrector Step (Heun) ---
            if self.integrator == 'heun' and i < num_steps_drift - 1:
                # Evaluate model at the predicted state
                y_pred_next = model(y_next_euler, x, t_next_batch, c_grid)
                
                # Use W_next to evaluate drift at the future step
                drift_next = (y_pred_next - x) - self.sigma_coef * W_next
                
                # Apply trapezoidal rule to the drift
                # Diffusion stays first-order (Euler-Maruyama level)
                y = y + 0.5 * (drift_curr + drift_next) * dt + diffusion_scale * dW
            else:
                # Fallback to Euler-Maruyama
                y = y_next_euler
            
            W_t = W_next # Track W_t for the next iteration

        if return_model_last:
            assert self.model_last is False 
            return y, y_pred

        # take last step without drift/noise
        if self.model_last:
            y = model(y, x, timesteps[-2].expand(x.shape[0]), c_grid)

        return y

    def forward(self, model, x, c_grid, num_steps=None):
        return self.sample(model, x, c_grid, num_steps)
