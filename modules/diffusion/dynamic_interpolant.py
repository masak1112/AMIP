import torch
import torch.nn as nn
from common.utils import disassemble_input
import torch.nn.functional as F

class Integrator:
    def __init__(self,
                 method='em',
                 ):
        self.method = method

    def step_fn(self, y, drift, dt, noise_t):
        # g_t: (b,) scalar sigma per sample; broadcast per-tensor to handle different ndims
        if self.method == 'em':  # Euler-Maruyama
            dW = torch.sqrt(dt) * torch.randn_like(y)
            y_next = y + drift * dt + noise_t * dW
        elif self.method == 'euler':  # ODE
            y_next = y + drift * dt
        return y_next

    def integrate(self,
                  y, c, c_scalar,
                  model, timesteps, noise_fn):
        
        # y is current state along interpolant (noised prognostic states)
        # c is conditioning (current prognostic + forcing state)
        # c_scalar is scalar conditioning (e.g. time, invariants)

        for i_t in range(len(timesteps) - 1):
            t_current = timesteps[i_t]
            t_next = timesteps[i_t + 1]
            dt = t_next - t_current
            noise_t = noise_fn(t_current.expand(y.shape[0]))  # shape (b, 1, 1, 1)

            scalar_in = torch.cat([c_scalar, t_current.float().expand(y.shape[0]).unsqueeze(-1)], dim=-1)

            drift = model(y, c, scalar_in)

            y = self.step_fn(y, drift, dt, noise_t)

        return y

class DriftScheduler(nn.Module):
    def __init__(self,
                 num_refinement_steps,  # this corresponds to physical time steps
                 num_train_steps=None,  # number of training steps
                 integrator='em',
                 sigma_coef=1.0,
                 beta_fn="t",
                 antithetic_sampling=False,
                 sigma_sample=None,
                 ):
        super(DriftScheduler, self).__init__()

        self.num_train_timesteps = num_train_steps if num_train_steps is not None else num_refinement_steps + 1
        self.num_refinement_steps = num_refinement_steps
        self.sigma_coef = sigma_coef
        self.method = integrator
        self.integrator = Integrator(method=integrator)

        self.beta_fn = beta_fn
        self.antithetic_sampling = antithetic_sampling
        self.sigma_sample = sigma_sample if sigma_sample is not None else sigma_coef

        print(f'Scheduler initialized with {self.num_train_timesteps} training steps and {self.num_refinement_steps} refinement steps.')
        print(f"sigma_coef: {self.sigma_coef}, integrator: {integrator}, beta_fn: {self.beta_fn}, antithetic_sampling: {self.antithetic_sampling}")

    def wide(self, t, ndim=2):
        if ndim == 2:
            return t[:, None, None, None]
        elif ndim == 3:
            return t[:, None, None, None, None]

    def alpha(self, t, ndim=2):
        return self.wide(1 - t, ndim)

    def alpha_dot(self, t, ndim=2):
        return self.wide(-1.0 * torch.ones_like(t), ndim)

    def beta(self, t, ndim=2):
        if self.beta_fn == "t":
            return self.wide(t, ndim)
        elif self.beta_fn == "t^2":
            return self.wide(t ** 2, ndim)

    def beta_dot(self, t, ndim=2):
        if self.beta_fn == "t":
            return self.wide(torch.ones_like(t), ndim)
        elif self.beta_fn == "t^2":
            return self.wide(2.0 * t, ndim)

    def sigma(self, t, sample=False, ndim=2):
        if sample:
            return self.sigma_sample * self.wide(1 - t, ndim)
        else:
            return self.sigma_coef * self.wide(1 - t, ndim)

    def sigma_dot(self, t, sample=False, ndim=2):
        if sample:
            return self.sigma_sample * self.wide(-1.0 * torch.ones_like(t), ndim)
        else:
            return self.sigma_coef * self.wide(-1.0 * torch.ones_like(t), ndim)

    def I(self, x0, x1, t, ndim=2):
        return self.alpha(t, ndim) * x0 + self.beta(t, ndim) * x1

    def dIdt(self, x0, x1, t, ndim=2):
        return self.alpha_dot(t, ndim) * x0 + self.beta_dot(t, ndim) * x1

    def get_noise(self, x):
        return torch.randn(x.shape, device=x.device, dtype=x.dtype)

    def compute_loss(self, model, criterion, x, c_grid, c_scalar, y):
        # x contains current prognostic state
        # c_grid contains current forcing state
        # c_scalar contains current scalar conditioning (hod, doy)
        # y contains next prognostic state 

        device = x.device

        noise = self.get_noise(x)
        # sample timestep, no need to train on t=1
        t = torch.randint(0, self.num_train_timesteps - 1, device=device, size=(x.shape[0],)).float() / (self.num_train_timesteps - 1)

        sigma_t = self.sigma(t)          # shape (b, 1, 1, 1)
        sigma_dot_t = self.sigma_dot(t)  # shape (b, 1, 1, 1)
        W_t = self.wide(torch.sqrt(t))   # shape (b, 1, 1, 1)

        I = self.I(x, y, t)  # shape (b, d, nx, ny)
        dIdt = self.dIdt(x, y, t)  # shape (b, d, nx, ny)

        c_scalar = torch.cat([c_scalar, t.view(-1, 1)], dim=-1) # shape (b, c_dim + 1)
        # use current state + forcing as conditioning
        c = torch.cat([x, c_grid], dim=1) # shape (b, d + c_dim, nx, ny)

        if self.antithetic_sampling:
            I_noised_p = I + sigma_t * W_t * noise
            I_noised_m = I - sigma_t * W_t * noise

            target_p = dIdt + sigma_dot_t * W_t * noise
            target_m = dIdt - sigma_dot_t * W_t * noise

            pred_p = model(I_noised_p, c, c_scalar)
            pred_m = model(I_noised_m, c, c_scalar)

            surface_pred_p, multi_pred_p, diag_pred_p = disassemble_input(pred_p)
            surface_pred_m, multi_pred_m, diag_pred_m = disassemble_input(pred_m)
            target_surface_p, target_multilevel_p, target_diagnostic_p = disassemble_input(target_p)
            target_surface_m, target_multilevel_m, target_diagnostic_m = disassemble_input(target_m)

            loss_p = criterion(surface_pred_p, target_surface_p,
                             multi_pred_p, target_multilevel_p,
                             diag_pred_p, target_diagnostic_p)
            loss_m = criterion(surface_pred_m, target_surface_m,
                                    multi_pred_m, target_multilevel_m,
                                    diag_pred_m, target_diagnostic_m)
            loss = 0.5 * (loss_p + loss_m)
        else:
            I_noised = I + sigma_t * W_t * noise
            target = dIdt + sigma_dot_t * W_t * noise

            pred = model(I_noised, c, c_scalar)

            loss = F.mse_loss(pred, target)

            #surface_pred, multi_pred, diag_pred = disassemble_input(pred)
            #target_surface, target_multilevel, target_diagnostic = disassemble_input(target)

            #loss = criterion(surface_pred, target_surface,
            #                 multi_pred, target_multilevel,
            #                 diag_pred, target_diagnostic)

        return loss

    def sample(self, model, x, c_grid, c_scalar, refinement_steps=None):
        # x contains current prognostic state
        # c_grid contains current forcing state
        # c_scalar contains current scalar conditioning (e.g. time, invariants)

        if refinement_steps is None:
            refinement_steps = self.num_refinement_steps

        timesteps = torch.linspace(0, 1, refinement_steps + 1, device=x.device)

        # start y at source distribution, which is current state
        y = x.clone()

        # assemble conditioning, which is current prognostic + forcing state
        c = torch.cat([x, c_grid], dim=1) # shape (b, d + c_dim, nx, ny)

        # first step taken analytically to avoid g_T singularity issues at t=0 with EM
        sigma_0 = self.sigma(timesteps[0].expand(x.shape[0]), sample=True)  # shape (b, 1, 1, 1)
        dt_0 = timesteps[1] - timesteps[0]
        scalar_in = torch.cat([c_scalar, timesteps[0].float().expand(x.shape[0]).unsqueeze(-1)], dim=-1)

        drift = model(y, c, scalar_in)

        if self.method == 'em':
            dW = torch.sqrt(dt_0)
            y = y + drift * dt_0 + sigma_0 * dW * torch.randn_like(y)
        else:
            y = y + drift * dt_0

        noise_fn = lambda t: self.sigma(t, sample=True)
        y = self.integrator.integrate(y, c, c_scalar, model, timesteps[1:], noise_fn)

        return y

    def forward(self, model, x, c_grid, c_scalar, refinement_steps=None):
        return self.sample(model, x, c_grid, c_scalar, refinement_steps)
