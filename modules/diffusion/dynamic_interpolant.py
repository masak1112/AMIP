import torch
import torch.nn as nn
import torch.nn.functional as F


class Integrator:
    def __init__(self,
                 method='em',
                 ):
        self.method = method

    @staticmethod
    def _w(val, x):
        """Reshape batch coefficient (b,) to broadcast against x's spatial dims."""
        return val.view(-1, *([1] * (x.ndim - 1)))

    def step_fn(self,
                surface_state, multilevel_state, diagnostic_state,
                drift_surface, drift_multilevel, drift_diagnostic,
                dt, g_t):
        # g_t: (b,) scalar sigma per sample; broadcast per-tensor to handle different ndims
        if self.method == 'em':  # Euler-Maruyama
            dW_surface = torch.sqrt(dt) * torch.randn_like(surface_state)
            dW_multilevel = torch.sqrt(dt) * torch.randn_like(multilevel_state)
            dW_diagnostic = torch.sqrt(dt) * torch.randn_like(diagnostic_state)
            surface_next = surface_state + dt * drift_surface + self._w(g_t, surface_state) * dW_surface
            multilevel_next = multilevel_state + dt * drift_multilevel + self._w(g_t, multilevel_state) * dW_multilevel
            diagnostic_next = diagnostic_state + dt * drift_diagnostic + self._w(g_t, diagnostic_state) * dW_diagnostic
        elif self.method == 'euler':  # ODE
            surface_next = surface_state + dt * drift_surface
            multilevel_next = multilevel_state + dt * drift_multilevel
            diagnostic_next = diagnostic_state + dt * drift_diagnostic
        return surface_next, multilevel_next, diagnostic_next

    def integrate(self,
                  surface_input, multilevel_input, diagnostic_input,
                  forcing_input, invariant_input, scalar_input,
                  surface_state, multilevel_state, diagnostic_state,
                  model, timesteps, g_fn):

        for i_t in range(len(timesteps) - 1):
            t_current = timesteps[i_t]
            t_next = timesteps[i_t + 1]
            dt = t_next - t_current
            g_t = g_fn(t_current.expand(surface_input.shape[0]))  # shape (b, 1, 1, 1)

            scalar_in = torch.cat([scalar_input, t_current.float().expand(surface_input.shape[0]).unsqueeze(-1)], dim=-1)

            drift_surface, drift_multilevel, drift_diagnostic = model(
                surface_input, multilevel_input, diagnostic_input,
                forcing_input, invariant_input, scalar_in,
                surface_state, multilevel_state, diagnostic_state
            )

            surface_state, multilevel_state, diagnostic_state = self.step_fn(
                surface_state, multilevel_state, diagnostic_state,
                drift_surface, drift_multilevel, drift_diagnostic,
                dt, g_t
            )

        return surface_state, multilevel_state, diagnostic_state


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

    def compute_loss(self, model, criterion,
                     surface_input, multilevel_input, diagnostic_input,
                     forcing_input, invariant_input, scalar_input,
                     surface_target, multilevel_target, diagnostic_target):
        # surface/multilevel_input: source state (x0), shape [b nx ny d]
        # surface/multilevel/diagnostic_target: target state (x1), shape [b nx ny d]
        # diagnostic has no corresponding input, so a noise draw is used as its source

        device = surface_input.device

        noise_surface = self.get_noise(surface_target)
        noise_multilevel = self.get_noise(multilevel_target)
        noise_diagnostic = self.get_noise(diagnostic_target)

        # sample timestep, no need to train on t=1
        t = torch.randint(0, self.num_train_timesteps - 1, device=device, size=(surface_input.shape[0],)).float() / (self.num_train_timesteps - 1)

        sigma_t = self.sigma(t)          # shape (b, 1, 1, 1)
        sigma_dot_t = self.sigma_dot(t)  # shape (b, 1, 1, 1)
        W_t = self.wide(torch.sqrt(t))   # shape (b, 1, 1, 1)

        I_surface = self.I(surface_input, surface_target, t)
        I_multilevel = self.I(multilevel_input, multilevel_target, t, ndim=3)
        I_diagnostic = self.I(diagnostic_input, diagnostic_target, t)

        dIdt_surface = self.dIdt(surface_input, surface_target, t)
        dIdt_multilevel = self.dIdt(multilevel_input, multilevel_target, t, ndim=3)
        dIdt_diagnostic = self.dIdt(diagnostic_input, diagnostic_target, t)

        scalar_in = torch.cat([scalar_input, t.view(-1, 1)], dim=-1)

        if self.antithetic_sampling:
            surface_noised_p = I_surface + sigma_t * W_t * noise_surface
            multilevel_noised_p = I_multilevel + sigma_t.unsqueeze(-1) * W_t.unsqueeze(-1) * noise_multilevel
            diagnostic_noised_p = I_diagnostic + sigma_t * W_t * noise_diagnostic

            surface_noised_m = I_surface - sigma_t * W_t * noise_surface
            multilevel_noised_m = I_multilevel - sigma_t.unsqueeze(-1) * W_t.unsqueeze(-1) * noise_multilevel
            diagnostic_noised_m = I_diagnostic - sigma_t * W_t * noise_diagnostic

            target_surface_p = dIdt_surface + sigma_dot_t * W_t * noise_surface
            target_multilevel_p = dIdt_multilevel + sigma_dot_t.unsqueeze(-1) * W_t.unsqueeze(-1) * noise_multilevel
            target_diagnostic_p = dIdt_diagnostic + sigma_dot_t * W_t * noise_diagnostic

            target_surface_m = dIdt_surface - sigma_dot_t * W_t * noise_surface
            target_multilevel_m = dIdt_multilevel - sigma_dot_t.unsqueeze(-1) * W_t.unsqueeze(-1) * noise_multilevel
            target_diagnostic_m = dIdt_diagnostic - sigma_dot_t * W_t * noise_diagnostic

            surface_pred_p, multi_pred_p, diag_pred_p = model(
                surface_input, multilevel_input, forcing_input, invariant_input, scalar_in,
                surface_noised_p, multilevel_noised_p, diagnostic_noised_p
            )
            surface_pred_m, multi_pred_m, diag_pred_m = model(
                surface_input, multilevel_input, forcing_input, invariant_input, scalar_in,
                surface_noised_m, multilevel_noised_m, diagnostic_noised_m
            )

            loss = criterion(surface_pred_p, target_surface_p,
                             multi_pred_p, target_multilevel_p,
                             diag_pred_p, target_diagnostic_p)
            loss = loss + criterion(surface_pred_m, target_surface_m,
                                    multi_pred_m, target_multilevel_m,
                                    diag_pred_m, target_diagnostic_m)
        else:
            surface_noised = I_surface + sigma_t * W_t * noise_surface
            multilevel_noised = I_multilevel + sigma_t.unsqueeze(-1) * W_t.unsqueeze(-1) * noise_multilevel
            diagnostic_noised = I_diagnostic + sigma_t * W_t * noise_diagnostic

            target_surface = dIdt_surface + sigma_dot_t * W_t * noise_surface
            target_multilevel = dIdt_multilevel + sigma_dot_t.unsqueeze(-1) * W_t.unsqueeze(-1) * noise_multilevel
            target_diagnostic = dIdt_diagnostic + sigma_dot_t * W_t * noise_diagnostic

            surface_pred, multi_pred, diag_pred = model(
                surface_input, multilevel_input, diagnostic_input, 
                forcing_input, invariant_input, scalar_in,
                surface_noised, multilevel_noised, diagnostic_noised
            )

            loss = criterion(surface_pred, target_surface,
                             multi_pred, target_multilevel,
                             diag_pred, target_diagnostic)

        return loss

    def sample(self, model, surface_input, multilevel_input, diagnostic_input,
               forcing_input, invariant_input, scalar_input, refinement_steps=None):

        if refinement_steps is None:
            refinement_steps = self.num_refinement_steps

        timesteps = torch.linspace(0, 1, refinement_steps + 1, device=surface_input.device)

        # source distribution at t=0: inputs for surface/multi, noise for diagnostic
        surface_state = surface_input.clone()
        multilevel_state = multilevel_input.clone()
        diagnostic_state = diagnostic_input.clone()

        # first step taken analytically to avoid g_T singularity issues at t=0 with EM
        sigma_0 = self.sigma(timesteps[0].expand(surface_input.shape[0]), sample=True)  # shape (b, 1, 1, 1)
        dt_0 = timesteps[1] - timesteps[0]
        scalar_in_0 = torch.cat([scalar_input, timesteps[0].float().expand(surface_input.shape[0]).unsqueeze(-1)], dim=-1)

        drift_surface, drift_multilevel, drift_diagnostic = model(
            surface_input, multilevel_input, diagnostic_input,
            forcing_input, invariant_input, scalar_in_0,
            surface_state, multilevel_state, diagnostic_state
        )

        if self.method == 'em':
            dW = torch.sqrt(dt_0)
            surface_state = surface_state + drift_surface * dt_0 + sigma_0 * dW * torch.randn_like(surface_state)
            multilevel_state = multilevel_state + drift_multilevel * dt_0 + sigma_0 * dW * torch.randn_like(multilevel_state)
            diagnostic_state = diagnostic_state + drift_diagnostic * dt_0 + sigma_0 * dW * torch.randn_like(diagnostic_state)
        else:
            surface_state = surface_state + drift_surface * dt_0
            multilevel_state = multilevel_state + drift_multilevel * dt_0
            diagnostic_state = diagnostic_state + drift_diagnostic * dt_0

        g_fn = lambda t: self.sigma(t, sample=True)
        surface_pred, multi_pred, diag_pred = self.integrator.integrate(
            surface_input, multilevel_input, diagnostic_input,
            forcing_input, invariant_input, scalar_input,
            surface_state, multilevel_state, diagnostic_state,
            model, timesteps[1:], g_fn
        )

        return surface_pred, multi_pred, diag_pred

    def forward(self, model, surface_input, multilevel_input, forcing_input, invariant_input, scalar_input,
                diagnostic_channels, refinement_steps=None):
        return self.sample(model, surface_input, multilevel_input, forcing_input, invariant_input, scalar_input,
                           diagnostic_channels, refinement_steps)
