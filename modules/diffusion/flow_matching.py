import torch
import torch.nn as nn

class ODEIntegrator:
    def __init__(self,
                 method='euler',  # 'euler' or 'heun' 
                 ):
        self.method = method
        assert method in ['euler', 'heun'], 'Method not implemented'

    def step_fn(self, surface_input, multilevel_input, forcing_input, invariant_input, scalar_in,
                     surface_noised, multilevel_noised, diagnostic_noised, 
                     model, dt, ts_next):
        method = self.method
        if ts_next == 0 and self.method == 'heun': # prevent irregularity at last time
            method = 'euler'

        if method == 'euler':
            surface_pred, multi_pred, diag_pred = model(surface_input, multilevel_input, forcing_input, invariant_input, scalar_in,
                        surface_noised, multilevel_noised, diagnostic_noised)

            surface_next = surface_noised + dt * surface_pred
            multilevel_next = multilevel_noised + dt * multi_pred
            diagnostic_next = diagnostic_noised + dt * diag_pred

            return surface_next, multilevel_next, diagnostic_next
        
        elif method == 'heun':
            surface_pred, multi_pred, diag_pred = model(surface_input, multilevel_input, forcing_input, invariant_input, scalar_in,
                        surface_noised, multilevel_noised, diagnostic_noised)
            
            surface_next = surface_noised + dt * surface_pred
            multilevel_next = multilevel_noised + dt * multi_pred
            diagnostic_next = diagnostic_noised + dt * diag_pred
            scalar_in_next = torch.cat([scalar_in[:, :-1], ts_next.float().view(-1, 1)], dim=-1)

            surface_pred1, multi_pred1, diag_pred1 = model(surface_input, multilevel_input, forcing_input, invariant_input, scalar_in_next,
                                                           surface_next, multilevel_next, diagnostic_next)

            surface_out = surface_noised + 0.5 * dt * (surface_pred + surface_pred1)
            multilevel_out = multilevel_noised + 0.5 * dt * (multi_pred + multi_pred1)
            diagnostic_out = diagnostic_noised + 0.5 * dt * (diag_pred + diag_pred1)

            return surface_out, multilevel_out, diagnostic_out

    def integrate(self, surface_input, multilevel_input, forcing_input, invariant_input, scalar_input,
                surface_noised, multilevel_noised, diagnostic_noised,
                model, stencils, timesteps,):

        for i_t in range(len(stencils)-1):
            t_current = stencils[i_t] # sigma_t
            t_next = stencils[i_t+1] # sigma_t+1
            dt = t_next - t_current # (sigma_t+1 - sigma_t)

            ts = timesteps[i_t]
            ts_next = timesteps[i_t+1]

            scalar_in = torch.cat([scalar_input, ts.expand(scalar_input.shape[0]).unsqueeze(-1)], dim=-1) 

            surface_noised, multilevel_noised, diagnostic_noised = self.step_fn(
                     surface_input, multilevel_input, forcing_input, invariant_input, scalar_in,
                     surface_noised, multilevel_noised, diagnostic_noised, 
                     model, dt, ts_next)

        return surface_noised, multilevel_noised, diagnostic_noised

class FlowScheduler(nn.Module):
    def __init__(self,
                 num_refinement_steps,  # this corresponds to physical time steps
                 num_train_steps=None,  # number of training steps
                 integrator='euler',  # 'euler' or 'heun' or 'midpoint', worth noting that this only available for flow
                 ):
        super(FlowScheduler, self).__init__()

        # for flow matching, the min_noise_std is not used
        self.num_train_timesteps = num_train_steps if num_train_steps is not None else num_refinement_steps + 1
        self.num_refinement_steps = num_refinement_steps
        self.sigmas = torch.linspace(0, 1,
                                     steps=self.num_train_timesteps)

        self.num_refinement_steps = num_refinement_steps
        self.ode_integrator = ODEIntegrator(method=integrator)

        self.training_criterion = nn.MSELoss()

        print(f"Using LinearScheduler with {self.num_train_timesteps} training steps and {self.num_refinement_steps} refinement steps.")

    def get_noise(self, x):
        return torch.randn(x.shape, device=x.device, dtype=x.dtype)
    
    def interpolant(self, x, noise, alpha, sigma):
        alpha = alpha.view(-1, *[1 for _ in range(x.ndim - 1)])
        sigma = sigma.view(-1, *[1 for _ in range(x.ndim - 1)])

        return alpha * x + sigma * noise

    def compute_loss(self, model, 
                     surface_input,
                    multilevel_input,
                    forcing_input,
                    invariant_input,
                    scalar_input,
                    surface_target,
                    multilevel_target,
                    diagnostic_target):
        # x: [b nx ny d], conditioning. For PDEs this is u(t)
        # y: [b nx ny d], label. For PDEs this is u(t+dt)
        # cond: [b cond_dim]
        
        noise_surface = self.get_noise(surface_target)
        noise_multilevel = self.get_noise(multilevel_target)
        noise_diagnostic = self.get_noise(diagnostic_target)

        # sample timestep (shape (b, )) no need to train on k=0
        k = torch.randint(1, self.num_train_timesteps, device=surface_input.device, size=(surface_input.shape[0],)).long()

        # retrieve from the scheduler
        sigma_t = self.sigmas.to(surface_input.device)[k] # noise coeff, shape b
        alpha_t = (1 - sigma_t) # signal coeff, shape b

        # Noise the labels
        surface_noised = self.interpolant(surface_target, noise_surface, alpha_t, sigma_t)
        multilevel_noised = self.interpolant(multilevel_target, noise_multilevel, alpha_t, sigma_t)
        diagnostic_noised = self.interpolant(diagnostic_target, noise_diagnostic, alpha_t, sigma_t)

        scalar_in = torch.cat([scalar_input, k.float().view(-1, 1)], dim=-1)  # [b cond_dim + 1]

        surface_pred, multi_pred, diag_pred = model(surface_input, multilevel_input, forcing_input, invariant_input, scalar_in,
                     surface_noised, multilevel_noised, diagnostic_noised)
        
        surface_target = noise_surface - surface_target # predict eps - y
        multi_target = noise_multilevel - multilevel_target
        diag_target = noise_diagnostic - diagnostic_target

        surface_loss = self.training_criterion(surface_pred, surface_target)
        multi_loss = self.training_criterion(multi_pred, multi_target)
        diag_loss = self.training_criterion(diag_pred, diag_target)

        return surface_loss + multi_loss + diag_loss

    def sample(self, model, surface_input,
                    multilevel_input,
                    forcing_input,
                    invariant_input,
                    scalar_input,
                    diagnostic_channels,
                    refinement_steps=None):
        
        if refinement_steps is None:
            refinement_steps = self.num_refinement_steps

        # x: [b nlat nlon d]
        surface_noised = self.get_noise(surface_input)
        multilevel_noised = self.get_noise(multilevel_input)

        # diagnostic inputs not provided, so need to manually get its shape 
        diagnostic_noised = torch.randn((surface_input.shape[0], surface_input.shape[1], surface_input.shape[2], diagnostic_channels), 
                                       device = surface_input.device, dtype = surface_input.dtype)

        timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1, device=surface_input.device).long()
        # trailing timesteps
        timesteps = timesteps[::((self.num_train_timesteps - 1) // refinement_steps)]
        sigmas = self.sigmas.to(surface_input.device)[timesteps]

        integrator = self.ode_integrator
        surface_pred, multi_pred, diag_pred = integrator.integrate(surface_input, multilevel_input, forcing_input, invariant_input, scalar_input,
                                        surface_noised, multilevel_noised, diagnostic_noised, 
                                        model, sigmas, timesteps)

        return surface_pred, multi_pred, diag_pred

    def forward(self, model, surface_input,
                    multilevel_input,
                    forcing_input,
                    invariant_input,
                    scalar_input,
                    diagnostic_channels,
                    refinement_steps=None):
        
        return self.sample(model, surface_input,
                    multilevel_input,
                    forcing_input,
                    invariant_input,
                    scalar_input,
                    diagnostic_channels,
                    refinement_steps)