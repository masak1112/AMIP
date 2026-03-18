import torch
import torch.nn as nn

class SI_Scheduler(nn.Module):
    def __init__(self,
                 num_refinement_steps,  # this corresponds to physical time steps
                 integrator='em',
                 ):
        super(SI_Scheduler, self).__init__()

        self.num_refinement_steps = num_refinement_steps
        self.method = integrator

    def wide(self, t, ndim=2):
        if ndim == 2:
            return t[:, None, None, None]
        elif ndim == 3:
            return t[:, None, None, None, None]

    def alpha(self, t, ndim=2):
        return self.wide(1 - t, ndim)

    def alpha_dot(self, t, ndim=2):
        return self.wide(-1.0 * torch.ones_like(t), ndim)

    def sigma(self, t, ndim=2):
        return self.wide(t, ndim)

    def sigma_dot(self, t, ndim=2):
        return self.wide(torch.ones_like(t), ndim)

    def I(self, x0, eps, t, ndim=2):
        return self.alpha(t, ndim) * x0 + self.sigma(t, ndim) * eps

    def dIdt(self, x0, eps, t, ndim=2):
        return self.alpha_dot(t, ndim) * x0 + self.sigma_dot(t, ndim) * eps

    def get_noise(self, x):
        return torch.randn(x.shape, device=x.device, dtype=x.dtype)

    def sde_drift(self, v, x, t, ndim=2):
        """Score-corrected drift for the generative SDE.

        Generative SDE (Eq. 4): dX = [v - (1/2)w_t s]dt + sqrt(w_t) dW_bar

        For linear interpolant (alpha=1-t, sigma=t):
          score:      s(x,t) = -((1-t)v + x) / t
          w_t:        sigma(t)^2 = t^2
          correction: -(1/2) t^2 s = (t/2)((1-t)v + x)
        """
        t_wide = self.wide(t, ndim)
        alpha_t = self.alpha(t, ndim)
        correction = (t_wide / 2) * (alpha_t * v + x)
        return v + correction

    def image_sq_norm(self, x):
        return x.pow(2).sum(-1).sum(-1).sum(-1)

    def compute_loss(self, x_lowres, x_highres, model):
        """

        Args:
            x_lowres: [b, c, h, w] — m(x1), the upsampled low-res (source base);
                      also concatenated channel-wise with I_t as model input
            x_highres: [b, c, h, w] — x1, ground truth (target distribution)
            model: velocity predictor, called as model(I_t, t, cond=x_lowres, history=cond)
            cond: [b, c, h, w] — optional high-res prior state/history for cross-attention

        Returns:
            scalar loss
        """

        device = x_lowres.device

        # sample timestep, no need to train on t=1
        t = torch.rand(x_lowres.shape[0], device=device)  # shape (b,)

        x0 = x_highres - x_lowres # target is the resolution residual
        eps = self.get_noise(x_lowres) # source is the noise distribution

        I_t = self.I(x0, eps, t)  # shape (b, d, nx, ny)
        dIdt = self.dIdt(x0, eps, t)  # shape (b, d, nx, ny)

        v_pred = model(I_t, x_lowres, t=t[:, None]) # pass lowres as conditioning

        loss = self.image_sq_norm(v_pred - dIdt)  # shape (b,)

        return loss.mean()

    @torch.no_grad()
    def sample(self, x_lowres, model, num_steps=None):

        if num_steps is None:
            num_steps = self.num_refinement_steps

        timesteps = torch.linspace(1, 0, num_steps + 1, device=x_lowres.device)

        # start y at source distribution (standard Gaussian, matching sigma(1)=1)
        y = self.get_noise(x_lowres)

        for i_t in range(len(timesteps) - 1):
            t_current = timesteps[i_t]
            t_next = timesteps[i_t + 1]
            dt = t_next - t_current  # negative (integrating 1 -> 0)

            t_batch = t_current.expand(y.shape[0])
            scalar_in = t_batch.float().unsqueeze(-1)

            v = model(y, x_lowres, scalar_in)

            if self.method == 'em':  # SDE (Euler-Maruyama)
                drift = self.sde_drift(v, y, t_batch)
                noise_t = self.sigma(t_batch)
                dW = torch.sqrt(torch.abs(dt)) * torch.randn_like(y)
                y = y + drift * dt + noise_t * dW
            else:  # ODE (Euler)
                y = y + v * dt

        out = x_lowres + y

        return out

    def forward(self, x_lowres, model, num_steps=None):
        return self.sample(x_lowres, model, num_steps=num_steps)
