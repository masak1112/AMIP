import torch
import torch.nn as nn

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

class SI_Scheduler(nn.Module):
    def __init__(self,
                 num_refinement_steps,  # this corresponds to physical time steps
                 sampler='euler',
                 train_sampler='logit_normal',
                 inference_sampler='power',
                 rho = 3.0,
                 noise_shape = (180, 360, 151)
                 ):
        super(SI_Scheduler, self).__init__()

        self.num_refinement_steps = num_refinement_steps
        self.method = sampler
        self.train_sampler = train_sampler
        self.inference_sampler = inference_sampler
        self.rho = rho
        self.noise_shape = noise_shape

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

    def get_initial_noise(self, x):
        """Generate initial noise for sampling, matching batch size from conditioning input x."""
        b = x.shape[0]
        if self.noise_shape is not None:
            return torch.randn(b, *self.noise_shape, device=x.device, dtype=x.dtype)
        else:
            return torch.randn_like(x)

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

    def compute_loss(self, x, y, model, **kwargs):
        """

        Args:
            x: conditional information. [b c zlat zlon]
            y: Target state. [b, c, h, w]
            model: velocity predictor
            **kwargs: additional arguments for the model (e.g., conditioning)

        Returns:
            scalar loss
        """

        device = x.device

        # sample timestep, no need to train on t=1
        if self.train_sampler == "logit_normal":
            t = sample_logit_normal(x.shape[0], device=device)
        elif self.train_sampler == "uniform":
            t = torch.rand(x.shape[0], device=device)  # shape (b,)
        else:
            raise ValueError(f"Unknown train_sampler: {self.train_sampler}")

        x0 = y
        eps = self.get_noise(y) # source is the noise distribution

        I_t = self.I(x0, eps, t)  # shape (b, d, nx, ny)
        dIdt = self.dIdt(x0, eps, t)  # shape (b, d, nx, ny)

        v_pred = model(I_t, t=t[:, None], cond=x, **kwargs) # pass lowres as conditioning

        loss = self.image_sq_norm(v_pred - dIdt)  # shape (b,)

        return loss.mean()

    @torch.no_grad()
    def sample(self, x, model, num_steps=None, **kwargs):

        if num_steps is None:
            num_steps = self.num_refinement_steps

        if self.inference_sampler == "power":
            timesteps = sample_power_law(num_steps + 1, self.rho, device = x.device)
        elif self.inference_sampler == "uniform":
            timesteps = torch.linspace(1, 0, num_steps + 1, device=x.device)
        else:
            raise ValueError(f"Unknown inference_sampler: {self.inference_sampler}")

        y = self.get_initial_noise(x)

        for i_t in range(len(timesteps) - 1):
            t_current = timesteps[i_t]
            t_next = timesteps[i_t + 1]
            dt = t_next - t_current  # negative (integrating 1 -> 0)

            t_batch = t_current.expand(y.shape[0]).float().unsqueeze(-1)

            v = model(y, t_batch, cond=x, **kwargs)  # predict velocity at current timestep

            if self.method == 'em':  # SDE (Euler-Maruyama)
                drift = self.sde_drift(v, y, t_batch) # drift correction
                noise_t = self.sigma(t_batch)
                dW = torch.sqrt(torch.abs(dt)) * torch.randn_like(y)
                y = y + drift * dt + noise_t * dW
            else:  # ODE (Euler)
                y = y + v * dt

        return y

    def forward(self, x_lowres, model, num_steps=None, **kwargs):
        return self.sample(x_lowres, model, num_steps=num_steps, **kwargs)
