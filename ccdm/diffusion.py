"""DDPM forward/reverse process with an x0-predicting network."""

import math

import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (2021), https://openreview.net/forum?id=-NEXDKk8gZ"""
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999).float()


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather ``a[t]`` and reshape to broadcast against a tensor of shape ``x_shape``."""
    return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion:
    """Precomputed DDPM coefficients for ``n_steps`` steps (t = 0, ..., n_steps - 1)."""

    def __init__(self, n_steps: int, device: torch.device):
        self.n_steps = n_steps
        betas = cosine_beta_schedule(n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt().to(device)
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt().to(device)
        self.posterior_mean_coef1 = (betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)).to(device)
        self.posterior_mean_coef2 = ((1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)).to(device)
        self.posterior_log_variance_clipped = posterior_variance.clamp(min=1e-20).log().to(device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0)."""
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def p_sample(self, x_t: torch.Tensor, x0_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_{t-1} ~ q(x_{t-1} | x_t, x0_pred); no noise is added at t == 0."""
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        nonzero_mask = (t != 0).float().reshape(-1, *((1,) * (x_t.dim() - 1)))
        return mean + nonzero_mask * (0.5 * log_var).exp() * torch.randn_like(x_t)
