"""A single-layer, periodic convolutional denoiser.

The network only sees a ``kernel_size x kernel_size`` neighbourhood of each
pixel and is equivariant to (circular) translations, i.e. it has exactly the
two inductive biases -- locality and equivariance -- that the analytic theory
of convolutional diffusion models identifies as the source of creativity.
"""

import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps: torch.Tensor, dim: int = 16, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embeddings.

    Args:
        timesteps: (batch,) tensor of (possibly fractional) timesteps.
        dim: embedding dimension.
        max_period: controls the minimum frequency of the embeddings.

    Returns:
        (batch, dim) tensor of embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def periodic_padding(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Wrap-around (circular) padding on the last two dimensions.

    Args:
        x: (batch, channels, height, width) tensor.
        pad: number of pixels to pad on each side.

    Returns:
        (batch, channels, height + 2 * pad, width + 2 * pad) tensor.
    """
    if pad == 0:
        return x
    x = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=3)
    x = torch.cat([x[..., -pad:, :], x, x[..., :pad, :]], dim=2)
    return x


class PeriodicConv2D(nn.Module):
    """Stride-1 2D convolution with periodic boundary conditions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so the output keeps the input size")
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(periodic_padding(x, self.padding))


class DiffusionModel2D(nn.Module):
    """Local x0-predictor: periodic conv -> +time embedding -> BN -> tanh -> per-pixel linear -> tanh.

    Args:
        channels: number of hidden channels of the convolution.
        kernel_size: receptive field (odd) of the convolution.
    """

    def __init__(self, channels: int = 4, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.conv1 = PeriodicConv2D(1, channels, kernel_size)
        self.bn1 = nn.BatchNorm2d(channels)
        self.linear = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, H, W) noisy images x_t.
            t: (batch,) integer diffusion steps.

        Returns:
            (batch, 1, H, W) prediction of the clean image x_0.
        """
        t_emb = timestep_embedding(t, dim=max(16, self.channels))[:, : self.channels]
        x = self.conv1(x) + t_emb[:, :, None, None]  # (batch, channels, H, W)
        x = torch.tanh(self.bn1(x))
        x = self.linear(x.permute(0, 2, 3, 1))  # per-pixel linear: (batch, H, W, 1)
        return torch.tanh(x.permute(0, 3, 1, 2))


def load_model(checkpoint: str, channels: int, kernel_size: int, device: torch.device) -> DiffusionModel2D:
    """Build a model and load weights from ``checkpoint`` (in eval mode)."""
    model = DiffusionModel2D(channels, kernel_size).to(device)
    state = torch.load(checkpoint, map_location=device)
    # Checkpoints from earlier versions contain an unused `time_emb` layer.
    state = {k: v for k, v in state.items() if not k.startswith("time_emb.")}
    model.load_state_dict(state)
    model.eval()
    return model
