from typing import Tuple

import torch
import torch.nn.functional as F

from ccdm.model import periodic_padding


def local_consistency_ratio(binary_images: torch.Tensor, kernel_size: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fraction of pixels that agree with the majority colour of their neighbourhood.

    For every pixel, the majority value (0 or 1) of its ``kernel_size x kernel_size``
    periodic neighbourhood (including the pixel itself) is computed; the pixel is
    "locally consistent" if it equals that majority. Uniform images score 1.0,
    i.i.d. noise scores noticeably lower.

    Args:
        binary_images: (batch, 1, H, W) tensor with values in {0, 1}.
        kernel_size: odd neighbourhood size.

    Returns:
        ratio: (batch, 1) per-image consistency ratio.
        std: scalar standard deviation of the ratio across the batch.
    """
    binary_images = binary_images.float()
    padded = periodic_padding(binary_images, kernel_size // 2)
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=binary_images.device)
    neighbour_sum = F.conv2d(padded, kernel)

    majority = (neighbour_sum >= kernel_size**2 / 2).float()
    agreement = (binary_images == majority).float()

    ratio = agreement.mean(dim=(2, 3))
    return ratio, ratio.std()
