"""Target patterns and reward functions used for FK steering.

All images are binary: 1 is white, 0 is black. Every reward function takes a
(batch, 1, H, W) tensor of x0 predictions plus a target and returns a
(batch,) tensor where larger is better.
"""

from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F

from ccdm.model import periodic_padding


# --------------------------------------------------------------------------- targets

def white_image(size: int = 16) -> torch.Tensor:
    return torch.ones((size, size))


def black_image(size: int = 16) -> torch.Tensor:
    return torch.zeros((size, size))


def checkerboard(size: int = 16) -> torch.Tensor:
    """Alternating black/white pixels."""
    idx = torch.arange(size)
    return ((idx[:, None] + idx[None, :]) % 2).float()


def top_white_image(size: int = 16) -> torch.Tensor:
    """Top half white, bottom half black."""
    image = torch.zeros((size, size))
    image[: size // 2] = 1
    return image


def top_black_image(size: int = 16) -> torch.Tensor:
    """Top half black, bottom half white."""
    return 1 - top_white_image(size)


def quadrant_image(size: int = 16) -> torch.Tensor:
    """Top-right and bottom-left quadrants white, the other two black."""
    image = torch.zeros((size, size))
    half = size // 2
    image[:half, half:] = 1
    image[half:, :half] = 1
    return image


def single_block_image(size: int = 16) -> torch.Tensor:
    """Example image with a single 3x3 white block that wraps around the right edge."""
    image = torch.zeros((size, size))
    image[8:11, size - 2 :] = 1
    image[8:11, :1] = 1
    return image


# --------------------------------------------------------------------------- rewards

def binarize(images: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (images > threshold).float()


def negative_l1_reward(images: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative number of pixels in which the binarized image differs from ``target``."""
    target = target.to(images.device)
    return -(binarize(images) - target).abs().sum(dim=(1, 2, 3))


def single_block_reward(images: torch.Tensor, target: torch.Tensor = None, scale: float = 100.0) -> torch.Tensor:
    """``scale`` if the binarized image contains exactly one fully white 3x3 block
    (with periodic boundaries), otherwise 0. ``target`` is ignored."""
    padded = periodic_padding(binarize(images), pad=1)
    kernel = torch.ones((1, 1, 3, 3), device=images.device)
    block_count = (F.conv2d(padded, kernel) == 9).flatten(1).sum(dim=1)
    return (block_count == 1).float() * scale


def projection_reward(images: torch.Tensor, target: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """(Experimental) negative mismatch between the row/column occupancy of the image and a target.

    Args:
        images: (batch, 1, H, W) tensor.
        target: ``(target_cols, target_rows)``, each of shape (W,) / (H,) with values in
            {0, 1}; entry ``j`` says whether column/row ``j`` should contain a white pixel.
    """
    binary = binarize(images)[:, 0]
    cols = (binary.sum(dim=1) > 0).float()  # (batch, W)
    rows = (binary.sum(dim=2) > 0).float()  # (batch, H)
    target_cols, target_rows = (x.to(images.device) for x in target)
    return -((cols - target_cols).abs().sum(dim=-1) + (rows - target_rows).abs().sum(dim=-1))


RewardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# name -> (target image, reward function)
REWARDS: Dict[str, Tuple[Callable[[int], torch.Tensor], RewardFn]] = {
    "checkerboard": (checkerboard, negative_l1_reward),
    "top-white": (top_white_image, negative_l1_reward),
    "top-black": (top_black_image, negative_l1_reward),
    "black": (black_image, negative_l1_reward),
    "white": (white_image, negative_l1_reward),
    "quadrant": (quadrant_image, negative_l1_reward),
    "block": (single_block_image, single_block_reward),
}


def get_reward(name: str, size: int = 16) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Return ``(target_image, reward_fn)`` where ``reward_fn(x0_preds) -> (batch,)`` rewards."""
    if name not in REWARDS:
        raise ValueError(f"Unknown reward '{name}'. Choose from: {', '.join(REWARDS)}")
    make_target, reward_fn = REWARDS[name]
    target = make_target(size)
    return target, lambda images: reward_fn(images, target)
