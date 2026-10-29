import random

import numpy as np
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_name(n_steps: int, kernel_size: int, channels: int) -> str:
    """Name shared by a checkpoint and the outputs generated from it."""
    return f"step{n_steps}_perioconv{kernel_size}_channel{channels}"
