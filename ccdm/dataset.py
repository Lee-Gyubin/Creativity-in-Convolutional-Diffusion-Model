import random

import torch
from torch.utils.data import Dataset


class BinaryImageDataset(Dataset):
    """Two-image training set: an all-black (0) and an all-white (1) image.

    Each item is drawn uniformly at random from the two images, so ``size``
    only sets the number of items per epoch.
    """

    def __init__(self, size: int = 500, image_size: int = 16):
        self.size = size
        self.images = [
            torch.zeros((1, image_size, image_size)),
            torch.ones((1, image_size, image_size)),
        ]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        return random.choice(self.images)
