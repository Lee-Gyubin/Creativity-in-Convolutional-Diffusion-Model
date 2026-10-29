from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def _to_numpy(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().reshape(image.shape[-2], image.shape[-1]).numpy()


def save_pixel_image(image: torch.Tensor, path: str) -> None:
    """Save a single (…, H, W) image with values in [0, 1] and a pixel grid overlay."""
    img = _to_numpy(image)
    h, w = img.shape
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_denoising_sequence(frames: List[torch.Tensor], path: str, num_frames: int = 10) -> None:
    """Save ``num_frames`` evenly spaced frames of a reverse trajectory in one row.

    ``frames[i]`` is the image after sampling step ``i``, i.e. x_{T-1-i}.
    """
    n_steps = len(frames)
    num_frames = min(num_frames, n_steps)
    selected = np.linspace(0, n_steps - 1, num_frames, dtype=int)
    fig, axes = plt.subplots(1, num_frames, figsize=(2 * num_frames, 2), squeeze=False)
    for ax, step in zip(axes[0], selected):
        ax.imshow(_to_numpy(frames[step]), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"t = {n_steps - 1 - step}")
        ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_loss_curve(losses: Sequence[float], path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, label="Training loss", color="blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid()
    fig.savefig(path)
    plt.close(fig)
