"""Train the periodic convolutional denoiser on the all-black / all-white dataset."""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ccdm.dataset import BinaryImageDataset
from ccdm.diffusion import GaussianDiffusion
from ccdm.model import DiffusionModel2D
from ccdm.plotting import save_loss_curve
from ccdm.utils import get_device, run_name, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-steps", type=int, default=100, help="number of diffusion steps")
    p.add_argument("--kernel-size", type=int, default=3, help="receptive field of the conv (odd)")
    p.add_argument("--channels", type=int, default=4, help="hidden channels")
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--dataset-size", type=int, default=500, help="items per epoch")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    name = run_name(args.n_steps, args.kernel_size, args.channels)

    diffusion = GaussianDiffusion(args.n_steps, device)
    dataset = BinaryImageDataset(size=args.dataset_size, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = DiffusionModel2D(args.channels, args.kernel_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training {name} on {device}")
    loss_history = []
    for epoch in range(args.epochs):
        total_loss = 0.0
        for x0 in dataloader:
            x0 = x0.to(device)
            t = torch.randint(0, args.n_steps, (x0.shape[0],), device=device)
            x_t = diffusion.q_sample(x0, t, torch.randn_like(x0))

            # The network predicts the clean image x_0 (not the noise).
            loss = F.mse_loss(model(x_t, t), x0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history.append(total_loss / len(dataloader))
        if (epoch + 1) % args.log_every == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}] loss: {loss_history[-1]:.6f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{name}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved: {ckpt_path}")

    plot_dir = os.path.join(args.output_dir, "training_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"loss_{name}.png")
    save_loss_curve(loss_history, plot_path)
    print(f"Loss curve saved: {plot_path}")


if __name__ == "__main__":
    main()
