"""Sample from a trained model, optionally with Feynman-Kac steering toward a target pattern.

Plain DDPM sampling:          python generate.py
FK steering toward a reward:  python generate.py --reward top-black
"""

import argparse
import os

import torch
from tqdm import tqdm

from ccdm.diffusion import GaussianDiffusion
from ccdm.fk_steering import FKSteering, PotentialType
from ccdm.metrics import local_consistency_ratio
from ccdm.model import load_model
from ccdm.plotting import save_denoising_sequence, save_pixel_image
from ccdm.rewards import REWARDS, binarize, get_reward
from ccdm.utils import get_device, run_name, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-steps", type=int, default=100, help="number of diffusion steps")
    p.add_argument("--kernel-size", type=int, default=3, help="receptive field of the conv (odd)")
    p.add_argument("--channels", type=int, default=4, help="hidden channels")
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--checkpoint", default=None, help="defaults to checkpoints/<run name>.pth")
    p.add_argument("--num-runs", type=int, default=10, help="number of independent sampling runs")
    p.add_argument("--batch-size", type=int, default=100, help="samples per run (= particles with FK steering)")
    p.add_argument("--threshold", type=float, default=0.5, help="binarization threshold for the final samples")
    p.add_argument("--consistency-kernel", type=int, default=3, help="neighbourhood size of the local consistency metric")
    p.add_argument("--num-frames", type=int, default=10, help="frames shown in each denoising-sequence figure")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs")

    fk = p.add_argument_group("FK steering (enabled by --reward)")
    fk.add_argument("--reward", choices=sorted(REWARDS), default=None, help="target pattern to steer toward")
    fk.add_argument("--lmbda", type=float, default=10.0, help="steering strength (inverse temperature)")
    fk.add_argument("--potential", choices=[pt.value for pt in PotentialType], default="diff")
    fk.add_argument("--adaptive-resampling", action="store_true", help="resample only when the ESS is low")
    fk.add_argument("--resample-frequency", type=int, default=1)
    return p.parse_args()


@torch.no_grad()
def sample(model, diffusion, x_t, fk=None):
    """Run the reverse process from x_T and return (x_0, frames of the first sample)."""
    frames = []
    for i, step in enumerate(tqdm(reversed(range(diffusion.n_steps)), total=diffusion.n_steps, leave=False)):
        t = torch.full((x_t.shape[0],), step, device=x_t.device, dtype=torch.long)
        x0_pred = model(x_t, t).clamp(-1.0, 1.0)
        if fk is not None:
            x_t, x0_pred = fk.resample(sampling_idx=i, latents=x_t, x0_preds=x0_pred)
        x_t = diffusion.p_sample(x_t, x0_pred, t)
        frames.append(x_t[0])
    return x_t, frames


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    name = run_name(args.n_steps, args.kernel_size, args.channels)
    checkpoint = args.checkpoint or os.path.join("checkpoints", f"{name}.pth")

    model = load_model(checkpoint, args.channels, args.kernel_size, device)
    diffusion = GaussianDiffusion(args.n_steps, device)

    out_dir = os.path.join(args.output_dir, f"fk_{args.reward}_{name}" if args.reward else name)
    seq_dir = os.path.join(out_dir, "denoising_sequences")
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    reward_fn = None
    if args.reward:
        target, reward_fn = get_reward(args.reward, args.image_size)
        save_pixel_image(target, os.path.join(out_dir, f"target_{args.reward}.png"))

    generated, initial_noise, rewards = [], [], []
    for run in range(args.num_runs):
        fk = None
        if args.reward:
            fk = FKSteering(
                potential_type=PotentialType(args.potential),
                lmbda=args.lmbda,
                num_particles=args.batch_size,
                adaptive_resampling=args.adaptive_resampling,
                resample_frequency=args.resample_frequency,
                resampling_t_start=0,
                resampling_t_end=args.n_steps - 1,
                time_steps=args.n_steps,
                reward_fn=reward_fn,
                device=device,
            )

        x_T = torch.randn(args.batch_size, 1, args.image_size, args.image_size, device=device)
        x_0, frames = sample(model, diffusion, x_T, fk)

        generated.append(binarize(x_0, args.threshold))
        initial_noise.append(binarize(x_T, args.threshold))
        if reward_fn is not None:
            rewards.append(reward_fn(x_0))

        save_denoising_sequence(frames, os.path.join(seq_dir, f"sample_{run + 1}.png"), args.num_frames)
        save_pixel_image(generated[-1][0], os.path.join(sample_dir, f"sample_{run + 1}.png"))
        print(f"[{run + 1}/{args.num_runs}] saved to {out_dir}")

    gen_ratio, gen_std = local_consistency_ratio(torch.cat(generated), args.consistency_kernel)
    noise_ratio, noise_std = local_consistency_ratio(torch.cat(initial_noise), args.consistency_kernel)
    print(f"Local consistency over {args.num_runs * args.batch_size} samples "
          f"({args.consistency_kernel}x{args.consistency_kernel} neighbourhood):")
    print(f"  initial noise: {noise_ratio.mean():.4f} ± {noise_std:.4f}")
    print(f"  generated:     {gen_ratio.mean():.4f} ± {gen_std:.4f}")
    if rewards:
        rewards = torch.cat(rewards).float()
        print(f"Reward '{args.reward}': {rewards.mean():.4f} ± {rewards.std():.4f}")


if __name__ == "__main__":
    main()
