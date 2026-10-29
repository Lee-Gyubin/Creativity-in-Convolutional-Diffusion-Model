import torch

from ccdm.fk_steering import FKSteering, PotentialType
from ccdm.metrics import local_consistency_ratio
from ccdm.model import DiffusionModel2D, periodic_padding
from ccdm.rewards import REWARDS, get_reward, single_block_reward


def test_periodic_padding_wraps_around():
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    padded = periodic_padding(x, 1)
    assert padded.shape == (1, 1, 6, 6)
    assert padded[0, 0, 0, 0] == x[0, 0, -1, -1]
    assert padded[0, 0, -1, -1] == x[0, 0, 0, 0]
    assert torch.equal(padded[0, 0, 1:-1, 1:-1], x[0, 0])


def test_model_preserves_shape():
    model = DiffusionModel2D(channels=4, kernel_size=3)
    out = model(torch.randn(4, 1, 16, 16), torch.tensor([0, 10, 50, 99]))
    assert out.shape == (4, 1, 16, 16)


def test_single_block_reward_detects_wrapped_block():
    image = torch.zeros((2, 1, 16, 16))
    # A 3x3 block wrapping around the right edge.
    image[0, :, 8:11, 14:] = 1
    image[0, :, 8:11, :1] = 1
    # A 3x3 block wrapping around the corner: one block, not four.
    image[1, :, 14:, 14:] = 1
    image[1, :, :1, 14:] = 1
    image[1, :, 14:, :1] = 1
    image[1, :, :1, :1] = 1
    assert torch.equal(single_block_reward(image), torch.tensor([100.0, 100.0]))
    assert single_block_reward(torch.ones((1, 1, 16, 16))).item() == 0.0


def test_l1_reward_is_zero_on_target():
    for name in REWARDS:
        if name == "block":
            continue
        target, reward_fn = get_reward(name)
        assert reward_fn(target[None, None]).item() == 0.0
        assert reward_fn((1 - target)[None, None]).item() == -target.numel()


def test_local_consistency():
    uniform = torch.zeros((3, 1, 16, 16))
    ratio, _ = local_consistency_ratio(uniform)
    assert torch.all(ratio == 1.0)

    # In 1-pixel-wide vertical stripes every pixel disagrees with its 3x3 majority.
    stripes = (torch.arange(16) % 2).float().expand(16, 16)
    ratio, _ = local_consistency_ratio(stripes[None, None])
    assert ratio.item() == 0.0


def test_fk_steering_resamples_toward_reward():
    torch.manual_seed(0)
    target, reward_fn = get_reward("white")
    particles = torch.cat([torch.ones(1, 1, 16, 16), torch.zeros(7, 1, 16, 16)])
    fk = FKSteering(
        potential_type=PotentialType.DIFF,
        lmbda=10.0,
        num_particles=8,
        adaptive_resampling=False,
        resample_frequency=1,
        resampling_t_start=0,
        resampling_t_end=9,
        time_steps=10,
        reward_fn=reward_fn,
    )
    latents, images = fk.resample(sampling_idx=0, latents=particles, x0_preds=particles)
    assert torch.all(images == 1) and torch.all(latents == 1)
