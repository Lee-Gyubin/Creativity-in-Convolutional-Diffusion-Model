"""Feynman-Kac (FK) steering: particle-based inference-time steering of diffusion models.

Adapted from the reference implementation of Singhal et al., "A General Framework
for Inference-time Scaling and Steering of Diffusion Models" (2025),
https://github.com/zacharyhorvitz/Fk-Diffusion-Steering
"""

import math
from enum import Enum
from typing import Callable, Tuple

import torch

LOG_W_MIN, LOG_W_MAX = math.log(1e-10), math.log(1e10)


class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"


class FKSteering:
    """Maintains a population of particles and resamples them by reward-based potentials.

    Create one instance per sampling run and call :meth:`resample` at every step of the
    reverse process.

    Args:
        potential_type: how rewards are turned into potentials (see :class:`PotentialType`).
        lmbda: inverse temperature; larger values steer more aggressively.
        num_particles: number of particles in the population.
        adaptive_resampling: only resample when the effective sample size drops below
            half the number of particles.
        resample_frequency: resample every ``resample_frequency`` sampling steps.
        resampling_t_start: first sampling step (0-indexed) at which to resample.
        resampling_t_end: last sampling step (0-indexed) at which to resample.
        time_steps: total number of sampling steps.
        reward_fn: maps decoded x0 predictions (num_particles, ...) to (num_particles,) rewards.
        reward_min_value: initial reward of every particle (relevant for the MAX potential).
        latent_to_decode_fn: decodes latents to images (identity for pixel-space models).
        device: device holding the particle state.
    """

    def __init__(
        self,
        *,
        potential_type: PotentialType,
        lmbda: float,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        reward_min_value: float = 0.0,
        latent_to_decode_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.potential_type = PotentialType(potential_type)
        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.resampling_steps = set(range(resampling_t_start, resampling_t_end + 1, resample_frequency))
        self.time_steps = time_steps
        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn

        self.population_rs = torch.full((num_particles,), reward_min_value, device=device)
        self.log_product_of_potentials = torch.zeros(num_particles, device=device)

    def resample(
        self, *, sampling_idx: int, latents: torch.Tensor, x0_preds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resample particles if ``sampling_idx`` is a resampling step.

        Args:
            sampling_idx: index of the current sampling step, counting up from 0
                (i.e. ``time_steps - 1`` is the last, least noisy step).
            latents: current noisy latents x_t, (num_particles, ...).
            x0_preds: model predictions of x_0 for ``latents``.

        Returns:
            ``(latents, x0_preds)``, resampled if a resampling step was taken.
        """
        if sampling_idx not in self.resampling_steps:
            return latents, x0_preds

        population_images = self.latent_to_decode_fn(x0_preds)
        rs_candidates = self.reward_fn(population_images)
        is_last_step = sampling_idx == self.time_steps - 1

        # Importance weights, computed in log space so the running product cannot underflow.
        if self.potential_type == PotentialType.MAX:
            log_w = self.lmbda * torch.max(rs_candidates, self.population_rs)
        elif self.potential_type == PotentialType.ADD:
            rs_candidates = rs_candidates + self.population_rs
            log_w = self.lmbda * rs_candidates
        else:  # DIFF
            log_w = self.lmbda * (rs_candidates - self.population_rs)

        if is_last_step and self.potential_type in (PotentialType.MAX, PotentialType.ADD):
            log_w = self.lmbda * rs_candidates - self.log_product_of_potentials

        log_w = torch.clamp(log_w, LOG_W_MIN, LOG_W_MAX)  # i.e. w in [1e-10, 1e10]
        w = torch.exp(log_w)
        w[torch.isnan(w)] = 0.0

        if self.adaptive_resampling or is_last_step:
            normalized_w = w / w.sum()
            ess = 1.0 / normalized_w.pow(2).sum()
            if ess >= 0.5 * self.num_particles:
                self.population_rs = rs_candidates
                return latents, population_images

        indices = torch.multinomial(w, num_samples=self.num_particles, replacement=True)
        self.population_rs = rs_candidates[indices]
        self.log_product_of_potentials = self.log_product_of_potentials[indices] + log_w[indices]
        return latents[indices], population_images[indices]
