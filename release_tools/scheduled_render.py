"""Render with native per-interval gains over the active particle strength."""
from math import isfinite
from sys import float_info

import torch


def step_gains(values, steps, strength):
    """Validate every effective strength before loading the native model."""
    if not isinstance(values, list) or len(values) != steps:
        raise ValueError("Provide one step gain per native denoising interval")
    if not 0 <= strength <= 5:
        raise ValueError("Strength must be between 0 and 5")
    if any(type(value) not in (int, float) or not 0 <= value <= float_info.max
           for value in values):
        raise ValueError("Step gains must be finite, nonnegative numbers")
    if any(strength * value > 5 for value in values):
        raise ValueError("Effective strengths must be between 0 and 5")
    return tuple(float(value) for value in values)


@torch.no_grad()
def render_scheduled(runtime, prompt, seed, width, height, steps, gains):
    """Apply gains through Anima's scoped mixer and original solver methods."""
    nominal = runtime.mixer.strengths.copy()
    if len(nominal) != 1 or not set(nominal) <= runtime.mixer.adapters.keys():
        raise ValueError("Scheduled release rendering requires one attached particle adapter")
    gains = step_gains(list(gains), steps, next(iter(nominal.values())))
    scheduler = runtime.scheduler(steps)
    sigmas = scheduler.sigmas.tolist()
    if len(scheduler.timesteps) != steps or len(sigmas) != steps + 1:
        raise ValueError("Native scheduler geometry differs from the step gains")
    if any(not isfinite(sigma) for sigma in sigmas) or any(
        left <= right for left, right in zip(sigmas, sigmas[1:])
    ):
        raise ValueError("Expected finite, decreasing native sigmas")
    embedding = runtime.encode(prompt)
    latent = runtime.noise(seed, width, height)
    (name, strength), = nominal.items()
    effective = [strength * gain for gain in gains]
    for timestep, scale in zip(scheduler.timesteps, effective, strict=True):
        with runtime.mixer.scales({name: scale}):
            velocity = runtime.predict(latent, timestep, embedding)
        latent = runtime.step(scheduler, velocity, timestep, latent)
    return runtime.decode(latent), dict(sigmas=sigmas, effective_strengths=effective)
