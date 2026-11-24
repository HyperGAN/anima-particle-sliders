"""Convert two frozen teacher steps into an average edit velocity."""

import torch


def two_step_target(runtime, steps, position, latent, neutral_embedding, positive_embedding,
                    neutral_velocity, positive_velocity, weight):
    neutral_scheduler, positive_scheduler = runtime.scheduler(steps), runtime.scheduler(steps)
    neutral_scheduler.set_begin_index(position)
    positive_scheduler.set_begin_index(position)
    timestep, next_timestep = neutral_scheduler.timesteps[position:position + 2]
    first, second, end = neutral_scheduler.sigmas[position:position + 3].tolist()
    span = (first - second) / (first - end)

    positive_velocity = neutral_velocity + (positive_velocity - neutral_velocity)
    next_neutral = runtime.step(neutral_scheduler, neutral_velocity, timestep, latent)
    next_positive = runtime.step(positive_scheduler, positive_velocity, timestep, latent)
    future = (runtime.predict(next_positive, next_timestep, positive_embedding)
              - runtime.predict(next_neutral, next_timestep, neutral_embedding))
    immediate = positive_velocity - neutral_velocity
    weighted = immediate.lerp(future, 1 - span) * weight
    norm = weighted.norm()
    if not torch.isfinite(norm) or norm.item() <= 1e-10:
        raise ValueError("Degenerate two-step teacher contrast")
    return neutral_velocity + weighted * (immediate.norm() / norm)
