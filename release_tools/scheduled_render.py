"""Render an Anima RAW adapter across native FlowMatchEuler intervals."""
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


def _adapter_strength(runtime):
    nominal = runtime.mixer.strengths
    if len(nominal) != 1 or not set(nominal) <= runtime.mixer.adapters.keys():
        raise ValueError("Scheduled release rendering requires one attached particle adapter")
    return next(iter(nominal.items()))


def _euler_grid(runtime, steps):
    scheduler = runtime.scheduler(steps)
    sigmas = scheduler.sigmas.tolist()
    if len(scheduler.timesteps) != steps or len(sigmas) != steps + 1:
        raise ValueError("Native scheduler geometry differs from the requested intervals")
    if any(not isfinite(sigma) for sigma in sigmas) or any(
        left <= right for left, right in zip(sigmas, sigmas[1:])
    ):
        raise ValueError("Expected finite, decreasing native sigmas")
    return scheduler, sigmas


@torch.no_grad()
def render_scheduled(runtime, prompt, seed, width, height, steps, gains):
    """Apply fixed gains through Anima's scoped mixer and original solver methods."""
    name, strength = _adapter_strength(runtime)
    gains = step_gains(list(gains), steps, strength)
    scheduler, sigmas = _euler_grid(runtime, steps)
    embedding = runtime.encode(prompt)
    latent = runtime.noise(seed, width, height)
    effective = [strength * gain for gain in gains]
    for timestep, scale in zip(scheduler.timesteps, effective, strict=True):
        with runtime.mixer.scales({name: scale}):
            velocity = runtime.predict(latent, timestep, embedding)
        latent = runtime.step(scheduler, velocity, timestep, latent)
    return runtime.decode(latent), dict(sigmas=sigmas, effective_strengths=effective)


class LatentGatePlan:
    """Reproducible gate inputs for Anima's mean pre-denoiser latent channels."""

    def __init__(self, *, method, parameter, direction=None, gate_seed=None, clabs=None):
        from particle_sliders.adaptive_steering import CLABSParameters

        self.method = method
        self.parameter = parameter
        self.gate_seed = gate_seed
        self.clabs = CLABSParameters(**clabs) if isinstance(clabs, dict) else clabs
        if direction is None and method == "random":
            self.direction = None
        else:
            if direction is None:
                raise ValueError("An activation-dependent gate requires a calibrated direction")
            probe = torch.as_tensor(direction, dtype=torch.float32).detach().clone()
            if (probe.ndim != 1 or not torch.isfinite(probe).all()
                    or not torch.isclose(probe.norm(), probe.new_tensor(1.0))):
                raise ValueError("Provide a finite unit direction in Anima latent channel space")
            self.direction = probe
        self.new_gate()

    def new_gate(self):
        from particle_sliders.adaptive_steering import ActivationGate

        return ActivationGate(self.method, self.parameter, seed=self.gate_seed, clabs=self.clabs)


@torch.no_grad()
def render_gated(runtime, prompt, seed, width, height, steps, plan):
    """Gate one attached RAW adapter from the current latent at each native interval."""
    from particle_sliders.adaptive_steering import activation_projection

    name, strength = _adapter_strength(runtime)
    if not 0 <= strength <= 5:
        raise ValueError("Strength must be between 0 and 5")
    if plan.direction is not None and plan.direction.numel() != runtime.channels:
        raise ValueError("Direction must match Anima's native latent channel count")
    gate = plan.new_gate()
    scheduler, sigmas = _euler_grid(runtime, steps)
    embedding = runtime.encode(prompt)
    latent = runtime.noise(seed, width, height)
    direction = None if plan.direction is None else plan.direction.to(latent.device)
    trace = []
    effective = []
    for index, timestep in enumerate(scheduler.timesteps):
        scale = 0.0
        if strength:
            score = None
            if direction is not None:
                # Use the same latent cast as TurboRuntime.predict, then pool in FP32.
                channels = latent.to(runtime.dtype).mean(dim=(0, 2, 3, 4), dtype=torch.float32)
                score = float(activation_projection(channels, direction, strength=strength))
            decision = gate.step(score)
            scale = strength if decision.fired else 0.0
            trace.append(dict(interval=index, projection=score, probability=decision.probability,
                              fired=decision.fired, effective_strength=scale))
        with runtime.mixer.scales({name: scale}):
            velocity = runtime.predict(latent, timestep, embedding)
        latent = runtime.step(scheduler, velocity, timestep, latent)
        effective.append(scale)
    usage = gate.usage
    dose = sum(scale * (sigmas[index] - sigmas[index + 1])
               for index, scale in enumerate(effective))
    return runtime.decode(latent), dict(
        sigmas=sigmas, effective_strengths=effective, activation_gate_trace=trace,
        gate_firings=usage.fired, gate_eligible_intervals=usage.eligible,
        gate_firing_rate=usage.rate if usage.eligible else None, sigma_weighted_strength=dose,
    )
