from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from release_tools.scheduled_render import render_scheduled, step_gains


class Mixer:
    def __init__(self):
        self.strengths = {"candlelit": 1.0}
        self.adapters = dict(self.strengths)

    @contextmanager
    def scales(self, values):
        previous = self.strengths
        self.strengths = values
        try:
            yield
        finally:
            self.strengths = previous


class Runtime:
    def __init__(self, *, strength=1.0, fail=False):
        self.mixer = Mixer()
        self.original_strengths = {"candlelit": strength}
        self.mixer.strengths = self.original_strengths.copy()
        self.seen = []
        self.fail = fail

    def scheduler(self, steps):
        return SimpleNamespace(sigmas=torch.tensor([1.0, 0.5, 0.0]), timesteps=(1.0, 0.5))

    def encode(self, prompt):
        return prompt

    def noise(self, seed, width, height):
        return 0.0

    def predict(self, latent, timestep, embedding):
        strength = self.mixer.strengths["candlelit"]
        self.seen.append((timestep, strength))
        if self.fail and timestep == 0.5:
            raise RuntimeError("native forward failed")
        return strength

    def step(self, scheduler, velocity, timestep, latent):
        assert self.mixer.strengths == self.original_strengths
        return latent + velocity

    def decode(self, latent):
        return latent


def test_release_render_uses_native_interval_order_and_records_gains():
    runtime = Runtime()
    image, schedule = render_scheduled(runtime, "neutral", 7, 512, 512, 2, [1.5, 0.5])
    assert runtime.seen == [(1.0, 1.5), (0.5, 0.5)]
    assert image == 2.0
    assert schedule == dict(sigmas=[1.0, 0.5, 0.0], effective_strengths=[1.5, 0.5])
    assert runtime.mixer.strengths == {"candlelit": 1.0}


def test_release_render_restores_mixer_after_forward_failure():
    runtime = Runtime(fail=True)
    with pytest.raises(RuntimeError, match="native forward failed"):
        render_scheduled(runtime, "neutral", 7, 512, 512, 2, [1.5, 0.5])
    assert runtime.mixer.strengths == {"candlelit": 1.0}


@pytest.mark.parametrize("gains", [None, [1.0], [False, 1.0], [float("inf"), 1.0],
                                   [10**400, 1.0], [-1.0, 1.0], [5.01, 0.0]])
def test_step_gain_validation(gains):
    with pytest.raises(ValueError):
        step_gains(gains, 2, 1.0)


def test_zero_strength_replays_off_across_scheduled_steps():
    runtime = Runtime(strength=0.0)
    image, schedule = render_scheduled(runtime, "neutral", 7, 512, 512, 2, [1.5, 0.5])
    assert runtime.seen == [(1.0, 0.0), (0.5, 0.0)]
    assert image == 0.0
    assert schedule["effective_strengths"] == [0.0, 0.0]
