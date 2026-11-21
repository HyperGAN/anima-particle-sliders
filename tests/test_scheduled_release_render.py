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


class LatentRuntime(Runtime):
    dtype = torch.bfloat16
    channels = 2

    def scheduler(self, steps):
        return SimpleNamespace(sigmas=torch.tensor([1.0, 0.25, 0.0]), timesteps=(1.0, 0.25))

    def noise(self, seed, width, height):
        return torch.zeros(1, self.channels, 1, 2, 2)

    def step(self, scheduler, velocity, timestep, latent):
        step = super().step(scheduler, velocity, timestep, latent)
        return step + 0.75


def test_cfp_reads_current_pre_forward_latent_and_logs_dose():
    from release_tools.scheduled_render import LatentGatePlan, render_gated

    runtime = LatentRuntime()
    plan = LatentGatePlan(method="cfp", parameter=0.5, direction=[1.0, 0.0])
    image, schedule = render_gated(runtime, "neutral", 7, 512, 512, 2, plan)
    assert runtime.seen == [(1.0, 1.0), (0.25, 0.0)]
    assert schedule['activation_gate_trace'] == [
        dict(interval=0, projection=0.0, probability=1.0, fired=True, effective_strength=1.0),
        dict(interval=1, projection=1.75, probability=0.0, fired=False, effective_strength=0.0),
    ]
    assert schedule['gate_firings'] == 1
    assert schedule['gate_eligible_intervals'] == 2
    assert schedule['gate_firing_rate'] == 0.5
    assert schedule['sigma_weighted_strength'] == 0.75
    expected, _ = render_scheduled(LatentRuntime(), "neutral", 7, 512, 512, 2, [1, 0])
    torch.testing.assert_close(image, expected, atol=0, rtol=0)
    assert runtime.mixer.strengths == {"candlelit": 1.0}


def test_activation_gate_off_preserves_old_path_without_consuming_gate():
    from release_tools.scheduled_render import LatentGatePlan, render_gated

    runtime = LatentRuntime(strength=0.0)
    plan = LatentGatePlan(method="cfp", parameter=1.0, direction=[1, 0])
    image, schedule = render_gated(runtime, "neutral", 7, 512, 512, 2, plan)
    replay, _ = render_scheduled(runtime, "neutral", 7, 512, 512, 2, [1, 1])
    torch.testing.assert_close(image, replay, atol=0, rtol=0)
    assert schedule['activation_gate_trace'] == []
    assert schedule['gate_firing_rate'] is None
    assert schedule['effective_strengths'] == [0.0, 0.0]


def test_activation_gate_restores_mixer_after_native_error(monkeypatch):
    from release_tools.scheduled_render import LatentGatePlan, render_gated

    runtime = LatentRuntime()
    plan = LatentGatePlan(method="random", parameter=0, gate_seed=41)

    def fail(*args):
        assert runtime.mixer.strengths == {'candlelit': 0.0}
        raise RuntimeError("native forward failed")

    monkeypatch.setattr(runtime, 'predict', fail)
    with pytest.raises(RuntimeError, match="native forward failed"):
        render_gated(runtime, "neutral", 7, 512, 512, 2, plan)
    assert runtime.mixer.strengths == {"candlelit": 1.0}


def test_real_euler_and_cosmos_route_replay_activation_gate_exactly(monkeypatch):
    from diffusers import CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler

    from lumen_studio.backends.anima import TurboRuntime
    from lumen_studio.particles import Mixer, ParticleAdapter
    from release_tools.scheduled_render import LatentGatePlan, render_gated

    with torch.random.fork_rng():
        torch.manual_seed(9)
        runtime = TurboRuntime.__new__(TurboRuntime)
        runtime.transformer = CosmosTransformer3DModel(
            in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=32,
            num_layers=1, mlp_ratio=2, text_embed_dim=8, adaln_lora_dim=4, max_size=(2, 4, 4),
            patch_size=(1, 1, 1), extra_pos_embed_type=None,
        ).to(torch.bfloat16).eval().requires_grad_(False)
        runtime.mixer = Mixer(runtime.transformer)
        adapter = ParticleAdapter(runtime.transformer).eval().requires_grad_(False)
        for branch in adapter.branches:
            torch.nn.init.normal_(branch.up.weight, std=0.1)
    runtime.device, runtime.dtype = torch.device('cpu'), torch.bfloat16
    runtime.scale, runtime.channels = 1, 4
    runtime.pipe = SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0))
    runtime.mixer.add('candlelit', adapter)
    runtime.mixer.strengths = {'candlelit': 1.0}
    runtime.encode = lambda _: torch.ones(1, 2, 8)
    runtime.noise = lambda seed, *_: torch.randn(1, 4, 1, 2, 2,
                                                generator=torch.Generator().manual_seed(seed))
    runtime.decode = lambda latent: latent
    seen_steps = []
    native_step = runtime.step

    def step(scheduler, velocity, timestep, latent):
        seen_steps.append((scheduler.begin_index, scheduler.step_index))
        return native_step(scheduler, velocity, timestep, latent)

    monkeypatch.setattr(runtime, 'step', step)
    args = ('neutral', 7, 512, 512, 8)
    plan = LatentGatePlan(method='clabs', parameter=0.5, direction=[1, 0, 0, 0], gate_seed=9,
                          clabs=dict(mean=0.0, std=1.0, epsilon=1e-6, setpoint=0.5,
                                     proportional_gain=0.1, integral_gain=0.1, beta=0.5))
    adapted, trace = render_gated(runtime, *args, plan)
    assert seen_steps == [(0, None), *((0, index) for index in range(1, 8))]
    assert 0 < trace['gate_firings'] < 8
    replay, _ = render_scheduled(runtime, *args, trace['effective_strengths'])
    torch.testing.assert_close(replay, adapted, rtol=0.0, atol=0.0)
    off_plan = LatentGatePlan(method='random', parameter=0.0, gate_seed=9)
    inactive, inactive_trace = render_gated(runtime, *args, off_plan)
    assert inactive_trace['gate_firings'] == 0
    runtime.mixer.strengths = {'candlelit': 0.0}
    torch.testing.assert_close(runtime.render(*args), inactive, rtol=0, atol=0)
    assert not torch.equal(adapted, inactive)
    runtime.mixer.close()


def test_gate_plan_cli_records_plan_trace_and_provenance(monkeypatch, tmp_path):
    import hashlib
    import json
    import sys

    from release_tools import infer

    plan = tmp_path / 'gate.json'
    plan.write_text(json.dumps({'method': 'cfp', 'parameter': 0.5, 'direction': [1, 0]}))
    adapter_path = tmp_path / 'RAW.safetensors'
    adapter_path.write_bytes(b'raw fixture')
    output = tmp_path / 'gated.png'
    runtime = LatentRuntime()
    runtime.transformer = object()
    runtime.device = 'cpu'
    runtime.identity = {'model': 'fixture'}
    runtime.mixer.adapters.clear()
    runtime.mixer.add = lambda name, adapter: runtime.mixer.adapters.update({name: adapter})
    runtime.step = lambda scheduler, velocity, timestep, latent: latent + velocity + 0.75
    runtime.decode = lambda latent: SimpleNamespace(save=lambda path: path.write_bytes(b'fixture image'))
    closed = []
    runtime.close = lambda: closed.append(True)

    class Adapter:
        def __init__(self, transformer):
            assert transformer is runtime.transformer

        def load_export(self, path, *, model_identity):
            assert path == adapter_path and model_identity == runtime.identity
            return {'weights': 'live'}

        def to(self, device):
            return self

        def requires_grad_(self, enabled):
            assert not enabled
            return self

        def eval(self):
            return self

    def scheduler(steps):
        return SimpleNamespace(sigmas=torch.linspace(1, 0, steps + 1),
                               timesteps=tuple(range(steps, 0, -1)))

    runtime.scheduler = scheduler
    runtime.predict = lambda latent, timestep, embedding: runtime.mixer.strengths['release']
    monkeypatch.setattr(infer, 'TurboRuntime', lambda *args: runtime)
    monkeypatch.setattr(infer, 'adapter_class', lambda path: Adapter)
    monkeypatch.setattr(infer, 'read_metadata', lambda path: {'weights': 'live'})
    monkeypatch.setattr(sys, 'argv', [
        'infer', '--model', str(tmp_path / 'model'), '--adapter', str(adapter_path), '--prompt', 'fixture',
        '--strength', '1', '--steps', '8', '--gate-plan', str(plan), '--output', str(output),
    ])
    infer.main()
    assert output.read_bytes() == b'fixture image'
    assert closed == [True]
    receipt = json.loads(output.with_suffix('.json').read_bytes())
    assert receipt['activation_gate_plan_sha256'] == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert receipt['activation_gate_plan'] == json.loads(plan.read_text())
    assert receipt['adapter_sha256'] == hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    assert receipt['gate_firings'] == 1 and receipt['gate_eligible_intervals'] == 8
    assert receipt['effective_strengths'] == [1.0] + [0.0] * 7
