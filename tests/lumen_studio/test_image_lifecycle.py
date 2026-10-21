"""Bounded lifecycle checks using the existing random native Cosmos fixture."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest
import torch

from particle_sliders.lifecycle import CheckpointError, SessionConfig, UnsupportedOperation, open_session
from particle_sliders.pg09 import _hash
from lumen_studio import lifecycle
from lumen_studio.particles import ParticleAdapter
from test_pg09_training import cache_fixture, equal, runtime_fixture


@pytest.fixture(autouse=True)
def one_cpu_thread():
    # The historical parent conftest sets two threads at import time.
    # Override it before any fixture/model/update work in this allocated slot.
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    yield
    assert torch.get_num_threads() == 1 and not torch.cuda.is_initialized()


def configuration(tmp_path, runtime):
    cache = cache_fixture(tmp_path / 'cache', runtime)
    config = SessionConfig('anima', 'anima-image-v1-cpu-conformance',
        cache_dir=str(cache.directory),
        cache_index_sha256=hashlib.sha256((cache.directory / 'index.json').read_bytes()).hexdigest(),
        variation='finalform', device='cpu', base_precision='bfloat16',
        steps=4, batch=2, seed=7, critic_width=8, ema_decay=.9)
    return config


def checkpoint_settings(root):
    keys = ('gradient_checkpointing', '_gradient_checkpointing_func')
    return [(module, {key: getattr(module, key) for key in keys if hasattr(module, key)})
            for module in root.modules()]


def assert_cpu():
    assert torch.get_num_threads() == 1
    assert not torch.cuda.is_initialized()


def test_exact_split_resume_ema_coverage_and_inference(tmp_path):
    runtime = runtime_fixture()
    config = configuration(tmp_path, runtime)
    frozen = deepcopy(runtime.transformer.state_dict())
    with open_session(config, runtime=runtime) as session:
        assert session.game.prior.z is session.game.backend.adapter.particles
        assert session.game.rows.response_shape == (1, 4, 1, 2, 2)
        initial_bank = session.game.prior.z.detach().clone()
        session.train(stop_after=2)
        packet = session.checkpoint(tmp_path / 'training.pt')
        exported = session.export(tmp_path / 'ema.pt', selection='ema')
        assert not torch.equal(initial_bank, session.game.prior.z)
        expected_metrics = session.train()
        expected = session.checkpoint()
        assert all(torch.equal(value, frozen[name]) for name, value in runtime.transformer.state_dict().items())
    assert not runtime.mixer.adapters and not runtime.mixer.handles
    other = runtime_fixture()
    with open_session(config, resume=tmp_path / 'training.pt', runtime=other) as resumed:
        assert equal(packet, resumed.checkpoint())
        assert equal(expected_metrics, resumed.train())
        assert equal(expected, resumed.checkpoint())
    with open_session(config, artifact=tmp_path / 'ema.pt', operation='evaluate', runtime=other) as inference:
        assert inference.inference_only
        response = inference.evaluate(strength=.5)
        assert response.shape == (1, 4, 1, 2, 2) and torch.isfinite(response).all()
        with pytest.raises(RuntimeError, match='inference export is not a resumable'):
            inference.step()
        assert inference.closed
    runtime.mixer.close()
    other.mixer.close()
    assert_cpu()


def test_observation_zero_fractional_rng_modes_and_invalid_restore(tmp_path):
    runtime = runtime_fixture()
    config = configuration(tmp_path, runtime)
    with lifecycle.BACKEND.open(config, runtime=runtime) as session:
        session.train(stop_after=2)
        before = session.checkpoint()
        rng = torch.get_rng_state().clone()
        modes = [(module, module.training) for module in session.game.network.modules()]
        prepared = session.game.rows[0]
        with runtime.mixer.scales({}):
            base = runtime.predict(prepared.row.latent, prepared.row.timestep, prepared.row.neutral)
        assert torch.equal(session.evaluate(strength=0.), base)
        assert torch.isfinite(session.evaluate(strength=.5)).all()
        assert torch.equal(rng, torch.get_rng_state())
        assert equal(before, session.checkpoint())
        assert all(module.training == previous for module, previous in modes)
        assert equal(session.evaluate(seed=11, strength=.5), session.evaluate(seed=11, strength=.5))
        for kind in ('checksum', 'ema', 'cohort'):
            bad = deepcopy(before)
            if kind == 'checksum':
                bad['sha256'] = '0' * 64
            elif kind == 'ema':
                key = next(iter(bad['payload']['ema']['values']))
                bad['payload']['ema']['values'][key] = torch.zeros(1)
            else:
                bad['manifest']['config']['cohort'] = 'different'
            if kind != 'checksum':
                bad['sha256'] = _hash({key: value for key, value in bad.items() if key != 'sha256'})
            with pytest.raises(CheckpointError):
                session.restore(bad)
            assert equal(before, session.checkpoint())
            assert not session.closed
    runtime.mixer.close()
    assert_cpu()


def test_capability_preflight_never_invokes_factory(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('unsupported request reached model allocation')
    guarded = lifecycle.Backend('anima', lifecycle.CAPABILITIES, forbidden, lifecycle.validate)
    config = SessionConfig('anima', 'preflight')
    for request in ({'operation': 'render'}, {'strength': -.5}, {'strength': 1.1}, {'references': True}):
        with pytest.raises(UnsupportedOperation):
            guarded.open(config, **request)
    with pytest.raises(UnsupportedOperation, match='requires CPU'):
        guarded.open(SessionConfig('anima', 'preflight', device='cuda:0'))
    assert_cpu()


def test_borrowed_runtime_cleanup_preserves_resident_hooks_mix_and_checkpointing(tmp_path):
    runtime = runtime_fixture()
    config = configuration(tmp_path, runtime)
    resident = ParticleAdapter(runtime.transformer)
    runtime.mixer.add('resident', resident)
    runtime.mixer.strengths['resident'] = .25
    runtime.transformer.enable_gradient_checkpointing()
    before = checkpoint_settings(runtime.transformer)
    handles, strengths = tuple(runtime.mixer.handles), runtime.mixer.strengths.copy()
    with lifecycle.BACKEND.open(config, runtime=runtime) as session:
        session.train(stop_after=1)
        session.evaluate(strength=.5)
        assert runtime.mixer.strengths['resident'] == .25
    session.close()
    assert runtime.mixer.adapters == {'resident': resident}
    assert runtime.mixer.strengths == strengths
    assert tuple(runtime.mixer.handles) == handles
    assert checkpoint_settings(runtime.transformer) == before
    runtime.mixer.close()
    assert_cpu()


def test_setup_and_update_failure_cleanup(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    config = configuration(tmp_path, runtime)
    runtime.transformer.enable_gradient_checkpointing()
    before = checkpoint_settings(runtime.transformer)
    from particle_sliders import pg09_backends
    original = pg09_backends.AnimaRuntimeResponseAdapter
    def fail_binding(*args):
        original(*args)
        raise RuntimeError('injected setup failure')
    with monkeypatch.context() as scoped:
        scoped.setattr(pg09_backends, 'AnimaRuntimeResponseAdapter', fail_binding)
        with pytest.raises(RuntimeError, match='injected setup failure'):
            lifecycle.BACKEND.open(config, runtime=runtime)
    assert not runtime.mixer.adapters and not runtime.mixer.handles
    assert checkpoint_settings(runtime.transformer) == before
    session = lifecycle.BACKEND.open(config, runtime=runtime)
    def fail_update():
        raise RuntimeError('injected update failure')
    monkeypatch.setattr(session.game, 'step', fail_update)
    with pytest.raises(RuntimeError, match='injected update failure'):
        session.step()
    assert session.closed and not runtime.mixer.adapters and not runtime.mixer.handles
    assert checkpoint_settings(runtime.transformer) == before
    runtime.mixer.close()
    assert_cpu()


def test_owned_factory_load_and_partial_load_cleanup(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    config = configuration(tmp_path, runtime)
    model = tmp_path / 'synthetic-model-directory'
    model.mkdir()
    (model / 'anima-lock.json').write_text('{"synthetic_cpu_fixture": true}')
    config = SessionConfig(config.backend, config.cohort, **dict(config.options, model_dir=str(model)))
    from lumen_studio.backends.anima import TurboRuntime
    loaded = []
    def synthetic_load(self, model_dir, device, verify, checkpointing):
        assert str(model_dir) == str(model) and device == 'cpu' and verify and not checkpointing
        self.__dict__.update(runtime_fixture().__dict__)
        loaded.append(self)
    monkeypatch.setattr(TurboRuntime, '__init__', synthetic_load)
    with open_session(config) as session:
        session.train(stop_after=1)
    assert not hasattr(loaded[-1], 'pipe') and not hasattr(loaded[-1], 'transformer')
    assert not loaded[-1].mixer.adapters and not loaded[-1].mixer.handles
    def fail_load(self, *args, **kwargs):
        self.pipe = SimpleNamespace(synthetic=True)
        loaded.append(self)
        raise RuntimeError('injected partial load failure')
    monkeypatch.setattr(TurboRuntime, '__init__', fail_load)
    with pytest.raises(RuntimeError, match='injected partial load failure'):
        lifecycle.BACKEND.open(config)
    assert not hasattr(loaded[-1], 'pipe')
    runtime.mixer.close()
    assert_cpu()
