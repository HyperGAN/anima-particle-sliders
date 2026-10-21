"""Installed core 0.8.1 qualification; tiny random Cosmos, no pretrained loads."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import importlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, UnsupportedOperation, open_session
from particle_sliders.pg09 import _hash
from lumen_studio import lifecycle
from test_image_lifecycle import configuration
from test_pg09_training import equal, runtime_fixture


def staged_model(tmp_path, monkeypatch, decorate=None):
    model = tmp_path / 'synthetic-model'
    model.mkdir()
    (model / 'anima-lock.json').write_text('{"synthetic_cpu_fixture": true}')
    loaded = []
    def load(self, model_dir, device, verify, checkpointing):
        assert model_dir == str(model) and device == 'cpu' and verify and not checkpointing
        self.__dict__.update(runtime_fixture().__dict__)
        if decorate:
            decorate(self)
        loaded.append(self)
    from lumen_studio.backends.anima import TurboRuntime
    monkeypatch.setattr(TurboRuntime, '__init__', load)
    return dict(model_dir=str(model), device='cpu', base_precision='bfloat16'), loaded


def ban_training_construction(monkeypatch):
    from particle_sliders.pg09 import ConditionalPG09
    from lumen_studio.cache import TargetCache
    from lumen_studio.pg09_training import CachedResponses, NativeFieldCritic
    def forbidden(*args, **kwargs):
        pytest.fail('native-only inference constructed training/cache/critic state')
    for cls in (ConditionalPG09, TargetCache, CachedResponses, NativeFieldCritic):
        monkeypatch.setattr(cls, '__init__', forbidden)


def request_from(session):
    row = session.game.rows[0].row
    return dict(latent=row.latent.detach().clone(), embedding=row.neutral.detach().clone(),
                timestep=float(row.timestep))


def test_raw_and_ema_native_inference_without_cache_or_training(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    config = configuration(tmp_path, runtime)
    artifacts, expected = {}, {}
    with open_session(config, runtime=runtime) as session:
        session.train(stop_after=2)
        request = request_from(session)
        before = session.checkpoint()
        raw = deepcopy(session.game.network.state_dict())
        for selection in ('raw', 'ema'):
            path = tmp_path / (selection + '.pt')
            artifacts[selection] = session.export_inference(path, selection=selection)
            session.game.network.load_state_dict(raw if selection == 'raw' else session.ema.values)
            expected[selection] = session.evaluate(strength=.5)
            session.game.network.load_state_dict(raw)
        assert not equal(artifacts['raw']['network'], artifacts['ema']['network'])
        assert equal(before, session.checkpoint())
        assert 'cache_dir' not in json.dumps(artifacts['ema']['specification'])
        assert 'critic' not in artifacts['ema'] and 'payload' not in artifacts['ema']
        with runtime.mixer.scales({}):
            base = runtime.predict(request['latent'], request['timestep'], request['embedding'])
    runtime.mixer.close()
    shutil.rmtree(Path(config.options['cache_dir']))
    options, loaded = staged_model(tmp_path, monkeypatch)
    ban_training_construction(monkeypatch)
    rng = torch.get_rng_state().clone()
    for selection in ('raw', 'ema'):
        with open_inference(tmp_path / (selection + '.pt'), options=options,
                            operation='response', native_request=request) as native:
            assert native.selection == selection
            assert not any(hasattr(native, name) for name in ('game', 'rows', 'critic', 'optimizer'))
            assert equal(native.binding.network.state_dict(), artifacts[selection]['network'])
            assert torch.equal(native.response(strength=0., seed=11, **request), base)
            assert torch.equal(native.response(strength=.5, seed=11, **request), expected[selection])
            for method in ('train', 'step', 'checkpoint', 'restore'):
                with pytest.raises(UnsupportedOperation):
                    getattr(native, method)()
            assert native.manifest['origin']['pg_version'] == '0.9.1'
        assert not loaded[-1].mixer.adapters and not loaded[-1].mixer.handles
        assert not hasattr(loaded[-1], 'pipe')
    assert torch.equal(rng, torch.get_rng_state())


def test_staged_accelerator_preflight_before_rng_or_load(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    config = configuration(tmp_path, runtime)
    def forbidden(*args, **kwargs):
        pytest.fail('accelerator request reached RNG/device handling or model loading')
    monkeypatch.setattr(torch.random, 'fork_rng', forbidden)
    monkeypatch.setattr(torch.cuda, 'current_device', forbidden)
    monkeypatch.setattr(lifecycle.NativeOwner, '__init__', forbidden)
    staged = SimpleNamespace(device=torch.device('cuda:0'), dtype=torch.bfloat16)
    with pytest.raises(UnsupportedOperation, match='staged Anima runtime must be CPU'):
        lifecycle.BACKEND.open(config, runtime=staged)
    runtime.mixer.close()


def test_before_update_rejection_preserves_state_ema_rng_and_observer(tmp_path):
    runtime = runtime_fixture()
    config = configuration(tmp_path, runtime)
    with open_session(config, runtime=runtime) as session:
        assert session.binding.before_update({'completed_steps': 0, 'manifest': session.manifest}) is True
        before, rng, observed = session.checkpoint(), torch.get_rng_state().clone(), []
        original = session.binding.before_update
        session.binding.before_update = lambda request: False
        with pytest.raises(UnsupportedOperation, match='admission rejected'):
            session.train(stop_after=1, on_step=observed.append)
        assert not session.closed and not observed
        assert equal(before, session.checkpoint())
        assert torch.equal(rng, torch.get_rng_state())
        session.binding.before_update = original
        device = runtime.device
        try:
            runtime.device = torch.device('cuda:0')
            with pytest.raises(UnsupportedOperation, match='accelerator admission remains unqualified'):
                original({'completed_steps': 0, 'manifest': session.manifest})
        finally:
            runtime.device = device
        session.train(stop_after=1, on_step=observed.append)
        assert len(observed) == 1 and session.game.completed_steps == 1
    runtime.mixer.close()


def decorate_observation(runtime):
    runtime.transformer.register_buffer('observation_fixture', torch.tensor([3., 4.]), persistent=False)
    runtime.embeddings = OrderedDict(first=torch.tensor([1.]), second=torch.tensor([2.]))


def observe_mutation(runtime, failure, original):
    def changed(*args, **kwargs):
        runtime.transformer.observation_fixture.add_(10.)
        runtime.transformer.observation_fixture = torch.zeros(3)
        runtime.transformer.register_buffer('added_by_observation', torch.zeros(1))
        runtime.transformer._non_persistent_buffers_set.clear()
        runtime.transformer.train()
        next(runtime.transformer.parameters()).requires_grad_(True)
        runtime.pipe.scheduler.set_timesteps(2, device='cpu')
        runtime.mixer.strengths['injected'] = .75
        runtime.embeddings.pop('first')
        runtime.embeddings['third'] = torch.tensor([9.])
        if failure:
            raise RuntimeError('injected observation failure')
        return original(*args, **kwargs)
    return changed


def observation_snapshot(runtime):
    return (runtime.transformer.observation_fixture,
            deepcopy(runtime.transformer.observation_fixture),
            deepcopy(vars(runtime.pipe.scheduler)), runtime.pipe.scheduler,
            runtime.embeddings, list(runtime.embeddings.items()),
            [(m, m.training) for m in runtime.transformer.modules()],
            [(p, p.requires_grad) for p in runtime.transformer.parameters()],
            runtime.mixer.strengths, runtime.mixer.strengths.copy())


def assert_observation_restored(runtime, before):
    buffer, value, scheduler_state, scheduler, embeddings, entries, modes, gradients, strengths, mix = before
    assert runtime.transformer.observation_fixture is buffer and torch.equal(buffer, value)
    assert 'observation_fixture' in runtime.transformer._non_persistent_buffers_set
    assert not hasattr(runtime.transformer, 'added_by_observation')
    assert runtime.pipe.scheduler is scheduler and equal(vars(scheduler), scheduler_state)
    assert runtime.embeddings is embeddings and list(embeddings) == [key for key, _ in entries]
    assert all(embeddings[key] is value for key, value in entries)
    assert all(module.training == mode for module, mode in modes)
    assert all(parameter.requires_grad == grad for parameter, grad in gradients)
    assert runtime.mixer.strengths is strengths and strengths == mix


@pytest.mark.parametrize('kind', ['training', 'native'])
@pytest.mark.parametrize('failure', [False, True])
def test_product_observation_restores_native_state(tmp_path, monkeypatch, kind, failure):
    runtime = runtime_fixture()
    decorate_observation(runtime)
    config = configuration(tmp_path, runtime)
    session = open_session(config, runtime=runtime)
    if kind == 'native':
        packet = session.export_inference()
        request = request_from(session)
        session.close()
        runtime.mixer.close()
        options, loaded = staged_model(tmp_path, monkeypatch, decorate_observation)
        session = open_inference(packet, options=options, operation='response', native_request=request)
        runtime = loaded[-1]
    before, rng = observation_snapshot(runtime), torch.get_rng_state().clone()
    monkeypatch.setattr(runtime, 'predict', observe_mutation(runtime, failure, runtime.predict))
    try:
        action = (lambda: session.evaluate(strength=.5)) if kind == 'training' else (
            lambda: session.response(strength=.5, **request))
        if failure:
            with pytest.raises(RuntimeError, match='injected observation failure'):
                action()
        else:
            assert torch.isfinite(action()).all()
        assert_observation_restored(runtime, before)
        assert torch.equal(rng, torch.get_rng_state())
        assert not session.closed
    finally:
        session.close()
        runtime.mixer.close()


def test_native_preflight_packet_and_state_rejection(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    config = configuration(tmp_path, runtime)
    with open_session(config, runtime=runtime) as session:
        packet, request = session.export_inference(), request_from(session)
    runtime.mixer.close()
    options, loaded = staged_model(tmp_path, monkeypatch)
    for kind in ('checksum', 'factory', 'geometry', 'references', 'strength'):
        bad = deepcopy(packet)
        arguments = dict(options=options, operation='response', native_request=request)
        if kind == 'checksum':
            bad['sha256'] = '0' * 64
        elif kind == 'factory':
            bad['specification']['factory_sources']['prepare'] = '0' * 64
            bad['sha256'] = _hash({key: value for key, value in bad.items() if key != 'sha256'})
        elif kind == 'geometry':
            arguments['native_request'] = dict(request, latent=torch.zeros(1, 5, 1, 2, 2))
        elif kind == 'references':
            arguments['references'] = True
        else:
            arguments['strength'] = -.5
        with pytest.raises((CheckpointError, UnsupportedOperation, ValueError)):
            open_inference(bad, **arguments)
        assert not loaded
    created = []
    original = lifecycle.NativeOwner.attach
    def capture(owner):
        backend = original(owner)
        created.append((backend.network, deepcopy(backend.network.state_dict())))
        return backend
    monkeypatch.setattr(lifecycle.NativeOwner, 'attach', capture)
    invalid = deepcopy(packet)
    key = next(iter(invalid['network']))
    invalid['network'][key] = torch.zeros(1)
    invalid['sha256'] = _hash({name: value for name, value in invalid.items() if name != 'sha256'})
    with pytest.raises(ValueError, match='tensor schema differs'):
        open_inference(invalid, options=options, operation='response', native_request=request)
    assert len(created) == 1 and equal(created[0][0].state_dict(), created[0][1])
    assert not loaded[-1].mixer.adapters and not loaded[-1].mixer.handles


@pytest.mark.parametrize('command', ['train', 'benchmark'])
def test_installed_legacy_cli_refuses_before_side_effects(tmp_path, monkeypatch, command):
    cli = importlib.import_module('lumen_studio.__main__')
    root = tmp_path / 'untouched'
    monkeypatch.setattr(sys, 'argv', ['lumen_studio', '--root', str(root), command, 'candlelit'])
    with pytest.raises(RuntimeError, match='PG09Trainer'):
        cli.main()
    assert not root.exists()
