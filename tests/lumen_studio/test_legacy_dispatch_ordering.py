"""Unsupported legacy dispatch must stop before runtime/cache/output work."""
import asyncio
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def guarded_surfaces(monkeypatch):
    modules = {name: importlib.import_module(name) for name in (
        'lumen_studio.__main__', 'lumen_studio.benchmark', 'lumen_studio.preflight',
        'lumen_studio.coordinator', 'lumen_studio.backends.anima', 'lumen_studio.cache',
        'lumen_studio.training', 'lumen_studio.store', 'lumen_studio.contracts',
        'lumen_studio.dataset', 'release_tools.train', 'scripts.anima_campaign',
    )}
    effects = []

    def trap(name):
        def forbidden(*args, **kwargs):
            effects.append(name)
            pytest.fail('unsupported legacy dispatch reached ' + name)
        return forbidden

    class GuardPath(type(Path())):
        pass

    for name in ('resolve', 'exists', 'is_file', 'is_dir', 'stat', 'read_text', 'read_bytes',
                 'write_text', 'write_bytes', 'open', 'mkdir', 'rename', 'replace', 'unlink',
                 'rmdir', 'glob', 'rglob', 'iterdir'):
        monkeypatch.setattr(GuardPath, name, trap('Path.' + name))
    for module in modules.values():
        if hasattr(module, 'Path'):
            monkeypatch.setattr(module, 'Path', GuardPath)
        for name in ('atomic_json', 'save_tensor_file', 'prepare_targets', 'fit_normalization',
                     'compile_manifest'):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, trap(module.__name__ + '.' + name))

    for module_name, class_name in (
        ('lumen_studio.backends.anima', 'TurboRuntime'), ('lumen_studio.cache', 'TargetCache'),
        ('lumen_studio.training', 'Trainer'), ('lumen_studio.store', 'Store'),
        ('lumen_studio.coordinator', 'Coordinator'), ('lumen_studio.coordinator', 'GpuLease'),
    ):
        cls = getattr(modules[module_name], class_name)
        monkeypatch.setattr(cls, '__init__', trap(module_name + '.' + class_name))
    coordinator = modules['lumen_studio.coordinator'].Coordinator
    for name in ('load_runtime', 'release_runtime', 'enough_memory'):
        monkeypatch.setattr(coordinator, name, trap('Coordinator.' + name))
    for name in ('run', 'check_output', 'Popen'):
        monkeypatch.setattr(subprocess, name, trap('subprocess.' + name))
    for name in ('empty_cache', 'mem_get_info', 'get_device_properties', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, trap('torch.cuda.' + name))
    # Roll back any accidental device mutation even when a regression fails.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    return SimpleNamespace(modules=modules, effects=effects, trap=trap, Path=GuardPath,
                           coordinator=coordinator)


def expect_unavailable(action, surfaces, root):
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match='PG09Trainer') as caught:
        action()
    assert 'Historical' in str(caught.value)
    assert surfaces.effects == []
    assert dict(os.environ) == before
    assert not root.exists()
    assert not torch.cuda.is_initialized()


def test_coordinator_direct_training_rejects_before_releasing_a_resident_runtime(tmp_path, guarded_surfaces):
    surfaces = guarded_surfaces
    worker = surfaces.coordinator.__new__(surfaces.coordinator)
    resident = object()
    worker.runtime = resident
    worker.device, worker.gpu = 'cuda:0', '0'
    worker.store = SimpleNamespace(update_run=surfaces.trap('store.update_run'))
    root = tmp_path / 'untouched'
    run = dict(id='legacy', step=17, config=dict(variation='candlelit', directory=str(root),
        train_cache=str(root / 'targets/train'), dev_cache=str(root / 'targets/dev')))
    expect_unavailable(lambda: worker.train_quantum(run), surfaces, root)
    assert worker.runtime is resident


@pytest.mark.parametrize('entrypoint', ['benchmark', 'preflight'])
def test_legacy_diagnostics_reject_before_gpu_queries_cache_or_archive(tmp_path, guarded_surfaces, entrypoint):
    surfaces = guarded_surfaces
    root = tmp_path / 'untouched'
    if entrypoint == 'benchmark':
        action = lambda: surfaces.modules['lumen_studio.benchmark'].benchmark(root, 'candlelit', gpu='0')
    else:
        configuration = dict(name='legacy', microbatch=1, checkpointing=True)
        action = lambda: surfaces.modules['lumen_studio.preflight'].verify_gpu_resume(
            root, configuration, gpu='0', profile_update=True)
    expect_unavailable(action, surfaces, root)


@pytest.mark.parametrize('command', ['train', 'benchmark'])
def test_module_cli_rejects_before_output_directory_queue_or_device_change(
        tmp_path, monkeypatch, guarded_surfaces, command):
    surfaces = guarded_surfaces
    root = tmp_path / 'untouched'
    args = ['lumen_studio', '--root', str(root), command, 'candlelit']
    if command == 'train':
        args += ['--until', '1600']
    else:
        args += ['--gpu', '0']
    monkeypatch.setattr(sys, 'argv', args)
    expect_unavailable(surfaces.modules['lumen_studio.__main__'].main, surfaces, root)


@pytest.mark.parametrize('variation', ['candlelit', 'bad-intent', 'final-form', 'afterimage', 'dusk'])
def test_release_training_rejects_before_manifest_rebinding_or_target_preparation(
        tmp_path, monkeypatch, guarded_surfaces, variation):
    surfaces = guarded_surfaces
    root = tmp_path / 'untouched'
    cache_module = surfaces.modules['lumen_studio.cache']
    validator = cache_module.validate_manifest
    monkeypatch.setattr(sys, 'argv', ['release_tools.train', variation, '--root', str(root),
        '--model', str(root / 'model'), '--prepare-targets', '--until', '1600'])
    try:
        expect_unavailable(surfaces.modules['release_tools.train'].main, surfaces, root)
        assert cache_module.validate_manifest is validator
    finally:
        # A broken legacy route must not contaminate later tests with its rebind.
        cache_module.validate_manifest = validator


def test_campaign_cli_rejects_before_qualification_reads_pid_files_or_worker_spawn(
        tmp_path, monkeypatch, guarded_surfaces):
    surfaces = guarded_surfaces
    root = tmp_path / 'untouched'
    monkeypatch.setattr(sys, 'argv', ['anima_campaign', '--root', str(root), '--gpu', '0',
        '--variation', 'candlelit'])
    expect_unavailable(surfaces.modules['scripts.anima_campaign'].main, surfaces, root)


def test_training_api_rejects_before_store_or_cache_reads_and_maps_to_http409(tmp_path, monkeypatch):
    # Direct endpoint/handler calls avoid the separately reproduced TestClient stall.
    api = importlib.import_module('lumen_studio.api')
    app = api.create_app(tmp_path / 'studio', model_identity=dict(model='CPU fixture'))
    store = app.state.store
    before = deepcopy(store.runs())
    effects = []
    def forbidden(*args, **kwargs):
        effects.append('store/cache access')
        pytest.fail('unsupported training API reached store/cache work')
    for name in ('connect', 'runs', 'add_run'):
        monkeypatch.setattr(store, name, forbidden)
    class GuardPath(type(Path())):
        exists = forbidden
    monkeypatch.setattr(api, 'Path', GuardPath)
    endpoint = next(route.endpoint for route in app.routes
                    if getattr(route, 'path', None) == '/api/training/runs' and 'POST' in route.methods)
    with pytest.raises(RuntimeError, match='PG09Trainer') as caught:
        endpoint(api.TrainingRequest(variation='candlelit', until=1600))
    assert effects == []
    handler = app.exception_handlers[type(caught.value)]
    response = asyncio.run(handler(None, caught.value))
    assert response.status_code == 409
    assert 'PG09Trainer' in json.loads(response.body)['detail']
    monkeypatch.undo()
    assert store.runs() == before
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('status', ['queued', 'running'])
@pytest.mark.parametrize('image_work', [False, True])
def test_worker_fails_legacy_runs_inside_lease_and_preserves_inference_dispatch(
        tmp_path, monkeypatch, status, image_work):
    coordinator = importlib.import_module('lumen_studio.coordinator')
    events, lease_active = [], [False]
    root = tmp_path / 'untouched'

    @contextmanager
    def lease(*args, **kwargs):
        events.append('lease.enter')
        lease_active[0] = True
        try:
            yield
        finally:
            lease_active[0] = False
            events.append('lease.exit')

    class Store:
        def __init__(self):
            self.rows = [dict(id='legacy', status=status, step=17, config=dict(variation='candlelit', until=1600))]
            self.images = [dict(id='image', status='queued')] if image_work else []
            self.settings = dict(paused=False, render_burst=0, training_updates=20, owner=None)

        def recover(self):
            assert lease_active[0]
            events.append('store.recover')
            for row in self.rows:
                if row['status'] == 'running':
                    row['status'] = 'queued'

        def runs(self):
            return deepcopy(self.rows)

        def jobs(self):
            return deepcopy(self.images)

        def controls(self):
            return deepcopy(self.settings)

        def control(self, **values):
            assert lease_active[0]
            self.settings.update(values)

        def update_run(self, ident, status, step, metrics=None, error=None):
            assert lease_active[0]
            events.append('legacy.reject')
            assert ident == 'legacy' and status == 'failed' and step == 17
            assert 'PG09Trainer' in error
            self.rows[0].update(status=status, step=step, error=error)

    worker = coordinator.Coordinator.__new__(coordinator.Coordinator)
    worker.store, worker.gpu, worker.device = Store(), 'cpu', 'cpu'
    worker.runtime, worker.runtime_factory, worker.stopping = None, None, False
    def forbidden(*args, **kwargs):
        pytest.fail('legacy worker dispatch reached GPU/runtime/cache work')
    worker.train_quantum = worker.load_runtime = forbidden
    def enough_memory(training):
        assert training is False
        events.append('inference.memory')
        return True
    worker.enough_memory = enough_memory
    worker.release_runtime = lambda: events.append('inference.release')
    def render():
        events.append('inference.render')
        worker.store.images.clear()
        return True
    worker.render_one = render
    monkeypatch.setattr(coordinator, 'GpuLease', lease)
    for name in ('run', 'check_output', 'Popen'):
        monkeypatch.setattr(subprocess, name, forbidden)
    worker.run(once=True)
    assert events.index('lease.enter') < events.index('store.recover') < events.index('legacy.reject')
    assert worker.store.rows[0]['status'] == 'failed'
    assert events.count('inference.render') == int(image_work)
    assert events.count('inference.memory') == int(image_work)
    assert events[-1] == 'lease.exit'
    assert not root.exists()
    assert not torch.cuda.is_initialized()
