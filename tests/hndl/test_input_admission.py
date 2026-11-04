"""Self-created tiny file fixtures; no model loader consumes attack fixtures."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from anima_hndl import admission
from anima_hndl import lifecycle as candidate
from lumen_studio.contracts import digest


def stage_owned_runtime(tmp_path, monkeypatch):
    from test_image_lifecycle081 import staged_model
    from lumen_studio.backends.anima import TurboRuntime
    options, loaded = staged_model(tmp_path, monkeypatch)
    monkeypatch.setattr(candidate.LocalTurboRuntime, '__init__', TurboRuntime.__init__)
    return options, loaded


def stage_model_admission_fixture(root, monkeypatch):
    """Unit-test authority; production reads only its committed package manifests.

    Placeholder bytes are hashed but never parsed as weights. Existing owned
    runtime tests supply a separate already-created random Cosmos fixture.
    """
    root = Path(root)
    (root / 'transformer').mkdir(parents=True, exist_ok=True)
    weight = root / 'transformer/model.safetensors'
    weight.write_bytes(b'self-created trusted file fixture; never loaded as weights')
    index = {'components': [{'name': 'transformer',
              'pretrained_model_name_or_path': str(root), 'subfolder': 'transformer',
              'type_hint': ['diffusers', 'CosmosTransformer3DModel']}]}
    index_bytes = json.dumps(index).encode()
    (root / 'modular_model_index.json').write_bytes(index_bytes)
    trusted = {'model': 'circlestone-labs/Anima:turbo-v1.1', 'revision': 'a' * 40,
        'files': {'transformer/model.safetensors': hashlib.sha256(weight.read_bytes()).hexdigest(),
                  'modular_model_index.json': hashlib.sha256(index_bytes).hexdigest()}}
    (root / 'anima-lock.json').write_text(json.dumps(trusted))
    portable = deepcopy(trusted)
    portable_index = {'components': [{'name': 'transformer',
        'pretrained_model_name_or_path': '$MODEL_ROOT', 'subfolder': 'transformer',
        'type_hint': ['diffusers', 'CosmosTransformer3DModel']}]}
    portable['files']['modular_model_index.json'] = digest(portable_index)
    released = {'model': trusted['model'], 'sha256': digest(portable)}
    monkeypatch.setattr(admission, '_trusted_model', lambda: (deepcopy(trusted), deepcopy(released)))
    return trusted, released


def test_exact_trusted_fixture_layout_and_content_pass(tmp_path, monkeypatch):
    _, released = stage_model_admission_fixture(tmp_path, monkeypatch)
    receipt = admission.verify_model_directory(tmp_path)
    assert {k: receipt[k] for k in released} == released
    assert len(receipt['files_stat_sha256']) == 64


@pytest.mark.parametrize('attack', ['extra_index', 'self_attested_extra_index', 'self_attested_weights',
    'changed_weights', 'changed_revision', 'external_component', 'changed_component_class',
    'weight_symlink', 'index_symlink', 'component_symlink', 'fifo'])
def test_model_attack_fixtures_refused_before_any_loader(tmp_path, monkeypatch, attack):
    root = tmp_path / 'model'
    stage_model_admission_fixture(root, monkeypatch)
    lock = json.loads((root / 'anima-lock.json').read_text())
    weight = root / 'transformer/model.safetensors'
    if attack in ('extra_index', 'self_attested_extra_index'):
        name = 'transformer/model.safetensors.index.json'
        path = root / name
        path.write_text(json.dumps({'weight_map': {'weight': '../../outside.safetensors'}}))
        if attack == 'self_attested_extra_index':
            lock['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif attack in ('changed_weights', 'self_attested_weights'):
        weight.write_bytes(b'changed own fixture bytes')
        if attack == 'self_attested_weights':
            lock['files']['transformer/model.safetensors'] = hashlib.sha256(weight.read_bytes()).hexdigest()
    elif attack == 'changed_revision':
        lock['revision'] = 'b' * 40
    elif attack in ('external_component', 'changed_component_class'):
        path = root / 'modular_model_index.json'
        index = json.loads(path.read_text())
        if attack == 'external_component':
            index['components'][0]['pretrained_model_name_or_path'] = str(tmp_path / 'outside')
        else:
            index['components'][0]['type_hint'][1] = 'UnadmittedModel'
        path.write_text(json.dumps(index))
        lock['files']['modular_model_index.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif attack in ('weight_symlink', 'index_symlink'):
        path = weight if attack == 'weight_symlink' else root / 'modular_model_index.json'
        outside = tmp_path / 'own-outside-file'
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    elif attack == 'component_symlink':
        outside = tmp_path / 'own-outside-component'
        (root / 'transformer').rename(outside)
        (root / 'transformer').symlink_to(outside, target_is_directory=True)
    else:
        weight.unlink()
        os.mkfifo(weight)
    (root / 'anima-lock.json').write_text(json.dumps(lock))
    def forbidden(*args, **kwargs):
        pytest.fail('attack fixture reached a model loader')
    monkeypatch.setattr(candidate.LocalTurboRuntime, '__init__', forbidden)
    with pytest.raises(ValueError, match='model (layout|file|lock|component|index)'):
        candidate.NativeOwner({'model_dir': str(root), 'variation': 'finalform'})


def test_production_manifest_cannot_be_replaced_by_local_self_attestation(tmp_path, monkeypatch):
    (tmp_path / 'anima-lock.json').write_text(json.dumps({'model': 'circlestone-labs/Anima:turbo-v1.1',
                                                       'files': {}}))
    monkeypatch.setattr(candidate.LocalTurboRuntime, '__init__', lambda *a, **k: pytest.fail('unadmitted layout reached loader'))
    with pytest.raises(ValueError, match='committed conversion manifest'):
        candidate.NativeOwner({'model_dir': str(tmp_path), 'variation': 'finalform'})


@pytest.mark.parametrize('same_bytes', [False, True])
def test_owned_loader_changing_admitted_model_is_closed_and_refused(tmp_path, monkeypatch, same_bytes):
    options, loaded = stage_owned_runtime(tmp_path, monkeypatch)
    options['variation'] = 'finalform'
    stage_model_admission_fixture(options['model_dir'], monkeypatch)
    original = candidate.LocalTurboRuntime.__init__
    def changed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        path = Path(options['model_dir']) / 'transformer/model.safetensors'
        if same_bytes:
            replacement = path.with_suffix('.replacement')
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        else:
            path.write_bytes(b'changed own fixture')
    monkeypatch.setattr(candidate.LocalTurboRuntime, '__init__', changed)
    with pytest.raises(ValueError, match='model file (differs|identities changed)'):
        candidate.NativeOwner(options)
    assert not hasattr(loaded[-1], 'pipe')
    assert not loaded[-1].mixer.adapters and not loaded[-1].mixer.handles


@pytest.mark.parametrize('value', ['.', 'transformer', 'repo/model', None, '', 'trailing', 'component', 'noncanonical'])
def test_literal_component_root_required_before_loader(tmp_path, monkeypatch, value):
    root = tmp_path / 'model'
    stage_model_admission_fixture(root, monkeypatch)
    replacements = {'trailing': str(root) + '/', 'component': str(root / 'transformer'),
                    'noncanonical': str(root) + '/../model'}
    literal = replacements.get(value, value)
    path = root / 'modular_model_index.json'
    index = json.loads(path.read_text())
    index['components'][0]['pretrained_model_name_or_path'] = literal
    path.write_text(json.dumps(index))
    lock_path = root / 'anima-lock.json'
    lock = json.loads(lock_path.read_text())
    lock['files']['modular_model_index.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(candidate.LocalTurboRuntime, '__init__', lambda *a, **k: pytest.fail('ambiguous literal reached loader'))
    monkeypatch.chdir(tmp_path)  # Relative names could otherwise resolve this different CWD.
    with pytest.raises(ValueError, match='exact absolute admitted root'):
        candidate.NativeOwner({'model_dir': str(root), 'variation': 'finalform'})


def test_fifo_descriptor_open_is_bounded_and_refused(tmp_path):
    os.mkfifo(tmp_path / 'shard')
    code = ('from anima_hndl.admission import _regular\n'
            f'with _regular({str(tmp_path)!r}, "shard"):\n    raise AssertionError("FIFO admitted")\n')
    result = subprocess.run([sys.executable, '-I', '-c', code],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert 'must be a regular file' in result.stderr


def test_verify_false_reader_cannot_bypass_candidate_content_hashes(tmp_path, monkeypatch):
    from test_anima_hndl import config_for
    from test_pg09_training import runtime_fixture, equal
    from particle_sliders.lifecycle import open_session
    runtime = runtime_fixture()
    config = config_for(tmp_path, runtime)
    before, rng = deepcopy(runtime.transformer.state_dict()), torch.get_rng_state().clone()
    (Path(config.options['cache_dir']) / '0.pt').write_bytes(b'own corrupt shard fixture')
    monkeypatch.setattr(candidate.NativeOwner, 'attach', lambda *a: pytest.fail('corrupt shard reached attachment'))
    try:
        with pytest.raises(ValueError, match='Corrupt PG09 target shard'):
            open_session(config, runtime=runtime)
        assert equal(before, runtime.transformer.state_dict())
        assert torch.equal(rng, torch.get_rng_state())
        assert not runtime.mixer.adapters and not runtime.mixer.handles
    finally:
        runtime.mixer.close()


def test_index_replacement_between_admission_and_reader_is_refused(tmp_path, monkeypatch):
    from test_anima_hndl import config_for
    from test_pg09_training import runtime_fixture
    from lumen_studio.cache import TargetCache
    from particle_sliders.lifecycle import open_session
    runtime = runtime_fixture()
    config = config_for(tmp_path, runtime)
    original = TargetCache.__init__
    def replace_index(self, directory, **kwargs):
        path = Path(directory) / 'index.json'
        index = json.loads(path.read_text())
        index['unexpected_replacement'] = True
        path.write_text(json.dumps(index))
        original(self, directory, **kwargs)
    monkeypatch.setattr(TargetCache, '__init__', replace_index)
    monkeypatch.setattr(candidate.NativeOwner, 'attach', lambda *a: pytest.fail('changed index reached attachment'))
    try:
        with pytest.raises(ValueError, match='between admission and provider'):
            open_session(config, runtime=runtime)
        assert not runtime.mixer.adapters and not runtime.mixer.handles
    finally:
        runtime.mixer.close()


@pytest.mark.parametrize('attack', ['traversal', 'symlink', 'fifo'])
def test_selected_cache_paths_are_regular_and_contained(tmp_path, attack):
    path = tmp_path / 'shard'
    if attack == 'fifo':
        os.mkfifo(path)
    elif attack == 'symlink':
        target = tmp_path / 'own-target'
        target.write_bytes(b'own data')
        path.symlink_to(target)
    else:
        path.write_bytes(b'own data')
    index = {'shards': [{'path': '../outside' if attack == 'traversal' else 'shard'}]}
    data = json.dumps(index).encode()
    (tmp_path / 'index.json').write_bytes(data)
    with pytest.raises(ValueError, match='admitted (file|input)'):
        admission.verified_cache_index({'cache_dir': str(tmp_path),
                                        'cache_index_sha256': hashlib.sha256(data).hexdigest()})
