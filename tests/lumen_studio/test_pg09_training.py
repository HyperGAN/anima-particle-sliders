"""Real tiny Cosmos architecture, native product bank and released PG0.9 on CPU."""
from copy import deepcopy
from importlib.metadata import version
import json
import os
from types import SimpleNamespace

import pytest
import torch

from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.cache import TargetCache, save_tensor_file
from lumen_studio.contracts import atomic_json, digest, file_hash
from lumen_studio.particles import Mixer, ParticleAdapter
from lumen_studio.pg09_training import CachedResponses, PG09Trainer


@pytest.mark.parametrize('device,fixture', [('cuda:0', False), ('cuda:0', True), ('cpu', False)])
def test_historical_native_training_is_denied_before_allocations(tmp_path, monkeypatch, device, fixture):
    from lumen_studio import training
    def forbidden(*args, **kwargs):
        pytest.fail('historical native training reached allocation')
    monkeypatch.setattr(training, 'ThreadPoolExecutor', forbidden)
    runtime = SimpleNamespace(device=torch.device(device))
    with pytest.raises(RuntimeError, match='Historical gmix training is disabled'):
        training.Trainer(runtime, object(), tmp_path / 'unused', 'finalform', legacy_cpu_fixture=fixture)
    assert not (tmp_path / 'unused').exists()
    assert not torch.cuda.is_initialized()


def runtime_fixture():
    from diffusers import CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler
    torch.manual_seed(9)
    runtime = TurboRuntime.__new__(TurboRuntime)  # Never invokes a model loader.
    runtime.transformer = CosmosTransformer3DModel(
        in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=32,
        num_layers=1, mlp_ratio=2, text_embed_dim=8, adaln_lora_dim=4,
        max_size=(2, 4, 4), patch_size=(1, 1, 1), extra_pos_embed_type=None,
    ).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    runtime.device, runtime.dtype, runtime.scale = torch.device('cpu'), torch.bfloat16, 1
    runtime.identity = dict(model='CPU random tiny Cosmos; no pretrained weights', sha256='a' * 64)
    runtime.pipe = SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000))
    runtime.mixer = Mixer(runtime.transformer)
    return runtime


def cache_fixture(path, runtime):
    path.mkdir()
    identity = dict(model=runtime.identity, split='train', variation='finalform', manifest_sha256='b' * 64)
    fingerprint = digest(identity)
    shards = []
    for index in range(2):
        latent = torch.randn(1, 4, 1, 2, 2, generator=torch.Generator().manual_seed(index))
        embedding = torch.full((1, 2 + index, 8), .2 + index * .1)
        timestep = 500. - 100 * index
        with torch.no_grad():
            neutral = runtime.predict(latent, timestep, embedding)
            positive = runtime.predict(latent, timestep, embedding + .3)
        record = dict(latent=latent, neutral=neutral, positive=positive,
            timestep=timestep, position=index, trajectory=('neutral', 'positive')[index])
        name = f'{index}.pt'
        save_tensor_file(path / name, dict(fingerprint=fingerprint, records=[record],
            embedding=embedding, row=dict(id=f'display-{index}'), seed=index))
        shards.append(dict(path=name, sha256=file_hash(path / name), count=1))
    atomic_json(path / 'index.json', dict(identity=identity, fingerprint=fingerprint, shards=shards))
    return TargetCache(path)


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def test_native_bank_shared_pg09_exact_resume_and_inference_export(tmp_path):
    from particle_sliders.pg09 import ConditionalPG09, PG_VERSION
    from particle_sliders.pg09_backends import AnimaRuntimeResponseAdapter
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    before = deepcopy(runtime.transformer.state_dict())
    trainer = PG09Trainer(runtime, cache, tmp_path / 'run', 'finalform', total_steps=4, critic_width=8)
    assert type(trainer.game) is ConditionalPG09
    assert type(trainer.backend) is AnimaRuntimeResponseAdapter
    assert trainer.game.prior.z is trainer.adapter.particles
    assert trainer.responses.response_shape == (1, 4, 1, 2, 2)
    bank_before = trainer.adapter.particles.detach().clone()
    branches_before = deepcopy(trainer.adapter.branches.state_dict())
    metrics = trainer.update()
    assert metrics['g_row_exposures'] == metrics['d_row_exposures'] == 2
    trainer.update()
    assert not torch.equal(bank_before, trainer.adapter.particles)
    assert any(not torch.equal(v, branches_before[k]) for k, v in trainer.adapter.branches.state_dict().items())
    with trainer.backend.network.scaled(1):
        value = trainer.backend.generated(trainer.responses[0])
        bank_grad = torch.autograd.grad(value.sum(), trainer.adapter.particles)[0]
    assert bank_grad.abs().sum() > 0
    trainer.save()
    checkpoint = torch.load(tmp_path / 'run/pg09-resume.pt', weights_only=True)
    assert checkpoint['config']['pg_version'] == PG_VERSION == version('particlegan')
    assert checkpoint['config']['objective'] == 'g_adv+public_prior;d_adv+public_penalty'
    assert checkpoint['config']['response_domain'] == 'continuous_field'
    assert checkpoint['config']['resolved_penalty_domain'] == 'image'
    assert checkpoint['config']['sampling'] == 'balanced'
    assert checkpoint['config']['generator_arm'] == 'routed_particle_gan'
    assert all(torch.equal(value, before[name]) for name, value in runtime.transformer.state_dict().items())
    cache.index['fingerprint'] = 'c' * 64
    cache.index['identity']['manifest_sha256'] = 'd' * 64
    export = tmp_path / 'native-live.safetensors'
    trainer.export(export)
    # Caller-owned cache metadata cannot relabel the actual trained law.
    cache.index = deepcopy(trainer.responses.identity()['index'])
    export_prediction = trainer.game.evaluate(0)
    expected_metrics = trainer.update()
    expected_state = trainer.game.state_dict()
    other_runtime = runtime_fixture()
    resumed = PG09Trainer(other_runtime, cache, tmp_path / 'run', 'finalform', total_steps=4, critic_width=8)
    assert resumed.step == 2
    assert equal(expected_metrics, resumed.update())
    assert equal(expected_state, resumed.game.state_dict())
    from single_particle import adapter_class
    native = adapter_class(export)(other_runtime.transformer)
    assert isinstance(native, ParticleAdapter)
    metadata = native.load_export(export, model_identity=other_runtime.identity)
    assert metadata['weights'] == 'live' and metadata['step'] == 2
    assert metadata['manifest_sha256'] == 'b' * 64
    assert metadata['normalization']['cache_fingerprint'] == trainer.responses.identity()['index']['fingerprint']
    assert json.loads(export.with_suffix('.pg09.json').read_text())['weights'] == 'live'
    other_runtime.mixer.add('inference', native)
    with other_runtime.mixer.scales({'inference': 1.}):
        rec = cache[0]
        actual = other_runtime.predict(rec['latent'], rec['timestep'], rec['embedding'])
    assert torch.equal(actual, export_prediction)
    with pytest.raises(FileExistsError):
        trainer.export(export)
    assert not torch.cuda.is_initialized()
    trainer.close()
    resumed.close()
    runtime.mixer.close()
    other_runtime.mixer.close()


def test_cache_features_native_geometry_and_training_only(tmp_path):
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    provider = CachedResponses(cache, runtime, 'finalform')
    conditions = provider.conditions(torch.tensor([0, 1]))
    assert conditions.shape == (2, 12)
    assert conditions[0, -4:].tolist() == pytest.approx([.5, 0., 0., 1.])
    assert conditions[1, -4:].tolist() == pytest.approx([.4, 1 / 9, 1., 1.])
    # Display IDs/seeds never alter conditioning.
    cache[0]
    cache.shards['0.pt']['row']['id'] = 'different-name'
    cache.shards['0.pt']['seed'] = 999
    assert torch.equal(conditions, provider.conditions(torch.tensor([0, 1])))
    row = provider[0]
    row.target.add_(1.)
    assert not torch.equal(row.target, provider[0].target)
    cache.index['identity']['split'] = 'dev'
    with pytest.raises(ValueError, match='training cache'):
        CachedResponses(cache, runtime, 'finalform')
    cache.index['identity']['split'] = 'train'
    # Constructor owns a private reader, so caller memory mutations cannot
    # change the frozen field or its conditions.
    cache.shards['0.pt']['records'][0]['positive'] = cache[0]['positive'].flatten(1)
    assert CachedResponses(cache, runtime, 'finalform').response_shape == (1, 4, 1, 2, 2)
    snapshot = provider.cache
    snapshot.shards['0.pt']['records'][0]['positive'] = snapshot[0]['positive'].flatten(1)
    snapshot.offsets.reverse()
    assert provider[0].target.shape == (1, 4, 1, 2, 2)
    assert torch.equal(conditions, provider.conditions(torch.tensor([0, 1])))
    runtime.mixer.close()


def test_reject_corrupt_cache_even_when_original_verification_disabled(tmp_path):
    runtime = runtime_fixture()
    cache_fixture(tmp_path / 'cache', runtime)
    with (tmp_path / 'cache/0.pt').open('ab') as file:
        file.write(b'corruption')
    cache = TargetCache(tmp_path / 'cache', verify=False)
    with pytest.raises(ValueError, match='Corrupt PG09 target shard'):
        CachedResponses(cache, runtime, 'finalform')
    runtime.mixer.close()


def test_resume_rejects_recipe_change_and_unhooks_failed_binding(tmp_path):
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    trainer = PG09Trainer(runtime, cache, tmp_path / 'run', 'finalform', total_steps=4, critic_width=8)
    trainer.update()
    trainer.save()
    trainer.close()
    assert 'finalform' not in runtime.mixer.adapters
    with pytest.raises(ValueError, match='checkpoint objective, recipe'):
        PG09Trainer(runtime, cache, tmp_path / 'run', 'finalform', total_steps=5, critic_width=8)
    assert 'finalform' not in runtime.mixer.adapters
    assert 'finalform' not in runtime.mixer.strengths
    runtime.mixer.close()


@pytest.mark.parametrize('kind', ['traversal', 'absolute', 'symlink'])
def test_provider_contains_cache_shard_paths(tmp_path, kind):
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    outside = tmp_path / 'outside.pt'
    outside.write_bytes((tmp_path / 'cache/0.pt').read_bytes())
    if kind == 'traversal':
        cache.index['shards'][0]['path'] = '../outside.pt'
    elif kind == 'absolute':
        cache.index['shards'][0]['path'] = str(outside)
    else:
        (tmp_path / 'cache/link.pt').symlink_to(outside)
        cache.index['shards'][0]['path'] = 'link.pt'
    with pytest.raises(ValueError, match='escapes the cache directory'):
        CachedResponses(cache, runtime, 'finalform')
    runtime.mixer.close()


def test_external_native_admission_requires_exact_teacher_parity(tmp_path):
    # Exercise admission with a device descriptor only; no CUDA allocation.
    trainer = PG09Trainer.__new__(PG09Trainer)
    trainer.runtime = SimpleNamespace(device=torch.device('cuda:0'))
    trainer.responses = SimpleNamespace(response_shape=(1, 4, 1, 2, 2))
    trainer.directory = tmp_path
    trainer._frozen_identity = dict(runtime=dict(model='verified-native'), index=dict(fingerprint='a' * 64))
    trainer._native_admitted = False
    trainer._native_admission = lambda request: False
    with pytest.raises(ValueError, match='teacher parity'):
        trainer._admit_native_update()
    assert not trainer._native_admitted and not (tmp_path / 'pg09-admission.json').exists()
    trainer._native_admission = lambda request: True
    trainer._admit_native_update()
    proof = json.loads((tmp_path / 'pg09-admission.json').read_text())
    assert proof['external_admission'] and proof['cache_fingerprint'] == 'a' * 64
    assert not torch.cuda.is_initialized()


def test_public_cache_snapshots_cannot_mutate_the_frozen_response_law(tmp_path):
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    trainer = PG09Trainer(runtime, cache, tmp_path / 'run', 'finalform', total_steps=2, critic_width=8)
    provider = trainer.responses
    original = provider[0]
    original_conditions = provider.conditions(torch.tensor([0, 1]))
    original_identity = provider.identity()
    for snapshot in (provider.cache, trainer.cache, cache):
        snapshot[0]
        snapshot.shards['0.pt']['records'][0]['positive'].add_(10.)
        snapshot.shards['0.pt']['embedding'].mul_(0.)
        snapshot.index['fingerprint'] = 'c' * 64
        snapshot.offsets.reverse()
        snapshot.max_shards = 999
        snapshot.shards.clear()
    assert torch.equal(original.target, provider[0].target)
    assert torch.equal(original.row.neutral, provider[0].row.neutral)
    assert torch.equal(original_conditions, provider.conditions(torch.tensor([0, 1])))
    assert provider.identity() == original_identity
    assert provider.cache is not trainer.cache
    assert provider.cache.max_shards == 4
    trainer.close()
    runtime.mixer.close()


@pytest.mark.parametrize('state', ['resident', 'evicted', 'unloaded'])
@pytest.mark.parametrize('mutation', ['replacement', 'same_inode_same_size', 'deleted'])
@pytest.mark.parametrize('access', ['response', 'conditions'])
def test_selected_shard_changes_fail_without_changing_frozen_identity(tmp_path, state, mutation, access):
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    cache.max_shards = 1
    provider = CachedResponses(cache, runtime, 'finalform')
    selected = 1 if state == 'unloaded' else 0
    if state == 'evicted':
        provider[1]
        assert '0.pt' not in provider.cache.shards
    original_identity = provider.identity()
    path = tmp_path / 'cache' / f'{selected}.pt'
    before = path.stat()
    if mutation == 'deleted':
        path.unlink()
    else:
        data = torch.load(path, map_location='cpu', weights_only=True)
        old_shape, old_fingerprint = data['records'][0]['positive'].shape, data['fingerprint']
        data['records'][0]['positive'].add_(1.)
        if mutation == 'replacement':
            save_tensor_file(path, data)
        else:
            torch.save(data, path)
            assert path.stat().st_ino == before.st_ino
            assert path.stat().st_size == before.st_size
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        actual = torch.load(path, weights_only=True)
        assert actual['fingerprint'] == old_fingerprint and actual['records'][0]['positive'].shape == old_shape
    with pytest.raises(ValueError, match='PG09 frozen target shard'):
        if access == 'response':
            provider[selected]
        else:
            provider.conditions(torch.tensor([selected]))
    assert provider.identity() == original_identity
    assert len(provider.cache.shards) <= 1
    runtime.mixer.close()


def test_lazy_loading_hashes_only_misses_and_preserves_bounded_lru(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    cache.max_shards = 1
    provider = CachedResponses(cache, runtime, 'finalform')
    calls = []
    original_hash = provider._hash_open_file
    def counted(file):
        calls.append(file.name)
        return original_hash(file)
    monkeypatch.setattr(provider, '_hash_open_file', counted)
    provider[0]
    provider.conditions(torch.tensor([0, 0]))
    assert calls == []
    provider[1]
    provider.conditions(torch.tensor([1]))
    assert len(calls) == 1 and list(provider.cache.shards) == ['1.pt']
    provider[0]
    assert len(calls) == 2 and list(provider.cache.shards) == ['0.pt']
    runtime.mixer.close()


def test_runtime_mutations_fail_but_returned_identity_snapshot_is_isolated(tmp_path):
    runtime = runtime_fixture()
    runtime.identity['nested'] = dict(weights=['verified'])
    cache = cache_fixture(tmp_path / 'cache', runtime)
    provider = CachedResponses(cache, runtime, 'finalform')
    identity = provider.identity()
    conditions = provider.conditions(torch.tensor([0, 1]))
    identity['runtime']['nested']['weights'][0] = 'changed-snapshot'
    assert provider.identity()['runtime']['nested']['weights'] == ['verified']
    assert torch.equal(provider.conditions(torch.tensor([0, 1])), conditions)
    runtime.identity['nested']['weights'][0] = 'mutated'
    for read in (provider.identity, lambda: provider[0], lambda: provider.conditions(torch.tensor([0]))):
        with pytest.raises(ValueError, match='runtime identity or native timestep scale changed'):
            read()
    runtime.identity['nested']['weights'][0] = 'verified'
    runtime.pipe.scheduler.config.num_train_timesteps = 1
    for read in (provider.identity, lambda: provider[0], lambda: provider.conditions(torch.tensor([0]))):
        with pytest.raises(ValueError, match='runtime identity or native timestep scale changed'):
            read()
    runtime.mixer.close()


def test_shard_replaced_during_initial_hash_is_not_adopted_as_verified(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    original_hash = CachedResponses._hash_open_file
    def replace_after_hash(file):
        result = original_hash(file)
        path = tmp_path / 'cache/0.pt'
        if str(file.name) == str(path):
            data = torch.load(path, weights_only=True)
            data['records'][0]['positive'].add_(3.)
            save_tensor_file(path, data)
        return result
    monkeypatch.setattr(CachedResponses, '_hash_open_file', staticmethod(replace_after_hash))
    with pytest.raises(ValueError, match='changed during verification'):
        CachedResponses(cache, runtime, 'finalform')
    runtime.mixer.close()


def test_shard_replaced_during_lazy_load_is_rejected_before_lru_commit(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    provider = CachedResponses(cache, runtime, 'finalform')
    path = tmp_path / 'cache/1.pt'
    original_load = torch.load
    def replace_after_load(file, *args, **kwargs):
        data = original_load(file, *args, **kwargs)
        if getattr(file, 'name', None) == str(path):
            replacement = deepcopy(data)
            replacement['records'][0]['positive'].add_(3.)
            save_tensor_file(path, replacement)
        return data
    monkeypatch.setattr(torch, 'load', replace_after_load)
    with pytest.raises(ValueError, match='PG09 frozen target shard changed'):
        provider[1]
    assert '1.pt' not in provider.cache.shards
    runtime.mixer.close()
