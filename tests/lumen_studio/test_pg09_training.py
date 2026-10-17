"""Real tiny Cosmos architecture, native product bank and released PG0.9 on CPU."""
from copy import deepcopy
from importlib.metadata import version
import json
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
    provider.cache.shards['0.pt']['records'][0]['positive'] = provider.cache[0]['positive'].flatten(1)
    with pytest.raises(ValueError, match='native'):
        provider[0]
    provider.cache.offsets.reverse()
    with pytest.raises(ValueError, match='record ordering'):
        provider.conditions(torch.tensor([0]))
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
