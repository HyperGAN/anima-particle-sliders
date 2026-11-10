"""NEW UNRUN software cases, no pretrained model or completed proof replay.

The synthetic two-row cache below is software test data, not native teachers or
pretrained/effect evidence. All actual execution belongs to Root's bounded job.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler

from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.cache import TargetCache, save_tensor_file
from lumen_studio.contracts import atomic_json, digest, file_hash
from lumen_studio.particles import Mixer
from lumen_studio.pg09_training import NativeFieldCritic
from lumen_studio_fresh95 import open_training, open_inference
from lumen_studio_fresh95 import policy, training


def runtime_fixture():
    # New random tiny software runtime; absolutely no pretrained loader.
    with torch.random.fork_rng():
        torch.manual_seed(9)
        runtime = TurboRuntime.__new__(TurboRuntime)
        runtime.transformer = CosmosTransformer3DModel(in_channels=4, out_channels=4,
            num_attention_heads=2, attention_head_dim=32, num_layers=1, mlp_ratio=2,
            text_embed_dim=8, adaln_lora_dim=4, max_size=(2, 4, 4),
            patch_size=(1, 1, 1), extra_pos_embed_type=None).to(torch.bfloat16).eval().requires_grad_(False)
    runtime.device, runtime.dtype, runtime.scale = torch.device('cpu'), torch.bfloat16, 1
    runtime.identity = {'model': 'CPU random tiny Cosmos; no pretrained weights', 'sha256': 'a' * 64}
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
        # Explicit synthetic finite targets; NO teacher runtime.predict calls.
        record = dict(latent=latent, neutral=torch.zeros_like(latent), positive=torch.full_like(latent, .1),
            timestep=500. - index * 100, position=index, trajectory=('neutral', 'positive')[index])
        destination = path / f'{index}.pt'
        save_tensor_file(destination, dict(fingerprint=fingerprint, records=[record],
            embedding=torch.full((1, index + 2, 8), .2 + index * .1), row={'id': f'test{index}'}, seed=index))
        shards.append(dict(path=destination.name, sha256=file_hash(destination), count=1))
    atomic_json(path / 'index.json', dict(identity=identity, fingerprint=fingerprint, shards=shards))
    return TargetCache(path)



def cleanup_fault_fixture(monkeypatch):
    # New owned-cleanup behavior only; no model arithmetic or old tests.
    class Handle:
        def __init__(self, fail_once=False):
            self.fail_once, self.calls = fail_once, 0
        def remove(self):
            self.calls += 1
            if self.fail_once and self.calls == 1:
                raise RuntimeError('NEW synthetic handle removal failure')

    for hook_fails, restore_fails in ((True, False), (False, True)):
        calls = []
        def restore():
            calls.append('restore')
            if restore_fails and len(calls) == 1:
                raise RuntimeError('NEW synthetic native callback failure')
        with monkeypatch.context() as scoped:
            scoped.setattr(training, 'native_cleanup', lambda root: restore)
            mixer = SimpleNamespace(adapters={}, strengths={}, handles=[])
            runtime = SimpleNamespace(mixer=mixer, transformer=object())
            owner = training.Attachment(runtime, 'owned')
            adapter, handle = object(), Handle(hook_fails)
            owner.adapter = mixer.adapters['owned'] = adapter
            mixer.strengths['owned'] = 0
            mixer.handles.append(handle)
            with pytest.raises(BaseExceptionGroup):
                owner.close()
            assert owner.closing and owner.cleanup_pending and not owner.closed
            assert mixer.handles == ([handle] if hook_fails else [])
            assert 'owned' not in mixer.adapters
            # Pending close blocks both public update and inference contexts.
            stub = training.FreshTrainer.__new__(training.FreshTrainer)
            stub._attachment = owner
            with pytest.raises(RuntimeError):
                stub.update()
            from lumen_studio_fresh95.inference import InferenceOwner
            with pytest.raises(RuntimeError):
                with InferenceOwner.scaled(owner, .5):
                    raise AssertionError('closing owner yielded native inference')
            owner.close()
            assert owner.closed and not owner.closing and not owner.cleanup_pending
            assert not mixer.handles
            assert handle.calls == (2 if hook_fails else 1)
            assert len(calls) == (2 if restore_fails else 1)
            owner.close()  # No replay of any successful cleanup step.
            assert handle.calls == (2 if hook_fails else 1)
            assert len(calls) == (2 if restore_fails else 1)

def arguments(runtime, cache, directory, cohort='fresh-software-case'):
    return dict(runtime=runtime, cache=cache, directory=directory, variation='finalform',
        mode='fresh', cohort_id=cohort, seed=7, batch_size=2, total_steps=4,
        critic_width=8, ema_decay=None)


def test_new_fresh_policy_precedes_optimizers_and_keeps_identities(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    foreign = training.ParticleAdapter(runtime.transformer)
    runtime.mixer.add('foreign', foreign)
    runtime.mixer.strengths['foreign'] = .25
    handles = tuple(runtime.mixer.handles)
    checkpoint_settings = [(m, {name: (name in vars(m), getattr(m, name, None))
        for name in ('gradient_checkpointing', '_gradient_checkpointing_func')})
        for m in runtime.transformer.modules()]
    seen = {}
    original_initialize, original_game = policy.initialize, training.ConditionalPG095

    def initialize(*args, **kwargs):
        adapter, critic = args[1:3]
        seen['before_ids'] = [id(p) for p in adapter.parameters()] + [id(p) for p in critic.parameters()]
        seen['before'] = {row['name']: policy.tensor_sha(row['parameter']) for row in
            policy.policies('anima', adapter, critic, particles=adapter.particles)}
        receipt = original_initialize(*args, **kwargs)
        seen['receipt'] = receipt
        seen['after_ids'] = [id(p) for p in adapter.parameters()] + [id(p) for p in critic.parameters()]
        return receipt

    def game(backend, responses, critic, **kwargs):
        assert seen['receipt'] == backend.metadata()['fresh_initialization']['receipt']
        assert backend.particles is backend.network.particles is kwargs['particles']
        seen['before_optimizer'] = True
        return original_game(backend, responses, critic, **kwargs)

    monkeypatch.setattr(policy, 'initialize', initialize)
    monkeypatch.setattr(training, 'ConditionalPG095', game)
    trainer = open_training(**arguments(runtime, cache, tmp_path / 'entry'))
    try:
        assert seen['before_optimizer'] and seen['before_ids'] == seen['after_ids']
        assert trainer.step == 0 and trainer.entry_FULL.bytes > 0 and trainer.entry_RAW.bytes > 0
        assert trainer.initialization['parameter_identity_preserved'] is True
        roles = {row['role'] for row in trainer.initialization['parameter_policies']}
        assert {'protected-zero-UP', 'declared-host-scale-down-QR', 'declared-bank-R2', 'constructor-law-KEEP'} <= roles
        for row in trainer.initialization['parameter_policies']:
            assert row['before_sha256'] == seen['before'][row['name']]
            if row['role'] in ('protected-zero-UP', 'constructor-law-KEEP'):
                assert row['after_sha256'] == row['before_sha256']
        assert all(torch.count_nonzero(branch.up.weight).item() == 0 for branch in trainer.adapter.branches)
        assert trainer.backend.backend.adapter is trainer.adapter
        assert trainer.game.network is trainer.backend.network
    finally:
        trainer.close()
    assert runtime.mixer.adapters == {'foreign': foreign}
    assert runtime.mixer.strengths == {'foreign': .25}
    assert tuple(runtime.mixer.handles) == handles
    for module, attributes in checkpoint_settings:
        for name, (present, value) in attributes.items():
            assert (name in vars(module)) is present
            assert getattr(module, name, None) is value
    runtime.mixer.close()
    cleanup_fault_fixture(monkeypatch)


def test_new_policy_determinism_and_critic_constructor_law(tmp_path):
    first_runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', first_runtime)
    # Original seeded bank/branches -> critic order, before any scoped QR/bank.
    with torch.random.fork_rng():
        torch.manual_seed(7)
        original_adapter = training.ParticleAdapter(first_runtime.transformer)
        expected_critic = NativeFieldCritic(4, 12, 8)
    first = open_training(**arguments(first_runtime, cache, tmp_path / 'first', 'first'))
    second = None
    try:
        second = open_training(**arguments(runtime_fixture(), cache, tmp_path / 'second', 'second'))
        assert training.equal_state(first.adapter.state_dict(), second.adapter.state_dict())
        assert training.equal_state(first.game.critic.state_dict(), expected_critic.state_dict())
        assert training.equal_state(first.game.critic.state_dict(), second.game.critic.state_dict())
        assert policy.tensor_sha(first.adapter.particles) != policy.tensor_sha(original_adapter.particles)
        # QR is tested as the declared official law, not as a changed forward.
        for branch in first.adapter.branches:
            weight = branch.down.weight.detach()
            bound = 1 / (branch.down.in_features ** .5)
            target_variance = (2 * bound) ** 2 / 12
            gram = weight @ weight.T
            assert torch.allclose(gram, torch.eye(gram.shape[0]) * branch.down.in_features * target_variance,
                                  atol=2e-6, rtol=2e-5)
        assert all(torch.count_nonzero(b.up.weight).item() == 0 for b in first.adapter.branches)
    finally:
        first.close()
        if second is not None:
            second.close()


def test_new_FULL_resume_and_native_RAW_loading_never_initialize(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    trainer = open_training(**arguments(runtime, cache, tmp_path / 'initial'))
    try:
        trainer.update()  # ONE new tiny software step to populate optimizer/RNG state.
        saved = trainer.save()
        raw = trainer.export(tmp_path / 'RAW1.safetensors')
        expected, receipt = trainer.game.state_dict(), deepcopy(trainer.initialization)
    finally:
        trainer.close()

    def forbidden(*args, **kwargs):
        raise AssertionError('resume/inference called fresh scoped initializer')

    monkeypatch.setattr(policy, 'initialize', forbidden)
    options = arguments(runtime_fixture(), cache, tmp_path / 'resume')
    options.update(mode='resume', resume=saved)
    resumed = open_training(**options)
    try:
        assert resumed.step == 1 and resumed.initialization == receipt
        assert training.equal_state(resumed.game.state_dict(), expected)
        assert resumed.adapter.particles is resumed.backend.network.particles
    finally:
        resumed.close()
    inference_runtime = runtime_fixture()
    with_owner = open_inference(inference_runtime, 'fresh-inference', raw=raw)
    try:
        assert not hasattr(with_owner, 'game') and with_owner.sidecar['initialization'] == receipt
        with with_owner.scaled(.5) as staged:
            assert staged is inference_runtime and staged.mixer.strengths['fresh-inference'] == .5
        assert inference_runtime.mixer.strengths['fresh-inference'] == 0
    finally:
        with_owner.close()
    assert not inference_runtime.mixer.adapters and not inference_runtime.mixer.handles


def test_new_entry_refuses_legacy_or_mixed_intent_before_components(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    legacy = tmp_path / 'old-bare-game.pt'
    torch.save({'schema': 'conditional-pg09-v1', 'completed_steps': 0}, legacy)

    def forbidden(*args, **kwargs):
        raise AssertionError('refused intent reached a component constructor')

    monkeypatch.setattr(training, 'ParticleAdapter', forbidden)
    for patch in ({'mode': 'fresh', 'resume': training.file_pin(legacy)},
                  {'mode': 'resume'}, {'mode': 'resume', 'resume': training.file_pin(legacy)},
                  {'ema_decay': .9}):
        kwargs = arguments(runtime, cache, tmp_path / 'refused')
        kwargs.update(patch)
        with pytest.raises(ValueError):
            open_training(**kwargs)
    assert not runtime.mixer.adapters and not (tmp_path / 'refused').exists()
