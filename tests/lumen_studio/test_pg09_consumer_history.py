"""Disk resume contracts through Anima's real Cosmos/PG09 consumer binding."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from lumen_studio import pg09_training
from lumen_studio.cache import save_tensor_file
from lumen_studio.pg09_training import PG09Trainer

from test_pg09_training import cache_fixture, equal, runtime_fixture


@pytest.fixture(scope='module')
def started_consumer(tmp_path_factory):
    """Restore a step-one disk packet into a distinct, live step-two consumer."""
    directory = tmp_path_factory.mktemp('anima-pg09-history')
    runtime = runtime_fixture()
    cache = cache_fixture(directory / 'cache', runtime)
    trainer = PG09Trainer(runtime, cache, directory / 'run', 'finalform',
                          total_steps=5, critic_width=8)
    trainer.update()
    packet = trainer.game.state_dict()
    trainer.update()
    assert packet['completed_steps'] == 1 and trainer.step == 2
    try:
        yield trainer, packet
    finally:
        trainer.close()
        runtime.mixer.close()


@pytest.mark.parametrize('phase', ['g', 'd'])
@pytest.mark.parametrize('corruption', [
    'missing_slot', 'extra_slot', 'deleted_history', 'emptied_history',
])
def test_started_consumer_rejects_corrupt_optimizer_history_atomically(
        tmp_path, started_consumer, phase, corruption):
    trainer, saved = started_consumer
    packet = deepcopy(saved)
    optimizer = packet['opt_' + phase]
    # Select an actual started parameter; no synthetic history/schema is added.
    key = next(key for key, slots in optimizer['state'].items() if slots)
    slots = optimizer['state'][key]
    # The product selects BISPF optimized, whose DualNorm histories are step
    # records rather than Adam moments. Corrupt its actual native slot fields.
    assert 'step' in slots
    if corruption == 'missing_slot':
        del slots['step']
    elif corruption == 'extra_slot':
        slots['unexpected_history'] = 0
    elif corruption == 'deleted_history':
        del optimizer['state'][key]
    else:
        optimizer['state'][key] = {}
    path = tmp_path / 'corrupt-resume.pt'
    save_tensor_file(path, packet)
    before = trainer.game.state_dict()
    with pytest.raises(ValueError, match='optimizer|history|histories'):
        trainer.restore(path)
    # Covers weights, both optimizers, bank/prior, regularizer, every RNG,
    # sampler order/cursors/exposures and step count rather than only weights.
    assert equal(before, trainer.game.state_dict())
    assert trainer.step == 2
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('phase', ['g', 'd'])
def test_product_native_public_bispf_consumer_rejects_negative_adam_moment_atomically(tmp_path, phase):
    from particle_sliders.pg09 import ConditionalPG09

    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    trainer = PG09Trainer(runtime, cache, tmp_path / 'run', 'finalform',
                          total_steps=3, critic_width=8)
    try:
        # exp_avg_sq is absent from the product's default optimized law. Use
        # the public original BISPF/Adam preset with the actual product bank,
        # backend, native Cosmos fields and critic to exercise real moments.
        trainer.game = ConditionalPG09(trainer.backend, trainer.responses,
            trainer.game.critic, particles=trainer.adapter.particles,
            preset='bispf', response_domain='continuous_field',
            batch_size=2, seed=7, total_steps=3, sampling='balanced')
        trainer.update()
        packet = trainer.game.state_dict()
        optimizer = packet['opt_' + phase]
        key = next(key for key, slots in optimizer['state'].items() if 'exp_avg_sq' in slots)
        moment = optimizer['state'][key]['exp_avg_sq']
        assert moment.numel() and (moment >= 0).all()
        moment.reshape(-1)[0] = -1.
        path = tmp_path / 'negative-moment.pt'
        save_tensor_file(path, packet)
        trainer.update()
        before = trainer.game.state_dict()
        with pytest.raises(ValueError, match='second moment is negative'):
            trainer.restore(path)
        assert equal(before, trainer.game.state_dict())
        assert trainer.step == 2
        assert not torch.cuda.is_initialized()
    finally:
        trainer.close()
        runtime.mixer.close()


@pytest.fixture(scope='module')
def sparse_adam_consumer(tmp_path_factory):
    """Earn a sparse D clock through actual native-field updates, without edits."""
    from particle_sliders.pg09 import ConditionalPG09

    class PartiallyUsedCritic(pg09_training.NativeFieldCritic):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.optional = nn.Parameter(torch.ones(1))
            self.register_buffer('optional_active', torch.tensor(False))

        def forward(self, response, condition):
            score = super().forward(response, condition)
            if bool(self.optional_active):
                score = score + self.optional * response.flatten(1).mean(1)
            return score

    directory = tmp_path_factory.mktemp('anima-pg09-sparse-adam-clock')
    runtime = runtime_fixture()
    cache = cache_fixture(directory / 'cache', runtime)
    # Supply a conditional critic at construction; the production default
    # preset/interface is unchanged. Original public BISPF owns real Adam
    # moments and is supplemental to the optimized product-law coverage.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pg09_training, 'NativeFieldCritic', PartiallyUsedCritic)
        trainer = PG09Trainer(runtime, cache, directory / 'run', 'finalform',
                              total_steps=4, critic_width=8)
    trainer.game = ConditionalPG09(trainer.backend, trainer.responses,
        trainer.game.critic, particles=trainer.adapter.particles,
        preset='bispf', response_domain='continuous_field',
        batch_size=2, seed=7, total_steps=4, sampling='balanced')
    try:
        trainer.update()
        trainer.game.critic.optional_active.fill_(True)
        trainer.update()
        packet = trainer.game.state_dict()
        optional = next(key
            for saved, actual in zip(packet['opt_d']['param_groups'], trainer.game.opt_d.param_groups)
            for key, parameter in zip(saved['params'], actual['params'])
            if parameter is trainer.game.critic.optional)
        assert packet['completed_steps'] == 2
        assert packet['opt_d']['state'][optional]['step'].item() == 1
        assert all(slots['step'].item() == 2 for slots in packet['opt_g']['state'].values())
        metrics = trainer.update()
        assert trainer.step == 3
        yield trainer, packet, optional, metrics
    finally:
        trainer.close()
        runtime.mixer.close()


@pytest.mark.parametrize('phase,count', [('g', 0), ('g', 1), ('g', 3), ('d', 0), ('d', 3)])
def test_product_native_public_bispf_rejects_invalid_adam_update_clock_atomically(
        tmp_path, sparse_adam_consumer, phase, count):
    trainer, saved, optional, _ = sparse_adam_consumer
    packet = deepcopy(saved)
    optimizer = packet['opt_' + phase]
    key = optional if phase == 'd' else next(iter(optimizer['state']))
    optimizer['state'][key]['step'].fill_(count)
    path = tmp_path / 'invalid-adam-clock.pt'
    save_tensor_file(path, packet)
    before = trainer.game.state_dict()
    with pytest.raises(ValueError, match='Adam step count differs from completed updates'):
        trainer.restore(path)
    assert equal(before, trainer.game.state_dict())
    assert trainer.step == 3
    assert not torch.cuda.is_initialized()


def test_product_native_public_bispf_sparse_adam_clock_disk_resume_is_exact(
        tmp_path, sparse_adam_consumer):
    trainer, packet, optional, metrics = sparse_adam_consumer
    assert packet['completed_steps'] == 2
    assert packet['opt_d']['state'][optional]['step'].item() == 1
    path = tmp_path / 'valid-sparse-adam-clock.pt'
    save_tensor_file(path, packet)
    before = trainer.game.state_dict()
    try:
        # Current live state is step three. Rewind through disk and reproduce
        # its unbroken next-update metrics and whole state from the honest packet.
        trainer.restore(path)
        assert equal(packet, trainer.game.state_dict())
        assert equal(metrics, trainer.update())
        assert equal(before, trainer.game.state_dict())
        assert not torch.cuda.is_initialized()
    finally:
        trainer.game.load_state_dict(before)


def test_consumer_requires_optimizer_history_schema_atomically(tmp_path, started_consumer):
    trainer, saved = started_consumer
    packet = deepcopy(saved)
    del packet['optimizer_history_schema']
    path = tmp_path / 'missing-schema.pt'
    save_tensor_file(path, packet)
    before = trainer.game.state_dict()
    with pytest.raises(ValueError, match='history presence'):
        trainer.restore(path)
    assert equal(before, trainer.game.state_dict())


def test_initial_empty_consumer_disk_resume_then_actual_update(tmp_path):
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    trainer = PG09Trainer(runtime, cache, tmp_path / 'run', 'finalform',
                          total_steps=3, critic_width=8)
    resumed = other_runtime = None
    try:
        trainer.save()
        initial = trainer.game.state_dict()
        assert initial['completed_steps'] == 0
        for phase in ('g', 'd'):
            assert initial['opt_' + phase]['state'] == {}
            assert initial['optimizer_history_schema'][phase]
            assert all(value is None for value in initial['optimizer_history_schema'][phase].values())
        metrics = trainer.update()
        expected = trainer.game.state_dict()
        other_runtime = runtime_fixture()
        # Constructor uses the consumer's disk restore, not a direct core fixture.
        resumed = PG09Trainer(other_runtime, cache, tmp_path / 'run', 'finalform',
                              total_steps=3, critic_width=8)
        assert equal(initial, resumed.game.state_dict())
        assert equal(metrics, resumed.update())
        assert equal(expected, resumed.game.state_dict())
        assert not torch.cuda.is_initialized()
    finally:
        trainer.close()
        if resumed is not None:
            resumed.close()
        runtime.mixer.close()
        if other_runtime is not None:
            other_runtime.mixer.close()


def test_genuinely_unused_critic_history_survives_consumer_resume(tmp_path, monkeypatch):
    class CriticWithUnusedParameter(pg09_training.NativeFieldCritic):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.unused = nn.Parameter(torch.ones(2))

    # The public core permits a critic trainable which receives no gradient.
    # Supply that critic through the real product constructor; neither the
    # optimizer history, its schema nor completed-step counts are fabricated.
    monkeypatch.setattr(pg09_training, 'NativeFieldCritic', CriticWithUnusedParameter)
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'cache', runtime)
    trainer = PG09Trainer(runtime, cache, tmp_path / 'run', 'finalform',
                          total_steps=3, critic_width=8)
    resumed = other_runtime = None
    try:
        trainer.update()
        trainer.save()
        packet = trainer.game.state_dict()
        unused = next(key
            for saved, actual in zip(packet['opt_d']['param_groups'], trainer.game.opt_d.param_groups)
            for key, parameter in zip(saved['params'], actual['params'])
            if parameter is trainer.game.critic.unused)
        assert trainer.game.critic.unused.grad is None
        assert unused not in packet['opt_d']['state']
        assert packet['optimizer_history_schema']['d'][unused] is None
        metrics = trainer.update()
        expected = trainer.game.state_dict()
        other_runtime = runtime_fixture()
        resumed = PG09Trainer(other_runtime, cache, tmp_path / 'run', 'finalform',
                              total_steps=3, critic_width=8)
        assert equal(packet, resumed.game.state_dict())
        assert equal(metrics, resumed.update())
        assert equal(expected, resumed.game.state_dict())
        assert not torch.cuda.is_initialized()
    finally:
        trainer.close()
        if resumed is not None:
            resumed.close()
        runtime.mixer.close()
        if other_runtime is not None:
            other_runtime.mixer.close()
