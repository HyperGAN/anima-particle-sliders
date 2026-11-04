"""LR selection changes optimizer identity, including exact resume admission."""
from copy import deepcopy

import pytest

from lumen_studio.learning_rates import recipe_learning_rates
from lumen_studio.pg09_training import PG09Trainer
from particle_sliders.lifecycle import CheckpointError, SessionConfig
from test_image_lifecycle import configuration
from test_pg09_training import cache_fixture, equal, runtime_fixture


@pytest.mark.parametrize('value', [0, -1, True, float('nan'), float('inf'), '0.01'])
@pytest.mark.parametrize('name', ['lr', 'd_lr_mult', 'prior_lr_mult'])
def test_bad_learning_rates_refused_before_allocation(name, value):
    with pytest.raises(ValueError, match=name):
        recipe_learning_rates(**{name: value})


def test_direct_api_and_common_configuration_resolve_identical_lrs(tmp_path):
    from lumen_studio.lifecycle import BACKEND
    runtime = runtime_fixture()
    cache = cache_fixture(tmp_path / 'direct-cache', runtime)
    settings = dict(lr=.001, d_lr_mult=1.5, prior_lr_mult=2.5)
    trainer = PG09Trainer(runtime, cache, tmp_path / 'direct', 'finalform',
                         total_steps=4, critic_width=8, **settings)
    recipe = deepcopy(trainer.game.state_dict()['config']['recipe'])
    trainer.update()
    trainer.save()
    trainer.close()
    with pytest.raises(ValueError, match='config|identity|signature|checkpoint'):
        PG09Trainer(runtime, cache, tmp_path / 'direct', 'finalform',
                    total_steps=4, critic_width=8, **dict(settings, lr=.002))
    config = configuration(tmp_path, runtime)
    config = SessionConfig(config.backend, config.cohort, **dict(config.options, **settings))
    with BACKEND.open(config, runtime=runtime) as session:
        assert session.game.state_dict()['config']['recipe'] == recipe
        session.train(stop_after=2)
        packet = session.checkpoint()
    with BACKEND.open(config, runtime=runtime) as resumed:
        resumed.restore(packet)
        assert resumed.game.completed_steps == 2
        assert equal(resumed.checkpoint(), packet)
    changed = SessionConfig(config.backend, config.cohort, **dict(config.options, lr=.002))
    with BACKEND.open(changed, runtime=runtime) as wrong:
        with pytest.raises(CheckpointError):
            wrong.restore(packet)
    runtime.mixer.close()
