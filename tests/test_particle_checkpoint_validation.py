"""Malformed native particle reloads must leave all live state unchanged."""
from copy import deepcopy
import math

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lumen_studio.alpha_adapter import AlphaParticleAdapter, ALPHA_FORMAT, alpha_values
from lumen_studio.checkpoint_validation import validated_tensor_state
from lumen_studio.contracts import canonical
from lumen_studio.particles import ParticleAdapter, ARCHITECTURE, FORMAT
from single_particle import SingleParticleAdapter, SINGLE_ARCH, ORIGINAL_FORMAT, ALPHA_FORMAT as SINGLE_ALPHA_FORMAT


KINDS = ('native', 'native-alpha-reader', 'alpha', 'single-original', 'single-alpha')


def checkpoint_fixture(kind):
    model = nn.Module()
    model.to_q = nn.Linear(4, 6, bias=False)
    model.to_v = nn.Linear(4, 6, bias=False)
    cls = (SingleParticleAdapter if kind.startswith('single') else
           AlphaParticleAdapter if kind in ('alpha', 'native-alpha-reader') else ParticleAdapter)
    torch.manual_seed(301)
    source = cls(model)
    for branch in source.branches:
        nn.init.normal_(branch.up.weight)
    state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    metadata = dict(format=FORMAT, architecture=ARCHITECTURE, targets=source.names,
        model_identity={'model': 'test'}, weights='ema', step=1200,
        normalization={'test': True}, manifest_sha256='a' * 64)
    if kind in ('alpha', 'single-original', 'single-alpha'):
        alpha = 8. if kind == 'single-original' else 12.
        state.update({f'branches.{i}.alpha': torch.tensor(alpha, dtype=torch.float64)
                      for i in range(len(source.branches))})
        metadata.update(format=(ORIGINAL_FORMAT if kind == 'single-original' else
            SINGLE_ALPHA_FORMAT if kind == 'single-alpha' else ALPHA_FORMAT),
            architecture=SINGLE_ARCH if kind.startswith('single') else ARCHITECTURE,
            scaling='alpha/rank', network_alpha=alpha)
    torch.manual_seed(702)
    target = cls(model)
    if hasattr(target, 'alphas'):
        target.alphas = [3., 5.]
    return target, state, deepcopy(metadata)


def write_checkpoint(tmp_path, state, metadata):
    path = tmp_path / 'checkpoint.safetensors'
    save_file(state, str(path), metadata={'anima': canonical(metadata)})
    return path


def snapshot(adapter):
    return deepcopy(adapter.state_dict()), deepcopy(getattr(adapter, 'alphas', None))


def assert_unchanged(adapter, before):
    state, alphas = before
    assert all(torch.equal(value, adapter.state_dict()[name]) for name, value in state.items())
    assert getattr(adapter, 'alphas', None) == alphas


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('corruption', ['missing', 'unexpected', 'shape', 'float_dtype', 'integer_dtype', 'nan', 'inf'])
def test_corrupt_later_tensor_cannot_partially_replace_live_particle_state(tmp_path, kind, corruption):
    adapter, state, metadata = checkpoint_fixture(kind)
    key = 'branches.1.up.weight'
    if corruption == 'missing':
        del state[key]
    elif corruption == 'unexpected':
        state['unexpected.weight'] = torch.ones(1)
    elif corruption == 'shape':
        state[key] = torch.ones(7, 8)
    elif corruption == 'float_dtype':
        state[key] = state[key].double()
    elif corruption == 'integer_dtype':
        state[key] = state[key].long()
    else:
        state[key][0, 0] = math.nan if corruption == 'nan' else math.inf
    before = snapshot(adapter)
    with pytest.raises(ValueError, match='Checkpoint tensor'):
        adapter.load_export(write_checkpoint(tmp_path, state, metadata), model_identity={'model': 'test'})
    assert_unchanged(adapter, before)


@pytest.mark.parametrize('kind', ('alpha', 'single-original', 'single-alpha'))
@pytest.mark.parametrize('corruption', ['missing', 'unexpected', 'shape', 'float_dtype', 'integer_dtype', 'nan', 'inf', 'value'])
def test_corrupt_alpha_cannot_replace_weights_or_existing_alphas(tmp_path, kind, corruption):
    adapter, state, metadata = checkpoint_fixture(kind)
    key = 'branches.1.alpha'
    if corruption == 'missing':
        del state[key]
    elif corruption == 'unexpected':
        state['unused.alpha'] = torch.tensor(8., dtype=torch.float64)
    elif corruption == 'shape':
        state[key] = state[key].reshape(1)
    elif corruption == 'float_dtype':
        state[key] = state[key].float()
    elif corruption == 'integer_dtype':
        state[key] = state[key].long()
    else:
        state[key] = torch.tensor(math.nan if corruption == 'nan' else math.inf if corruption == 'inf' else 0.,
                                  dtype=torch.float64)
    before = snapshot(adapter)
    with pytest.raises(ValueError):
        adapter.load_export(write_checkpoint(tmp_path, state, metadata), model_identity={'model': 'test'})
    assert_unchanged(adapter, before)


@pytest.mark.parametrize('kind', KINDS)
def test_valid_export_loads_exact_weights_and_then_commits_alpha(tmp_path, kind):
    adapter, state, metadata = checkpoint_fixture(kind)
    loaded = adapter.load_export(write_checkpoint(tmp_path, state, metadata), model_identity={'model': 'test'})
    assert loaded == metadata
    for name, value in adapter.state_dict().items():
        assert torch.equal(value, state[name])
    if hasattr(adapter, 'alphas'):
        assert adapter.alphas == [metadata.get('network_alpha', 8.)] * len(adapter.branches)


@pytest.mark.parametrize('kind', ('single-original', 'single-alpha'))
def test_fixed_single_particle_semantics_are_checked_before_any_copy(tmp_path, kind):
    adapter, state, metadata = checkpoint_fixture(kind)
    state['particles'][0, 0] = 1.
    before = snapshot(adapter)
    with pytest.raises(ValueError, match='fixed zero'):
        adapter.load_export(write_checkpoint(tmp_path, state, metadata), model_identity={'model': 'test'})
    assert_unchanged(adapter, before)


def test_boolean_network_alpha_is_not_a_numeric_alpha():
    _, state, metadata = checkpoint_fixture('alpha')
    metadata['network_alpha'] = True
    for name in state:
        if name.endswith('.alpha'):
            state[name] = torch.tensor(1., dtype=torch.float64)
    with pytest.raises(ValueError, match='Network alpha'):
        alpha_values(metadata, state)


def test_low_level_validation_returns_only_module_state_without_copying_it():
    state = {'weight': torch.ones(2, 3), 'branch.alpha': torch.tensor(8., dtype=torch.float64)}
    expected = {'weight': torch.zeros(2, 3)}
    validated = validated_tensor_state(state, expected, scalar_keys=('branch.alpha',))
    assert set(validated) == {'weight'}
    assert validated['weight'] is state['weight']
    assert torch.count_nonzero(expected['weight']) == 0
