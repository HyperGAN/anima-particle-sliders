"""Native release inference must apply one complete, validated LoRA checkpoint."""
import math

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from release_tools.distill import attach_lora


def model_and_state():
    model = nn.Module()
    model.a = nn.Linear(4, 6, bias=False)
    model.z = nn.Linear(4, 6, bias=False)
    state = {name + suffix: torch.randn(shape) for name in ('a', 'z')
             for suffix, shape in (('.lora_A.weight', (8, 4)), ('.lora_B.weight', (6, 8)))}
    state.update({name + '.alpha': torch.tensor(12., dtype=torch.float64) for name in ('a', 'z')})
    return model, state


def save(tmp_path, state, metadata=None):
    path = tmp_path / 'adapter.safetensors'
    save_file(state, str(path), metadata=metadata)
    return path


def assert_unpatched(model):
    assert not model.a._forward_hooks
    assert not model.z._forward_hooks


@pytest.mark.parametrize('kind', ['empty', 'comfy', 'particle', 'wrong_format', 'wrong_rank'])
def test_inference_rejects_wrong_format_instead_of_silent_base_render(tmp_path, kind):
    model, state = model_and_state()
    metadata = {'format': 'anima-distilled-rank8-v1', 'rank': '8'}
    if kind == 'empty':
        state = {}
    elif kind == 'comfy':
        state = {key.replace('.lora_A.weight', '.lora_down.weight').replace('.lora_B.weight', '.lora_up.weight'): value
                 for key, value in state.items()}
    elif kind == 'particle':
        state = {'particles': torch.zeros(128, 4)}
        metadata = {'anima': '{"format":"anima-turbo-routed-particles-v1"}'}
    else:
        metadata['format' if kind == 'wrong_format' else 'rank'] = 'wrong'
    with pytest.raises(ValueError):
        attach_lora(model, save(tmp_path, state, metadata), 1.)
    assert_unpatched(model)


@pytest.mark.parametrize('mutation', ['missing_up', 'orphan_up', 'orphan_alpha', 'unknown_target', 'shape',
                                     'rank', 'nan_down', 'inf_up', 'integer', 'vector_alpha', 'integer_alpha',
                                     'nan_alpha', 'zero_alpha'])
def test_invalid_later_branch_leaves_no_earlier_hook_active(tmp_path, mutation):
    model, state = model_and_state()
    if mutation == 'missing_up':
        del state['z.lora_B.weight']
    elif mutation == 'orphan_up':
        state['extra.lora_B.weight'] = torch.ones(6, 8)
    elif mutation == 'orphan_alpha':
        state['extra.alpha'] = torch.tensor(8.)
    elif mutation == 'unknown_target':
        state = {key.replace('z.', 'unknown.'): value for key, value in state.items()}
    elif mutation == 'shape':
        state['z.lora_B.weight'] = torch.ones(7, 8)
    elif mutation == 'rank':
        state['z.lora_B.weight'] = torch.ones(6, 7)
    elif mutation == 'nan_down':
        state['z.lora_A.weight'][0, 0] = math.nan
    elif mutation == 'inf_up':
        state['z.lora_B.weight'][0, 0] = math.inf
    elif mutation == 'integer':
        state['z.lora_A.weight'] = state['z.lora_A.weight'].long()
    elif mutation == 'vector_alpha':
        state['z.alpha'] = torch.tensor([8.])
    elif mutation == 'integer_alpha':
        state['z.alpha'] = torch.tensor(8)
    else:
        state['z.alpha'] = torch.tensor(math.nan if mutation == 'nan_alpha' else 0.)
    with pytest.raises(ValueError):
        attach_lora(model, save(tmp_path, state), 1.)
    assert_unpatched(model)


def test_hook_registration_failure_rolls_back_installed_hooks(tmp_path, monkeypatch):
    model, state = model_and_state()
    def fail(*args, **kwargs):
        raise RuntimeError('registration failed')
    monkeypatch.setattr(model.z, 'register_forward_hook', fail)
    with pytest.raises(RuntimeError, match='registration failed'):
        attach_lora(model, save(tmp_path, state), 1.)
    assert_unpatched(model)


@pytest.mark.parametrize('strength', [0., 1., -1., 3.])
@pytest.mark.parametrize('tagged', [False, True])
def test_native_tagged_and_legacy_schema_preserve_alpha_math(tmp_path, strength, tagged):
    model, state = model_and_state()
    metadata = {'format': 'anima-distilled-rank8-v1', 'rank': '8'} if tagged else None
    x = torch.randn(2, 3, 4)
    base = model.a(x)
    handles = attach_lora(model, save(tmp_path, state, metadata), strength)
    try:
        expected = base + (strength * 12. / 8.) * ((x @ state['a.lora_A.weight'].T) @ state['a.lora_B.weight'].T)
        assert torch.equal(model.a(x), expected)
    finally:
        for handle in handles:
            handle.remove()
    assert_unpatched(model)


@pytest.mark.parametrize('strength', [math.inf, -math.inf, math.nan])
def test_nonfinite_strength_is_rejected_before_installation(tmp_path, strength):
    model, state = model_and_state()
    with pytest.raises(ValueError, match='strength'):
        attach_lora(model, save(tmp_path, state), strength)
    assert_unpatched(model)


@pytest.mark.parametrize('matrix', ['lora_A.weight', 'lora_B.weight'])
def test_finite_fp64_matrix_that_overflows_fp32_cannot_install_any_hooks(tmp_path, matrix):
    model, state = model_and_state()
    state['z.' + matrix] = torch.full(state['z.' + matrix].shape, 1e300, dtype=torch.float64)
    with pytest.raises(ValueError, match='finite in FP32'):
        attach_lora(model, save(tmp_path, state), 1.)
    assert_unpatched(model)


def test_supported_fp64_matrices_use_finite_fp32_execution(tmp_path):
    model, state = model_and_state()
    for key in state:
        if key.endswith('.weight'):
            state[key] = state[key].double()
    x = torch.randn(2, 3, 4)
    base = model.a(x)
    handles = attach_lora(model, save(tmp_path, state), 1.)
    try:
        expected = base + (12. / 8.) * ((x @ state['a.lora_A.weight'].float().T) @ state['a.lora_B.weight'].float().T)
        assert torch.equal(model.a(x), expected)
    finally:
        for handle in handles:
            handle.remove()
    assert_unpatched(model)
