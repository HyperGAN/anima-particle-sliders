"""Actual pinned component loading must not call Accelerate's affected APIs.

These are local random model files, including genuine sharded safetensors.
They exercise ComponentSpec.load, the component path used by ModularPipeline.
They do not qualify the pretrained Anima model or arbitrary untrusted artifacts.
"""
import json
from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
import torch


def random_model(family):
    torch.set_num_threads(1)
    torch.manual_seed(29)
    if family == 'cosmos':
        from diffusers import CosmosTransformer3DModel
        model = CosmosTransformer3DModel(
            in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=32,
            num_layers=1, mlp_ratio=2, text_embed_dim=8, adaln_lora_dim=4,
            max_size=(2, 4, 4), patch_size=(1, 1, 1), extra_pos_embed_type=None)
    else:
        from transformers import Qwen3Config, Qwen3Model
        model = Qwen3Model(Qwen3Config(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=32))
    model.eval().requires_grad_(False)
    return model


@pytest.mark.parametrize('family', ['cosmos', 'qwen3'])
def test_native_component_loading_avoids_affected_accelerate_apis(tmp_path, monkeypatch, family):
    import accelerate
    import accelerate.big_modeling
    import accelerate.utils
    import accelerate.utils.modeling
    from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec

    def forbidden(*args, **kwargs):
        pytest.fail('component loading reached an Accelerate API affected by GHSA-4j2p-28q2-5m79')
    for module in (accelerate, accelerate.big_modeling, accelerate.utils, accelerate.utils.modeling):
        for name in ('load_checkpoint_in_model', 'load_checkpoint_and_dispatch'):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    model = random_model(family)
    model.save_pretrained(tmp_path, safe_serialization=True, max_shard_size='2KB')
    assert list(tmp_path.glob('*.index.json')), 'fixture must exercise a real shard index'
    spec = ComponentSpec(name=family, type_hint=type(model),
                         pretrained_model_name_or_path=str(tmp_path))
    loaded = spec.load(local_files_only=True, dtype=torch.float32)
    assert type(loaded) is type(model)
    assert model.state_dict().keys() == loaded.state_dict().keys()
    assert all(torch.equal(value, loaded.state_dict()[name])
               for name, value in model.state_dict().items())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family', ['cosmos', 'qwen3'])
def test_extra_index_component_behavior_with_own_outside_fixture(tmp_path, family):
    """Actual pinned loader behavior; no claim that all libraries ignore indexes."""
    from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec
    from safetensors.torch import save_file
    model = random_model(family)
    root = tmp_path / 'model'
    model.save_pretrained(root, safe_serialization=True)
    outside = tmp_path / 'own-outside.safetensors'
    save_file({k: v.detach().clone() + 1 for k, v in model.state_dict().items()}, str(outside))
    index_name = ('diffusion_pytorch_model.safetensors.index.json' if family == 'cosmos'
                  else 'model.safetensors.index.json')
    (root / index_name).write_text(json.dumps({'metadata': {},
        'weight_map': {key: '../own-outside.safetensors' for key in model.state_dict()}}))
    spec = ComponentSpec(name=family, type_hint=type(model),
                         pretrained_model_name_or_path=str(root))
    if family == 'cosmos':
        with pytest.raises(ValueError, match='path separator or a parent-directory reference'):
            spec.load(local_files_only=True, dtype=torch.float32)
    else:
        loaded = spec.load(local_files_only=True, dtype=torch.float32)
        assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    assert not torch.cuda.is_initialized()


def test_candidate_actual_pipeline_and_component_loaders_use_exact_local_root(tmp_path, monkeypatch):
    """Actual pipeline -> spec -> Cosmos loader, with only generated tiny weights.

    The fixture substitutes its own manifest authority and historical environment
    check. Production has neither override. This is a loader argument/path test,
    not qualification of the full historical environment or pretrained Anima.
    """
    import socket
    from diffusers import ModularPipeline, CosmosTransformer3DModel
    from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
    from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec
    from anima_hndl import admission, runtime as candidate_runtime
    from anima_hndl.lifecycle import NativeOwner
    from lumen_studio.backends.anima import TurboRuntime
    from lumen_studio.contracts import digest
    root = tmp_path / 'admitted-model'
    root.mkdir()
    model = random_model('cosmos')
    model.save_pretrained(root / 'transformer', safe_serialization=True)

    class Blocks(ModularPipelineBlocks):
        @property
        def expected_components(self):
            return [ComponentSpec(name='transformer', type_hint=CosmosTransformer3DModel,
                pretrained_model_name_or_path=str(root), subfolder='transformer')]

    definition = ModularPipeline(blocks=Blocks())
    index = dict(definition.config)
    index['_class_name'] = 'ModularPipeline'
    (root / 'modular_model_index.json').write_text(json.dumps(index))
    trusted = {'model': 'circlestone-labs/Anima:turbo-v1.1', 'revision': 'a' * 40,
        'files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in root.rglob('*') if p.is_file()}}
    (root / 'anima-lock.json').write_text(json.dumps(trusted))
    portable = deepcopy(trusted)
    portable_index = deepcopy(index)
    portable_index['transformer'][2]['pretrained_model_name_or_path'] = '$MODEL_ROOT'
    portable['files']['modular_model_index.json'] = digest(portable_index)
    released = {'model': trusted['model'], 'sha256': digest(portable)}
    monkeypatch.setattr(admission, '_trusted_model', lambda: (deepcopy(trusted), deepcopy(released)))
    calls = []
    monkeypatch.setattr(candidate_runtime, 'verify_environment', lambda: calls.append(('environment',)))
    actual_pipeline_load = ModularPipeline.from_pretrained
    actual_components_load = ModularPipeline.load_components
    actual_model_load = CosmosTransformer3DModel.from_pretrained

    def pipeline_load(cls, path, **kwargs):
        calls.append(('pipeline', path, dict(kwargs)))
        pipe = actual_pipeline_load(path, **kwargs)
        # Only the transformer needs real loading in this bounded path fixture.
        for name in ('text_encoder', 'text_conditioner', 'vae'):
            setattr(pipe, name, torch.nn.Linear(1, 1))
        pipe.vae_scale_factor = 1
        return pipe

    def components_load(self, **kwargs):
        calls.append(('components', dict(kwargs)))
        return actual_components_load(self, **kwargs)

    def model_load(cls, path, **kwargs):
        calls.append(('model', path, dict(kwargs)))
        return actual_model_load(path, **kwargs)

    def no_network(*args, **kwargs):
        pytest.fail('local generated fixture attempted a network connection')

    monkeypatch.setattr(ModularPipeline, 'from_pretrained', classmethod(pipeline_load))
    # The fixture block class is test-local, not a dynamically fetched plugin.
    monkeypatch.setattr(ModularPipelineBlocks, 'from_pretrained', classmethod(lambda cls, *a, **k: Blocks()))
    monkeypatch.setattr(ModularPipeline, 'load_components', components_load)
    monkeypatch.setattr(CosmosTransformer3DModel, 'from_pretrained', classmethod(model_load))
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.chdir(tmp_path)  # Actual component resolution must ignore CWD.
    owner = NativeOwner({'model_dir': str(root), 'variation': 'finalform'})
    try:
        assert calls[0] == ('environment',)
        assert ('pipeline', str(root), {'local_files_only': True}) in calls
        assert ('components', {'dtype': torch.bfloat16, 'local_files_only': True}) in calls
        model_calls = [call for call in calls if call[0] == 'model']
        assert len(model_calls) == 1
        _, path, kwargs = model_calls[0]
        assert path == str(root) and Path(path).is_absolute()
        assert kwargs['subfolder'] == 'transformer' and kwargs['local_files_only'] is True
        assert isinstance(owner.runtime, candidate_runtime.LocalTurboRuntime)
        assert candidate_runtime.LocalTurboRuntime.predict is TurboRuntime.predict
        assert all(torch.equal(v.to(torch.bfloat16), owner.runtime.transformer.state_dict()[k])
                   for k, v in model.state_dict().items())
    finally:
        owner.close()
    assert not hasattr(owner.runtime, 'pipe')
    assert not torch.cuda.is_initialized()
