"""Actual pinned component loading must not call Accelerate's affected APIs.

These are local random model files, including genuine sharded safetensors.
They exercise ComponentSpec.load, the component path used by ModularPipeline.
They do not qualify the pretrained Anima model or arbitrary untrusted artifacts.
"""
import pytest
import torch


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
