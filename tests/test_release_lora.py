"""The released LoRA hook must preserve the explicit two-matrix operation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import pytest
from torch import nn
from safetensors.torch import save_file, load_file
from release_tools.distill import attach_lora, save_loras, particle_sample_path, comfy_name
from lumen_studio.alpha_adapter import export_with_alpha, AlphaParticleAdapter, read_metadata
from lumen_studio.particles import ParticleAdapter
from particle_io import ParticleNetwork


def test_lora_delta_and_zero(tmp_path):
    torch.manual_seed(71)
    model=nn.Module();model.projection=nn.Linear(12,16,bias=False)
    down=torch.randn(8,12);up=torch.randn(16,8)
    file=tmp_path/'lora.safetensors'
    save_file({'projection.lora_A.weight':down,'projection.lora_B.weight':up},str(file))
    x=torch.randn(2,3,12);base=model.projection(x)
    for strength in (0,1,3,5):
        handles=attach_lora(model,file,strength)
        actual=model.projection(x)
        expected=base+strength*((x@down.T)@up.T)
        assert torch.equal(actual,expected)
        for h in handles:h.remove()
        assert torch.equal(model.projection(x),base)


def test_fractional_sample_paths_do_not_collide_with_off(tmp_path):
    paths = [particle_sample_path(tmp_path, 'bad-intent', 0, s, '.png') for s in (0, .5, 1)]
    assert len(set(paths)) == 3
    assert paths[1].name == 'str0.5.png'
    assert paths[1].with_suffix('.json').name == 'str0.5.json'


def test_alpha_export_comfy_particle_and_lora_agree(tmp_path):
    torch.manual_seed(91)
    model=nn.Module();model.to_q=nn.Linear(12,16,bias=False)
    original=ParticleAdapter(model)
    for b in original.branches:
        nn.init.normal_(b.up.weight)
    teacher=tmp_path/'teacher.safetensors'
    original.export(teacher,model_identity={'model':'test'},normalization={'test':True},
                    manifest_hash='a'*64,step=1600,ema=original.state_dict())
    alpha_file=tmp_path/'alpha.safetensors'
    export_with_alpha(teacher,alpha_file,alpha=23.08072421821097)
    adapted=AlphaParticleAdapter(model)
    adapted.load_export(alpha_file,model_identity={'model':'test'})
    network=ParticleNetwork(load_file(str(alpha_file)),read_metadata(alpha_file))
    x=torch.randn(2,3,12)
    assert torch.equal(adapted.branches[0](x,adapted.particles),network.branches[0](x,network.particles))
    down=torch.randn(8,12)
    save_loras(adapted,[down],tmp_path,'bad-intent',alpha_file)
    native=load_file(str(tmp_path/'native/bad-intent.safetensors'))
    comfy=load_file(str(tmp_path/'comfyui/bad-intent.safetensors'))
    prefix='diffusion_model.'+comfy_name('to_q')
    assert torch.equal(native['to_q.alpha'],comfy[prefix+'.alpha'])
    assert torch.equal(native['to_q.lora_A.weight'],comfy[prefix+'.lora_down.weight'])
    assert torch.equal(native['to_q.lora_B.weight'],comfy[prefix+'.lora_up.weight'])
    base=model.to_q(x)
    for s in (0,1):
        handles=attach_lora(model,tmp_path/'native/bad-intent.safetensors',s)
        expected=base+(s*23.08072421821097/8)*((x@down.T)@original.branches[0].up.weight.T)
        assert torch.equal(model.to_q(x),expected)
        for h in handles:h.remove()


@pytest.mark.parametrize('alpha',[float('nan'),float('inf'),0.,-1.])
def test_invalid_lora_alpha_is_rejected(tmp_path,alpha):
    model=nn.Module();model.projection=nn.Linear(12,16,bias=False)
    path=tmp_path/'invalid.safetensors'
    save_file({'projection.lora_A.weight':torch.randn(8,12),
        'projection.lora_B.weight':torch.randn(16,8),'projection.alpha':torch.tensor(alpha)},str(path))
    with pytest.raises(ValueError,match='alpha'):
        attach_lora(model,path,1)
    assert not model.projection._forward_hooks
