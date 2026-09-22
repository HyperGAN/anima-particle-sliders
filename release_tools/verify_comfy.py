"""Validate native particles and ordinary LoRAs with an actual ComfyUI checkout (CPU)."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
COMFY=Path(sys.argv[1]).resolve(); WEIGHTS=Path(sys.argv[2]).resolve()
sys.path[:0]=[str(ROOT),str(COMFY)]
sys.argv=[sys.argv[0],'--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch
from torch import nn
import comfy.ops
import comfy.lora
from comfy.ldm.anima.model import Anima
from comfy.model_patcher import ModelPatcher
from particle_io import ParticleNetwork,comfy_name
from comfy_particle import AnimaParticleSlider,ParticleWrapper
from safetensors.torch import load_file
import folder_paths

torch.set_num_threads(4); torch.set_grad_enabled(False)
# Real Anima topology; meta weights avoid allocating another 2B base model.
host=Anima(max_img_h=240,max_img_w=240,max_frames=128,in_channels=16,out_channels=16,
    patch_spatial=2,patch_temporal=1,model_channels=2048,num_blocks=28,num_heads=16,
    crossattn_emb_channels=1024,pos_emb_cls='rope3d',use_adaln_lora=True,
    device='meta',dtype=torch.float32,operations=comfy.ops.manual_cast)
base=nn.Module();base.diffusion_model=host
base.model_config=SimpleNamespace(unet_config={})
patcher=ModelPatcher(base,torch.device('cpu'),torch.device('cpu'))
folder_paths.add_model_folder_path('loras',str(WEIGHTS/'weights'))
node=AnimaParticleSlider();reports=[]
for entry in json.loads((WEIGHTS/'catalog.json').read_text())['sliders']:
    variation=entry['id']
    file=WEIGHTS/entry['particle']
    network,metadata=ParticleNetwork.load(file)
    wrapped=node.load(patcher,file.name,1.)[0]
    assert 'model_function_wrapper' not in patcher.model_options
    assert len(wrapped.model_options['model_function_wrapper'].modules)==224
    off=node.load(patcher,file.name,0.)[0]
    assert 'model_function_wrapper' not in off.model_options
    # Execute every real mapped projection with zero base weights and verify
    # exact source branch math. Supports ComfyUI's 5D spatial hidden states.
    wrapper=wrapped.model_options['model_function_wrapper']
    errors=[]
    for i,(module,branch) in enumerate(zip(wrapper.modules,network.branches)):
        original=module.weight
        module.weight=nn.Parameter(torch.zeros(original.shape),requires_grad=False)
        x=torch.randn(1,1,1,2,branch.down.in_features)
        reference=branch(x,network.particles)
        def call(x,t,**kwargs):return module(x)
        actual=wrapper(call,dict(input=x,timestep=torch.tensor([1.]),c={}))
        errors.append(float((actual-reference).abs().max()))
        assert not module._forward_hooks
        module.weight=original
    assert max(errors)==0, max(errors)
    # Hooks are restored on errors too.
    def fail(*args,**kwargs):raise RuntimeError('test')
    try:wrapper(fail,dict(input=torch.zeros(1),timestep=torch.zeros(1),c={}))
    except RuntimeError:pass
    assert all(not m._forward_hooks for m in wrapper.modules)
    state=load_file(str(WEIGHTS/entry['comfyui_lora']))
    mapping=comfy.lora.model_lora_keys_unet(base,{})
    patches=comfy.lora.load_lora(state,mapping)
    assert len(patches)==224,len(patches)
    assert len(patcher.clone().add_patches(patches,1.))==224
    native=load_file(str(WEIGHTS/entry['native_lora']))
    # The released native and ComfyUI tensors represent exactly the same deltas.
    for name in network.names:
        prefix='diffusion_model.'+comfy_name(name)
        assert torch.equal(native[name+'.lora_A.weight'],state[prefix+'.lora_down.weight'])
        assert torch.equal(native[name+'.lora_B.weight'],state[prefix+'.lora_up.weight'])
        alpha=native.get(name+'.alpha',torch.tensor(float(native[name+'.lora_A.weight'].shape[0])))
        assert float(alpha)==float(state[prefix+'.alpha'])
    reports.append(dict(variation=variation,particle_projections=224,lora_patches=224,
        particle_max_abs=max(errors),zero_bypass=True,clone_isolation=True,exception_restoration=True,
        particle=entry['particle'],lora=entry['comfyui_lora'],
        particle_alpha=metadata.get('network_alpha',8.),
        lora_alphas=sorted({float(v) for k,v in state.items() if k.endswith('.alpha')})))
import subprocess
result=dict(passed=True,device='cpu',comfy_revision=subprocess.check_output(
    ['git','-C',str(COMFY),'rev-parse','HEAD'],text=True).strip(),sliders=reports,
    limitation='Full ComfyUI GPU image generation is not covered by this CPU integration check.')
(ROOT/'validation/comfyui.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

# Release ComfyUI patchers while Python import machinery is still available.
del wrapped,off,patcher
import gc
gc.collect()
