"""Single-particle publication preserves learned weights and exact alpha math."""
import copy
import torch
import pytest
from torch import nn
from safetensors.torch import save_file,load_file
from lumen_studio.contracts import canonical
from lumen_studio.alpha_adapter import read_metadata
from single_particle import SingleParticleAdapter,SINGLE_ARCH,ORIGINAL_FORMAT,export_with_alpha,alpha_values
from particle_io import ParticleNetwork


def source_file(tmp_path):
    torch.manual_seed(391)
    model=nn.Module();model.to_q=nn.Linear(12,16,bias=False)
    adapter=SingleParticleAdapter(model)
    for branch in adapter.branches:nn.init.normal_(branch.up.weight)
    state={k:v.detach().contiguous() for k,v in adapter.state_dict().items()}
    state['branches.0.alpha']=torch.tensor(8.,dtype=torch.float64)
    metadata=dict(format=ORIGINAL_FORMAT,architecture=SINGLE_ARCH,scaling='alpha/rank',network_alpha=8.,
                  weights='ema',targets=adapter.names,model_identity={'model':'test'},step=1600)
    source=tmp_path/'source.safetensors'
    save_file(state,str(source),metadata={'anima':canonical(metadata)})
    return model,source


def test_calibrated_single_particle_math(tmp_path):
    model,source=source_file(tmp_path)
    calibrated=tmp_path/'calibrated.safetensors'
    export_with_alpha(source,calibrated,alpha=24.)
    original=SingleParticleAdapter(model);original.load_export(source,model_identity={'model':'test'})
    loaded=SingleParticleAdapter(model);loaded.load_export(calibrated,model_identity={'model':'test'})
    network=ParticleNetwork(load_file(str(calibrated)),read_metadata(calibrated))
    x=torch.randn(2,3,12)
    expected=original.branches[0](x,original.particles)*3.
    assert torch.equal(loaded.branches[0](x,loaded.particles),expected)
    assert torch.equal(network.branches[0](x,network.particles),expected)
    before,after=load_file(str(source)),load_file(str(calibrated))
    assert all(torch.equal(v,after[k]) for k,v in before.items() if not k.endswith('.alpha'))
    with pytest.raises(FileExistsError):export_with_alpha(source,calibrated,alpha=24.)


@pytest.mark.parametrize('mutation',['alpha','particles','architecture','legacy_alpha'])
def test_single_particle_rejects_mislabeled_files(tmp_path,mutation):
    _,source=source_file(tmp_path)
    metadata=copy.deepcopy(read_metadata(source));state=load_file(str(source))
    if mutation=='alpha':state['branches.0.alpha']=torch.tensor(float('nan'))
    elif mutation=='particles':state['particles'][0,0]=1.
    elif mutation=='architecture':metadata['architecture']['particles']=128
    else:
        metadata['network_alpha']=24.;state['branches.0.alpha']=torch.tensor(24.)
    with pytest.raises(ValueError):alpha_values(metadata,state)
