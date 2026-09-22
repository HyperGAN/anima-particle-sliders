"""Inference-only alpha exports for the fixed single-particle endpoint adapter."""
import math
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
try:
    from .lumen_studio.alpha_adapter import AlphaParticleAdapter,read_metadata
    from .lumen_studio.contracts import canonical,file_hash
    from .lumen_studio.particles import ARCHITECTURE
except ImportError:
    from lumen_studio.alpha_adapter import AlphaParticleAdapter,read_metadata
    from lumen_studio.contracts import canonical,file_hash
    from lumen_studio.particles import ARCHITECTURE

ORIGINAL_FORMAT='anima-turbo-single-particle-alpha-v1'
ALPHA_FORMAT='anima-turbo-single-particle-alpha-v2'
SINGLE_ARCH=dict(ARCHITECTURE,particles=1,particle_trainable=False)


def alpha_values(metadata,state):
    if metadata.get('format') not in (ORIGINAL_FORMAT,ALPHA_FORMAT) or metadata.get('architecture')!=SINGLE_ARCH:
        raise ValueError('Invalid single-particle architecture')
    alpha=metadata.get('network_alpha')
    if (isinstance(alpha,bool) or not isinstance(alpha,(int,float)) or not math.isfinite(alpha)
            or alpha<=0 or metadata.get('scaling')!='alpha/rank'):
        raise ValueError('Invalid single-particle alpha')
    if metadata['format']==ORIGINAL_FORMAT and alpha!=8.:
        raise ValueError('Original single-particle alpha must remain eight')
    if tuple(state['particles'].shape)!=(1,4) or torch.count_nonzero(state['particles']):
        raise ValueError('Expected one fixed zero routed input')
    keys={f'branches.{i}.alpha' for i in range(len(metadata['targets']))}
    if {k for k in state if k.endswith('.alpha')}!=keys:
        raise ValueError('Invalid branch alpha keys')
    if any(state[k].ndim!=0 or not state[k].is_floating_point() or float(state[k])!=alpha for k in keys):
        raise ValueError('Branch alpha differs from metadata')
    return [float(alpha)]*len(keys)


def export_with_alpha(source,destination,*,alpha,provenance=None):
    metadata=read_metadata(source);state=load_file(str(source))
    alpha_values(metadata,state)
    if metadata['format']!=ORIGINAL_FORMAT or metadata.get('weights')!='ema':
        raise ValueError('Use an original immutable EMA export')
    metadata.update(format=ALPHA_FORMAT,network_alpha=alpha,source_checkpoint_sha256=file_hash(source),
                    alpha_provenance=provenance or {})
    for i in range(len(metadata['targets'])):state[f'branches.{i}.alpha']=torch.tensor(alpha,dtype=torch.float64)
    alpha_values(metadata,state)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError('Exports are immutable')
    temporary=destination.with_suffix('.tmp')
    save_file(state,str(temporary),metadata={'anima':canonical(metadata)})
    temporary.replace(destination)
    return file_hash(destination)


class SingleParticleAdapter(AlphaParticleAdapter):
    def __init__(self,transformer):
        super().__init__(transformer)
        self.particles=torch.nn.Parameter(torch.zeros(1,4),requires_grad=False)

    def load_export(self,path,*,model_identity):
        metadata=read_metadata(path);state=load_file(str(path))
        alphas=alpha_values(metadata,state)
        if metadata.get('model_identity')!=model_identity or metadata.get('targets')!=self.names:
            raise ValueError('Checkpoint runtime or targets differ')
        self.load_state_dict({k:v for k,v in state.items() if not k.endswith('.alpha')},strict=True)
        self.alphas=alphas
        return metadata


def adapter_class(path):
    return SingleParticleAdapter if read_metadata(path)['format'] in (ORIGINAL_FORMAT,ALPHA_FORMAT) else AlphaParticleAdapter
