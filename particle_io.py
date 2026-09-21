"""Strict native checkpoint loading and Diffusers-to-ComfyUI target mapping."""
import json
from pathlib import Path
import re
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file

try:
    from .lumen_studio.particles import Branch, ARCHITECTURE, FORMAT
except ImportError:
    from lumen_studio.particles import Branch, ARCHITECTURE, FORMAT


def comfy_name(name):
    match = re.fullmatch(r'transformer_blocks\.(\d+)\.(attn[12])\.(to_q|to_k|to_v|to_out\.0)', name)
    if not match:
        raise ValueError(f'Unsupported projection: {name}')
    block, attention, projection = match.groups()
    attention = {'attn1':'self_attn','attn2':'cross_attn'}[attention]
    projection = {'to_q':'q_proj','to_k':'k_proj','to_v':'v_proj','to_out.0':'output_proj'}[projection]
    return f'blocks.{block}.{attention}.{projection}'


class ParticleNetwork(nn.Module):
    def __init__(self, state, metadata):
        super().__init__()
        self.names = metadata['targets']
        self.particles = nn.Parameter(torch.empty(128,4), requires_grad=False)
        self.branches = nn.ModuleList()
        for i,name in enumerate(self.names):
            down=state[f'branches.{i}.down.weight']; up=state[f'branches.{i}.up.weight']
            if down.ndim!=2 or up.ndim!=2 or down.shape[0]!=8 or up.shape[1]!=8:
                raise ValueError('Invalid rank-8 branch')
            self.branches.append(Branch(down.shape[1],up.shape[0]))
        self.load_state_dict(state,strict=True)
        self.requires_grad_(False).eval()

    @classmethod
    def load(cls,path):
        with safe_open(str(path),framework='pt',device='cpu') as f:
            metadata=json.loads(f.metadata()['anima'])
        if metadata['format']!=FORMAT or metadata['architecture']!=ARCHITECTURE:
            raise ValueError('Expected a native Anima Turbo particle adapter')
        if metadata['model_identity']['model']!='circlestone-labs/Anima:turbo-v1.1':
            raise ValueError('This adapter is not for Anima Turbo v1.1')
        names=metadata['targets']
        expected=[f'transformer_blocks.{i}.{a}.{p}' for i in range(28)
                  for a in ('attn1','attn2') for p in ('to_q','to_k','to_v','to_out.0')]
        if names!=expected:
            raise ValueError('Expected all 224 Anima attention projections in native order')
        state=load_file(str(path))
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Non-finite particle checkpoint')
        with torch.random.fork_rng(devices=[]):
            return cls(state,metadata), metadata
