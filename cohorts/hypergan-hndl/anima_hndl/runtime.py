"""Candidate-only local loading; inherit the unchanged native runtime methods."""
from collections import OrderedDict
import json
from pathlib import Path

import torch

from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.contracts import file_hash
from lumen_studio.particles import Mixer
from lumen_studio.provenance import model_identity, verify_environment


class LocalTurboRuntime(TurboRuntime):
    """Keep the historical initializer/identity intact; bind both local loaders."""
    def __init__(self, model_dir, device='cpu', *, verify=True, checkpointing=False):
        from diffusers import ModularPipeline
        self.model_dir = Path(model_dir).resolve()
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.lock = json.loads((self.model_dir / 'anima-lock.json').read_text())
        if self.lock['model'] != 'circlestone-labs/Anima:turbo-v1.1':
            raise ValueError('Expected the pinned official Turbo v1.1 conversion')
        if not verify:
            raise ValueError('Candidate native loading requires verification')
        verify_environment()
        for name, expected in self.lock['files'].items():
            if file_hash(self.model_dir / name) != expected:
                raise ValueError('Model file hash mismatch: ' + name)
        self.identity = model_identity(self.model_dir)
        self.pipe = ModularPipeline.from_pretrained(str(self.model_dir), local_files_only=True)
        self.pipe.load_components(dtype=self.dtype, local_files_only=True)
        for name in ('transformer', 'text_encoder', 'text_conditioner', 'vae'):
            getattr(self.pipe, name).requires_grad_(False).eval().to(device=self.device)
        self.transformer = self.pipe.transformer
        if checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.mixer = Mixer(self.transformer)
        self.embeddings = OrderedDict()
        self.embedding_limit = 256
        self.scale = self.pipe.vae_scale_factor
        self.channels = self.transformer.config.in_channels
