"""Opt-in Anima native-field cohort using the shared HyperGAN/HNDL builder.

Only product architecture and factory wiring live here. The shared package
owns routed graph source, ParticleGAN objectives, penalties and all updates.
The core0.8.1 default environment deliberately cannot open this backend.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
from importlib.metadata import version
from importlib.resources import files
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Backend, Binding, UnsupportedOperation
from particle_sliders.pg09_backends import AnimaRuntimeResponseAdapter

from . import lifecycle as legacy
from .particles import ARCHITECTURE


PROTOCOL = 'anima-hypergan-hndl-native-velocity-v1'
PROFILE = 'anima-hypergan-hndl-v1'
CORE_COMMIT = '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231'
# Same local-patch architecture as NativeFieldCritic, authored through HNDL.
# Unfold is native input geometry; the shared PG penalty still receives 5D fields.
CRITIC_SOURCE = '''field = linear(patches, $width, name="field0")
field = silu(field)
field = linear(field, $width, name="field2")
field = silu(field)
field = mean(field, 1)
context = linear(condition, $width, name="context0")
context = silu(context)
joined = concat(field, context, axis=1)
score = linear(joined, $width, name="logits0")
score = silu(score)
linear(score, 1, name="logits2")
'''


def verified_builder():
    if version('particle-sliders-core') != '0.12.0':
        raise RuntimeError('Anima HNDL requires the separate core0.12/PG0.9.4 candidate environment')
    # Reuse the common exact HNDL0.10/HyperGAN-develop source closure and clean
    # build provenance guard. No local builder, version override or fallback.
    from particle_sliders.fibo_hndl import verified_builder as shared_builder
    builder, identity = shared_builder()
    identity = dict(identity, product_profile=PROFILE,
        adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        owner_source_sha256=hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest(),
        anima_critic_sha256=hashlib.sha256(CRITIC_SOURCE.encode()).hexdigest())
    return builder, identity


class Branch(nn.Module):
    def __init__(self, module, builder, identity):
        super().__init__()
        self.inputs, self.outputs = module.in_features, module.out_features
        self.graph = builder(files('particle_sliders').joinpath('fibo_projection.hndl').read_text(),
            input_shape={'value': ['B', 1, self.inputs], 'particles': ['B', 128, 4]},
            output_shape=['B', 1, self.outputs], parameters={'output_width': self.outputs})
        nn.init.kaiming_uniform_(self.graph['down'].weight, a=5 ** .5)
        nn.init.zeros_(self.graph['up'].weight)
        self.identity = deepcopy(identity)

    def forward(self, x, particles):
        shape = x.shape
        value = x.to(self.graph['down'].weight).reshape(-1, 1, self.inputs)
        bank = particles.unsqueeze(0).expand(value.shape[0], -1, -1)
        return self.graph(value=value, particles=bank).reshape(*shape[:-1], self.outputs)

    def extra_repr(self):
        return json.dumps(self.identity, sort_keys=True)


class ParticleAdapter(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        builder, identity = verified_builder()
        self.particles = nn.Parameter(torch.randn(128, 4))
        targets = {name: module for name, module in transformer.named_modules()
                   if isinstance(module, nn.Linear)
                   and any(name.endswith(target) for target in ARCHITECTURE['targets'])}
        if not targets:
            raise ValueError('No Anima attention targets found')
        self.names = list(targets)
        self.branches = nn.ModuleList(Branch(module, builder, identity) for module in targets.values())
        self.identity = identity

    def extra_repr(self):
        return json.dumps(self.identity, sort_keys=True)


class NativeFieldCritic(nn.Module):
    score_bound = 0

    def __init__(self, channels, condition_width, spatial_shape, width=32):
        super().__init__()
        builder, identity = verified_builder()
        if any(type(n) is not int or n < 1 for n in (channels, condition_width, width, *spatial_shape)):
            raise ValueError('Anima HNDL critic dimensions must be positive integers')
        if len(spatial_shape) != 3 or spatial_shape[0] != 1:
            raise ValueError('Anima HNDL critic retains its singleton native time plane')
        self.channels, self.condition_width, self.spatial_shape = channels, condition_width, tuple(spatial_shape)
        self.graph = builder(CRITIC_SOURCE,
            input_shape={'patches': ['B', math.prod(spatial_shape), channels * 9],
                         'condition': ['B', condition_width]},
            output_shape=['B', 1], parameters={'width': width})
        self.identity = identity

    def forward(self, response, condition):
        if (tuple(response.shape[1:]) != (self.channels, *self.spatial_shape)
                or tuple(condition.shape) != (response.shape[0], self.condition_width)):
            raise ValueError('Anima HNDL critic native response/condition geometry differs')
        patches = torch.nn.functional.unfold(response.squeeze(2), kernel_size=3, padding=1).transpose(1, 2)
        return self.graph(patches=patches, condition=condition).squeeze(-1)

    def extra_repr(self):
        return json.dumps(self.identity, sort_keys=True)


class NativeOwner(legacy.NativeOwner):
    def attach(self):
        self.adapter = ParticleAdapter(self.runtime.transformer).to(device='cpu', dtype=torch.float32)
        self.attachment_started = True
        self.runtime.mixer.add(self.variation, self.adapter)
        return AnimaRuntimeResponseAdapter(self.runtime, self.adapter, self.variation)


CAPABILITIES = replace(legacy.CAPABILITIES,
    qualification='Anima HyperGAN develop/HNDL0.10 bounded CPU candidate; native pretrained/GPU/render unqualified')
NATIVE_CAPABILITIES = replace(legacy.NATIVE_CAPABILITIES, qualification=CAPABILITIES.qualification)


def validate(options):
    legacy.validate(options)
    verified_builder()


def prepare(options, *, runtime=None):
    validate(options)
    from particle_sliders.pg094 import ConditionalPG094, validate_pg09_domain
    from .cache import TargetCache
    from .pg09_training import CachedResponses
    owner = NativeOwner(options, runtime)
    try:
        responses = CachedResponses(TargetCache(options['cache_dir'], verify=False),
                                    owner.runtime, options['variation'])
        frozen = responses.identity()
        # Preserve actual native shape at admission and every penalty boundary.
        validate_pg09_domain(responses.response_shape, response_domain='continuous_field')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(options.get('seed', 7))
            backend = owner.attach()
            critic = NativeFieldCritic(responses.response_shape[1], responses.embedding_width + 4,
                                      responses.response_shape[2:], options.get('critic_width', 32))
        game = ConditionalPG094(backend, responses, critic, particles=owner.adapter.particles,
            preset='bispf_optimized', response_domain='continuous_field', sampling='balanced',
            batch_size=options.get('batch', 2), seed=options.get('seed', 7),
            total_steps=options.get('steps', 4), response_scale=options.get('response_scale', 1.),
            sigma=options.get('sigma', 1.))
        specification = NATIVE_BACKEND.specification(backend, model_identity=owner.runtime.identity,
            protocol=PROTOCOL, configuration=dict(variation=options['variation'], base_precision='bfloat16',
                latent_channels=int(owner.runtime.transformer.config.in_channels),
                embedding_width=int(owner.runtime.transformer.config.text_embed_dim)))

        def before_update(request):
            legacy.preflight_runtime(owner.runtime)
            if responses.identity() != frozen:
                raise ValueError('Anima HNDL frozen runtime/cache evidence changed before update')
            return True
        return Binding(game, owner.close, inference=specification,
                       before_update=before_update, observation=owner.observation)
    except BaseException:
        owner.close()
        raise


def _legacy_spec(specification):
    # Reuse unchanged tensor geometry/options checks, retaining a distinct protocol.
    return SimpleNamespace(configuration=specification.configuration, identity={'protocol': legacy.PROTOCOL})


def validate_inference(options, specification):
    if specification.identity['protocol'] != PROTOCOL:
        raise ValueError('unsupported native Anima HNDL inference protocol')
    legacy.validate_inference(options, _legacy_spec(specification))
    verified_builder()


def preflight_inference(options, request, specification):
    validate_inference(options, specification)
    legacy.preflight_inference(options, request, _legacy_spec(specification))


def prepare_inference(options, specification):
    validate_inference(options, specification)
    owner = NativeOwner(dict(options, variation=specification.configuration['variation']))
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            backend = owner.attach()
        runtime = owner.runtime

        def response(*, strength, seed, latent, embedding, timestep):
            if (latent.shape[1] != runtime.transformer.config.in_channels
                    or embedding.shape[-1] != runtime.transformer.config.text_embed_dim
                    or timestep > float(runtime.pipe.scheduler.config.num_train_timesteps)
                    or not torch.isfinite(latent).all() or not torch.isfinite(embedding).all()):
                raise ValueError('native Anima HNDL response geometry or values differ')
            with backend.network.scaled(strength):
                return runtime.predict(latent, timestep, embedding).detach().clone()
        return InferenceBinding(backend.network,
            native_identity(backend, model_identity=runtime.identity, protocol=PROTOCOL), owner.close,
            response=response, observation=owner.observation)
    except BaseException:
        owner.close()
        raise


NATIVE_BACKEND = InferenceBackend('anima-hndl', NATIVE_CAPABILITIES, prepare_inference,
                                  validate_inference, preflight=preflight_inference)
BACKEND = Backend('anima-hndl', CAPABILITIES, prepare, validate, preflight=legacy.preflight)
