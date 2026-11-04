"""Anima's bounded image factory; native law stays in the product and core.

This new source cohort is CPU response-only. Native admission, sampler parity
and HNDL definition migration require separate review; legacy routes stay put.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from particle_sliders.lifecycle import (
    Backend, Binding, Capabilities, UnsupportedOperation, native_cleanup,
)

from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity


CAPABILITIES = Capabilities(
    'routed_particle_gan', render=False, strength_min=0., strength_max=1.,
    base_precisions=('bfloat16',), render_outputs=(),
    qualification='Core 0.8.1 source proposal; prior 0.7 CPU receipts are historical',
    unavailable_reasons=(('render', 'Anima native sampler/decoder parity remains unqualified'),),
)
_OPTIONS = {'cache_dir', 'cache_index_sha256', 'model_dir', 'variation', 'device',
            'base_precision', 'steps', 'batch', 'seed', 'ema_decay', 'response_scale',
            'sigma', 'critic_width'}


def validate_runtime(options):
    if options.get('device', 'cpu') != 'cpu':
        raise UnsupportedOperation('Anima lifecycle requires CPU; native accelerator admission remains unqualified')
    if options.get('base_precision', 'bfloat16') != 'bfloat16':
        raise UnsupportedOperation('Anima preserves its native BF16 base and FP32 trainables')


def preflight_runtime(runtime):
    if runtime is not None and (str(runtime.device) != 'cpu' or str(runtime.dtype) != 'torch.bfloat16'):
        raise UnsupportedOperation('staged Anima runtime must be CPU with the native BF16 base')


class NativeOwner:
    """One runtime owner reused by training and data-independent inference."""
    def __init__(self, options, runtime=None):
        validate_runtime(options)
        preflight_runtime(runtime)
        self.runtime = runtime
        self.owns_runtime = runtime is None
        self.adapter = None
        self.restore = None
        self.original_handles = set()
        self.attachment_started = False
        self.closed = False
        self.variation = options['variation']
        try:
            if runtime is None:
                if not options.get('model_dir'):
                    raise ValueError('model_dir is required unless a staged CPU runtime is supplied')
                from .backends.anima import TurboRuntime
                self.runtime = TurboRuntime.__new__(TurboRuntime)
                TurboRuntime.__init__(self.runtime, options['model_dir'], device='cpu',
                                      verify=True, checkpointing=False)
            preflight_runtime(self.runtime)
            if self.variation in self.runtime.mixer.adapters:
                raise ValueError('Anima session requires its own unused adapter name')
            self.original_handles = set(self.runtime.mixer.handles)
            self.restore = native_cleanup(self.runtime.transformer)
        except BaseException:
            self.close()
            raise

    def attach(self):
        # Training controls the seed context across bank AND critic construction.
        import torch
        from particle_sliders.pg09_backends import AnimaRuntimeResponseAdapter
        from .particles import ParticleAdapter
        self.adapter = ParticleAdapter(self.runtime.transformer).to(device='cpu', dtype=torch.float32)
        self.attachment_started = True
        self.runtime.mixer.add(self.variation, self.adapter)
        return AnimaRuntimeResponseAdapter(self.runtime, self.adapter, self.variation)

    @contextmanager
    def observation(self):
        """Restore native state outside AdapterParameters on success or failure."""
        import torch
        runtime = self.runtime
        roots = [runtime.transformer, self.adapter]
        for key in ('text_encoder', 'text_conditioner', 'vae'):
            module = getattr(runtime.pipe, key, None)
            if isinstance(module, torch.nn.Module):
                roots.append(module)
        restores = [native_cleanup(module) for module in roots]
        # AdapterParameters registers leaves, not the original native modules.
        # Own their buffers here, including nonpersistent registrations.
        modules = {id(module): module for root in roots for module in root.modules()}.values()
        buffers = [(module, dict(module._buffers), set(module._non_persistent_buffers_set))
                   for module in modules]
        values = {id(buffer): (buffer, buffer.detach().clone())
                  for _, registered, _ in buffers for buffer in registered.values() if buffer is not None}
        scheduler = runtime.pipe.scheduler
        scheduler_state = deepcopy(vars(scheduler))
        strengths = runtime.mixer.strengths
        mix = strengths.copy()
        embeddings = getattr(runtime, 'embeddings', None)
        entries = list(embeddings.items()) if embeddings is not None else None
        try:
            self.adapter.eval()
            yield
        finally:
            try:
                with torch.no_grad():
                    for buffer, saved in values.values():
                        if (buffer.shape, buffer.dtype, buffer.device) == (saved.shape, saved.dtype, saved.device):
                            buffer.copy_(saved)
                        else:
                            buffer.data = saved.clone()
                    for module, registered, nonpersistent in buffers:
                        module._buffers.clear()
                        module._buffers.update(registered)
                        module._non_persistent_buffers_set.clear()
                        module._non_persistent_buffers_set.update(nonpersistent)
                for restore in reversed(restores):
                    restore()
            finally:
                runtime.pipe.scheduler = scheduler
                vars(scheduler).clear()
                vars(scheduler).update(scheduler_state)
                runtime.mixer.strengths = strengths
                strengths.clear()
                strengths.update(mix)
                if embeddings is not None:
                    runtime.embeddings = embeddings
                    embeddings.clear()
                    embeddings.update(entries)

    def close(self):
        if self.closed:
            return
        self.closed = True
        runtime = self.runtime
        try:
            mixer = getattr(runtime, 'mixer', None)
            if mixer is not None:
                if self.adapter is not None and mixer.adapters.get(self.variation) is self.adapter:
                    mixer.adapters.pop(self.variation)
                    mixer.strengths.pop(self.variation, None)
                if self.attachment_started and not mixer.adapters:
                    for handle in tuple(mixer.handles):
                        if handle not in self.original_handles:
                            handle.remove()
                            mixer.handles.remove(handle)
        finally:
            try:
                if self.restore is not None:
                    self.restore()
            finally:
                if self.owns_runtime and runtime is not None:
                    mixer = getattr(runtime, 'mixer', None)
                    if mixer is not None:
                        mixer.close()
                    embeddings = getattr(runtime, 'embeddings', None)
                    if embeddings is not None:
                        embeddings.clear()
                    for name in ('transformer', 'pipe'):
                        vars(runtime).pop(name, None)


def validate(options):
    """Allocation-free checks, including gates before native imports/loading."""
    validate_runtime(options)
    if set(options) - _OPTIONS:
        raise ValueError('unknown Anima session option: ' + ', '.join(sorted(set(options) - _OPTIONS)))
    variation = options.get('variation')
    if not isinstance(variation, str) or not variation.strip():
        raise ValueError('Anima requires an explicit adapter/cache variation')
    for key in ('cache_dir', 'model_dir'):
        value = options.get(key)
        if key == 'model_dir' and value is None:
            continue  # Backend.open(..., runtime=...) may borrow a staged runtime.
        if not isinstance(value, str) or not value or not Path(value).is_dir():
            raise ValueError(key + ' must be an existing local directory')
    expected = options.get('cache_index_sha256')
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)):
        raise ValueError('cache_index_sha256 must bind the existing local index bytes')
    index = Path(options['cache_dir']) / 'index.json'
    if not index.is_file():
        raise ValueError('Anima requires an existing local cache index')
    data = index.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Anima cache index differs from the configured SHA256')
    manifest = json.loads(data)
    if (not isinstance(manifest, dict) or not isinstance(manifest.get('identity'), dict)
            or manifest['identity'].get('split') != 'train'
            or manifest['identity'].get('variation') != variation):
        raise ValueError('Anima requires the matching frozen training cache')
    if options.get('model_dir') and not (Path(options['model_dir']) / 'anima-lock.json').is_file():
        raise ValueError('model_dir must contain the original staged anima-lock.json')
    for key in ('response_scale', 'sigma'):
        value = options.get(key, 1.)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + ' must be finite and positive')
    width = options.get('critic_width', 32)
    if type(width) is not int or not 1 <= width <= 512:
        raise ValueError('critic_width must be an integer in [1,512]')


def preflight(options, request, staged):
    if set(staged) - {'runtime'}:
        raise ValueError('Anima only accepts a staged runtime, not caller-built training components')
    preflight_runtime(staged.get('runtime'))
    if request['native']:
        raise UnsupportedOperation('Anima training-session native rendering remains unqualified')


def prepare(options, *, runtime=None):
    """Own construction; common core owns stepping, packets and observation."""
    validate(options)
    import torch
    from particle_sliders.pg09 import ConditionalPG09
    from .cache import TargetCache
    from .pg09_training import CachedResponses, NativeFieldCritic

    owner = NativeOwner(options, runtime)
    try:
        runtime = owner.runtime
        validate(options)  # Loading must not silently change the configured index.
        responses = CachedResponses(TargetCache(options['cache_dir'], verify=False),
                                    runtime, options['variation'])
        frozen_identity = responses.identity()
        # Keep the original bank-then-critic seeded initialization sequence.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(options.get('seed', 7))
            backend = owner.attach()
            critic = NativeFieldCritic(responses.response_shape[1],
                responses.embedding_width + 4, options.get('critic_width', 32)).to(dtype=torch.float32)
        game = ConditionalPG09(backend, responses, critic,
            particles=owner.adapter.particles, preset='bispf_optimized',
            response_domain='continuous_field', sampling='balanced',
            batch_size=options.get('batch', 2), seed=options.get('seed', 7),
            total_steps=options.get('steps', 4),
            response_scale=options.get('response_scale', 1.), sigma=options.get('sigma', 1.))
        specification = NATIVE_BACKEND.specification(backend, model_identity=runtime.identity,
            protocol=PROTOCOL, configuration=dict(variation=options['variation'], base_precision='bfloat16',
                latent_channels=int(runtime.transformer.config.in_channels),
                embedding_width=int(runtime.transformer.config.text_embed_dim)))
        def before_update(request):
            # Pure decision over existing evidence. No external I/O, cached
            # approval or inference that CPU fixtures admit accelerator work.
            if str(runtime.device) != 'cpu':
                raise UnsupportedOperation('Anima native accelerator admission remains unqualified')
            if responses.identity() != frozen_identity:
                raise ValueError('Anima frozen runtime/cache evidence changed before update')
            return True
        return Binding(game, owner.close, inference=specification,
                       before_update=before_update, observation=owner.observation)
    except BaseException:
        owner.close()
        raise


PROTOCOL = 'anima-turbo-native-velocity-v1'
NATIVE_CAPABILITIES = Capabilities('routed_particle_gan', render=False,
    base_precisions=('bfloat16',), render_outputs=(), operations=('response',),
    unavailable_reasons=(('render', 'Anima native sampler/decoder parity remains unqualified'),),
    qualification='Core 0.8.1 source proposal; Anima native-only conformance pending')


def validate_inference(options, specification):
    validate_runtime(options)
    if set(options) - {'model_dir', 'device', 'base_precision'}:
        raise ValueError('native inference accepts only local model_dir/device/base_precision; no training cache')
    directory = options.get('model_dir')
    if not isinstance(directory, str) or not (Path(directory) / 'anima-lock.json').is_file():
        raise ValueError('native inference requires the original local staged anima-lock.json')
    configuration = specification.configuration
    if (set(configuration) != {'variation', 'base_precision', 'latent_channels', 'embedding_width'}
            or not isinstance(configuration['variation'], str) or not configuration['variation'].strip()
            or configuration['base_precision'] != 'bfloat16'
            or any(type(configuration[key]) is not int or configuration[key] < 1
                   for key in ('latent_channels', 'embedding_width'))):
        raise ValueError('invalid native Anima architecture configuration')
    if specification.identity['protocol'] != PROTOCOL:
        raise ValueError('unsupported native Anima inference protocol')


def preflight_inference(options, request, specification):
    validate_inference(options, specification)
    native = request['native']
    if set(native) - {'latent', 'embedding', 'timestep'}:
        raise ValueError('native Anima response requires only latent, embedding and native timestep')
    if not native:
        return  # Opening may precede supplying a response request.
    if set(native) != {'latent', 'embedding', 'timestep'}:
        raise ValueError('supply latent, embedding and timestep together')
    # Inspect tensor metadata before a model is allocated; values are checked by
    # the callback without assuming tensor contents are finite from shape alone.
    for key, rank in (('latent', 5), ('embedding', 3)):
        tensor = native[key]
        if (getattr(tensor, 'ndim', None) != rank or tensor.shape[0] != 1
                or any(size < 1 for size in tensor.shape) or str(tensor.device) != 'cpu'
                or not tensor.is_floating_point()):
            raise ValueError('native Anima requires a CPU floating ' + key + ' tensor with singleton batch')
    if native['latent'].shape[2] != 1:
        raise ValueError('native Anima retains its singleton time plane')
    if (native['latent'].shape[1] != specification.configuration['latent_channels']
            or native['embedding'].shape[-1] != specification.configuration['embedding_width']):
        raise ValueError('native Anima channel/conditioning geometry differs from the export')
    timestep = native['timestep']
    if isinstance(timestep, bool) or not isinstance(timestep, (int, float)) or not math.isfinite(timestep) or timestep < 0:
        raise ValueError('native Anima timestep must be a finite nonnegative scheduler scalar')


def prepare_inference(options, specification):
    validate_inference(options, specification)
    import torch
    owner = NativeOwner(dict(options, variation=specification.configuration['variation']))
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            backend = owner.attach()
        runtime = owner.runtime
        identity = native_identity(backend, model_identity=runtime.identity, protocol=PROTOCOL)
        def response(*, strength, seed, latent, embedding, timestep):
            if (latent.shape[1] != runtime.transformer.config.in_channels
                    or embedding.shape[-1] != runtime.transformer.config.text_embed_dim):
                raise ValueError('native Anima latent channels or embedding width differ from the actual model')
            if (timestep > float(runtime.pipe.scheduler.config.num_train_timesteps)
                    or not torch.isfinite(latent).all() or not torch.isfinite(embedding).all()):
                raise ValueError('native Anima response values or scheduler timestep are invalid')
            with backend.network.scaled(strength):
                return runtime.predict(latent, timestep, embedding).detach().clone()
        return InferenceBinding(backend.network, identity, owner.close, response=response,
                                observation=owner.observation)
    except BaseException:
        owner.close()
        raise


NATIVE_BACKEND = InferenceBackend('anima', NATIVE_CAPABILITIES, prepare_inference,
                                  validate_inference, preflight=preflight_inference)


BACKEND = Backend('anima', CAPABILITIES, prepare, validate, preflight=preflight)
