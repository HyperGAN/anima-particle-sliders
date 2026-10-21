"""Anima's bounded image factory; native law stays in the product and core.

This new source cohort is CPU response-only. Native admission, sampler parity
and HNDL definition migration require separate review; legacy routes stay put.
"""
import hashlib
import json
import math
from pathlib import Path

from particle_sliders.lifecycle import (
    Backend, Binding, Capabilities, UnsupportedOperation, checkpoint_cleanup,
)


CAPABILITIES = Capabilities(
    'routed_particle_gan', render=False, strength_min=0., strength_max=1.,
    base_precisions=('bfloat16',), render_outputs=(),
    qualification='Tiny Cosmos CPU lifecycle checks only; pretrained/native parity unqualified',
)
_OPTIONS = {'cache_dir', 'cache_index_sha256', 'model_dir', 'variation', 'device',
            'base_precision', 'steps', 'batch', 'seed', 'ema_decay', 'response_scale',
            'sigma', 'critic_width'}


def validate(options):
    """Allocation-free checks, including gates before native imports/loading."""
    if options.get('device', 'cpu') != 'cpu':
        raise UnsupportedOperation('Anima lifecycle requires CPU; native update admission seam is pending')
    if options.get('base_precision', 'bfloat16') != 'bfloat16':
        raise UnsupportedOperation('Anima preserves its native BF16 base and FP32 trainables')
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


def prepare(options, *, runtime=None):
    """Own provider/bank/game construction and unwind only owned resources.

    A supplied runtime remains caller-owned. Normal installed CLI usage supplies
    model_dir and the factory loads frozen local components itself. Neither path
    captures teachers, discovers a checkpoint or creates a run directory.
    """
    validate(options)
    import torch
    from particle_sliders.pg09 import ConditionalPG09
    from particle_sliders.pg09_backends import AnimaRuntimeResponseAdapter
    from .cache import TargetCache
    from .particles import ParticleAdapter
    from .pg09_training import CachedResponses, NativeFieldCritic

    owns_runtime = runtime is None
    adapter = None
    restore_checkpointing = None
    original_handles = set()
    attachment_started = False
    closed = False
    variation = options['variation']

    def close():
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            mixer = getattr(runtime, 'mixer', None)
            if mixer is not None:
                if adapter is not None and mixer.adapters.get(variation) is adapter:
                    mixer.adapters.pop(variation)
                    mixer.strengths.pop(variation, None)
                # Mixer.add may fail after attaching only part of its hooks.
                if attachment_started and not mixer.adapters:
                    for handle in tuple(mixer.handles):
                        if handle not in original_handles:
                            handle.remove()
                            mixer.handles.remove(handle)
        finally:
            try:
                if restore_checkpointing is not None:
                    restore_checkpointing()
            finally:
                if owns_runtime and runtime is not None:
                    # Also works if native __init__ failed between components.
                    mixer = getattr(runtime, 'mixer', None)
                    if mixer is not None:
                        mixer.close()
                    embeddings = getattr(runtime, 'embeddings', None)
                    if embeddings is not None:
                        embeddings.clear()
                    for name in ('transformer', 'pipe'):
                        vars(runtime).pop(name, None)

    try:
        if runtime is None:
            if not options.get('model_dir'):
                raise ValueError('model_dir is required unless Backend.open receives a staged CPU runtime')
            from .backends.anima import TurboRuntime
            # Retain a reference so a partial native load can be cleaned up.
            runtime = TurboRuntime.__new__(TurboRuntime)
            TurboRuntime.__init__(runtime, options['model_dir'], device='cpu', verify=True, checkpointing=False)
        if runtime.device != torch.device('cpu') or runtime.dtype != torch.bfloat16:
            raise UnsupportedOperation('staged Anima runtime must be CPU with the native BF16 base')
        if variation in runtime.mixer.adapters:
            raise ValueError('Anima session requires its own unused adapter name')
        original_handles = set(runtime.mixer.handles)
        validate(options)  # Native loading must not silently change the cache snapshot.
        # CachedResponses contains/rehashes shard paths before any tensor read.
        cache = TargetCache(options['cache_dir'], verify=False)
        responses = CachedResponses(cache, runtime, variation)
        restore_checkpointing = checkpoint_cleanup(runtime.transformer)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(options.get('seed', 7))
            adapter = ParticleAdapter(runtime.transformer).to(device='cpu', dtype=torch.float32)
            critic = NativeFieldCritic(responses.response_shape[1],
                responses.embedding_width + 4, options.get('critic_width', 32)).to(dtype=torch.float32)
        attachment_started = True
        runtime.mixer.add(variation, adapter)
        backend = AnimaRuntimeResponseAdapter(runtime, adapter, variation)
        game = ConditionalPG09(backend, responses, critic,
            particles=adapter.particles, preset='bispf_optimized',
            response_domain='continuous_field', sampling='balanced',
            batch_size=options.get('batch', 2), seed=options.get('seed', 7),
            total_steps=options.get('steps', 4),
            response_scale=options.get('response_scale', 1.), sigma=options.get('sigma', 1.))
        return Binding(game, close)
    except BaseException:
        close()
        raise


BACKEND = Backend('anima', CAPABILITIES, prepare, validate)
