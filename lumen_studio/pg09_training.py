"""Explicit Anima PG0.9 training binding; GPU/runtime admission stays external.

The product owns its frozen runtime, cache, critic architecture and routed
adapter. The installed particle_sliders package owns every GAN update,
optimizer, constraint, sampler and resumable training packet.
"""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_backends import (
    AnimaRuntimeResponseAdapter, PreparedResponse, ResponseRow,
)

from .cache import TargetCache, save_tensor_file
from .contracts import atomic_json, digest, file_hash
from .particles import ParticleAdapter
from .provenance import cache_runtime_compatibility


class CachedResponses:
    """Lazy, content-bound native velocity fields from the existing TargetCache.

    Conditions are pooled neutral embeddings, normalized native time, denoise
    position, frozen trajectory choice and the +1 training endpoint. No row ID,
    seed, definition ID or character ID is presented to the critic.
    """

    def __init__(self, cache, runtime, variation):
        self.runtime = runtime
        identity = cache.index['identity']
        if identity.get('split') != 'train' or identity.get('variation') != variation:
            raise ValueError('PG09 requires the matching training cache')
        if len(cache) < 2:
            raise ValueError('PG09 requires at least two frozen response rows')
        self.compatibility = cache_runtime_compatibility(
            getattr(runtime, 'model_dir', None), identity['model'], runtime.identity,
            cache.index['fingerprint'])
        # TargetCache historically trusts its path strings. Resolve and contain
        # every path before reading bytes, including symlinks and absolute paths.
        root = cache.directory.resolve()
        for shard in cache.index['shards']:
            relative = Path(shard['path'])
            path = (root / relative).resolve()
            if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
                raise ValueError('PG09 target shard path escapes the cache directory')
            if file_hash(path) != shard['sha256']:
                raise ValueError('Corrupt PG09 target shard: ' + shard['path'])
        # Own a private verified reader. A caller changing its mutable cached
        # records/offsets/index cannot silently change this game's frozen law.
        self.cache = TargetCache(root, verify=True, max_shards=cache.max_shards)
        if self.cache.index != cache.index:
            raise ValueError('PG09 target cache index differs from its committed bytes')
        self._index = deepcopy(self.cache.index)
        self._offsets = deepcopy(self.cache.offsets)
        if 'input_fingerprint' in self._index:
            input_hash = digest(self._index['identity'])
            if (input_hash != self._index['input_fingerprint']
                    or digest(dict(input=input_hash, shards=self._index['shards'])) != self._index['fingerprint']):
                raise ValueError('PG09 target index fingerprint differs from its shard manifest')
        self._identity = dict(schema='anima-pg09-cache-v1', index=deepcopy(self._index),
            runtime=deepcopy(runtime.identity), teacher_runtime_compatibility=self.compatibility,
            cache_implementation_sha256=file_hash(Path(__file__).with_name('cache.py')),
            condition_features='mean-neutral-embedding,time,position,trajectory,scale+1')
        record = self.cache[0]
        self.response_shape = tuple(record['positive'].shape)
        self.embedding_width = int(record['embedding'].shape[-1])
        self._validate(record)

    def __len__(self):
        return len(self.cache)

    def identity(self):
        self._guard_index()
        return deepcopy(self._identity)

    def _guard_index(self):
        if self.cache.index != self._index or self.cache.offsets != self._offsets:
            raise ValueError('PG09 frozen cache identity or record ordering changed')

    def _validate(self, record):
        if len(self.response_shape) != 5 or self.response_shape[0] != 1 or self.response_shape[2] != 1:
            raise ValueError('Anima PG09 requires genuine native [1,C,1,H,W] fields')
        for name in ('latent', 'positive', 'neutral'):
            value = record[name]
            if tuple(value.shape) != self.response_shape or value.requires_grad or not torch.isfinite(value).all():
                raise ValueError('PG09 cache has a malformed native ' + name)
        embedding = record['embedding']
        if (embedding.ndim != 3 or embedding.shape[0] != 1 or embedding.shape[1] < 1
                or embedding.shape[-1] != self.embedding_width or embedding.requires_grad
                or not torch.isfinite(embedding).all()):
            raise ValueError('PG09 requires frozen finite neutral token embeddings')
        timestep = float(record['timestep'])
        if not 0 <= timestep <= self._time_scale():
            raise ValueError('PG09 cache time must be a native scheduler timestep')
        if record['trajectory'] not in ('neutral', 'positive') or not 0 <= record['position'] < 10:
            raise ValueError('PG09 cache path or denoise position is invalid')

    def _time_scale(self):
        return float(self.runtime.pipe.scheduler.config.num_train_timesteps)

    def __getitem__(self, index):
        self._guard_index()
        record = self.cache[index]
        self._validate(record)
        device = self.runtime.device
        clone = lambda value: value.detach().to(device=device).clone()
        # These teachers were evaluated at this exact latent/time with all
        # adapters disabled. They are read, never recomputed under a live mix.
        row = ResponseRow(neutral=clone(record['embedding']), positive=None, negative=None,
            latent=clone(record['latent']),
            timestep=torch.tensor(float(record['timestep']), device=device), scale=1., guidance=1.)
        return PreparedResponse(row, clone(record['positive']), clone(record['neutral']))

    def conditions(self, indices):
        self._guard_index()
        values = []
        for index in indices.tolist():
            record = self.cache[index]
            self._validate(record)
            context = record['embedding'].float().mean(dim=1).squeeze(0)
            extra = context.new_tensor([float(record['timestep']) / self._time_scale(),
                record['position'] / 9., float(record['trajectory'] == 'positive'), 1.])
            values.append(torch.cat((context, extra)))
        return torch.stack(values)


class NativeFieldCritic(nn.Module):
    """Conditional local-patch critic with released PG0.9-compatible weights.

    Anima has one time plane. Unfolding only inside the critic retains native
    fields at the PG penalty boundary; the public DualNorm optimizer sees
    supported rank-two Linear parameters rather than unsupported Conv3d ones.
    """

    def __init__(self, channels, condition_width, width=32):
        super().__init__()
        self.field = nn.Sequential(nn.Linear(channels * 9, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(condition_width, width), nn.SiLU())
        self.logits = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, response, condition):
        patches = torch.nn.functional.unfold(response.squeeze(2), kernel_size=3, padding=1).transpose(1, 2)
        field = self.field(patches).mean(dim=1)
        return self.logits(torch.cat((field, self.context(condition)), dim=1)).squeeze(-1)


class PG09Trainer:
    """Callable training entrypoint for an already admitted and loaded runtime.

    It never loads a model, launches a GPU job or dispatches coordinator work.
    Archived Trainer/Game remain a separate historical reproduction interface.
    """

    def __init__(self, runtime, cache, directory, variation, *, seed=7, batch_size=2,
                 total_steps=100, response_scale=1., sigma=1., critic_width=32,
                 native_admission=None):
        self.runtime, self.variation = runtime, variation
        self.directory = Path(directory)
        self.responses = CachedResponses(cache, runtime, variation)
        self.cache = self.responses.cache
        self._frozen_identity = self.responses.identity()
        self._native_admission, self._native_admitted = native_admission, False
        if runtime.device.type != 'cpu' and not callable(native_admission):
            raise ValueError('Native PG09 updates require an external admission callback with verified teacher parity')
        if variation in runtime.mixer.adapters:
            raise ValueError('PG09 requires its own adapter name in the admitted runtime')
        devices = [runtime.device.index or 0] if runtime.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
            critic = NativeFieldCritic(self.responses.response_shape[1],
                self.responses.embedding_width + 4, critic_width).to(runtime.device)
        runtime.mixer.add(variation, self.adapter)
        try:
            self.backend = AnimaRuntimeResponseAdapter(runtime, self.adapter, variation)
            self.game = ConditionalPG09(self.backend, self.responses, critic,
                particles=self.adapter.particles, preset='bispf_optimized',
                response_domain='continuous_field', batch_size=batch_size, seed=seed,
                total_steps=total_steps, response_scale=response_scale, sigma=sigma,
                sampling='balanced')
            self.directory.mkdir(parents=True, exist_ok=True)
            if (self.directory / 'pg09-resume.pt').exists():
                self.restore(self.directory / 'pg09-resume.pt')
        except BaseException:
            self.close()
            raise

    @property
    def step(self):
        return self.game.completed_steps

    def update(self):
        self._admit_native_update()
        return self.game.step()

    def _admit_native_update(self):
        if self.runtime.device.type == 'cpu' or self._native_admitted:
            return
        request = dict(model_identity=deepcopy(self._frozen_identity['runtime']),
            cache_fingerprint=self._frozen_identity['index']['fingerprint'],
            response_shape=list(self.responses.response_shape))
        admitted = self._native_admission(deepcopy(request))
        if admitted is not True:
            raise ValueError('External native admission must verify exact model/cache teacher parity')
        atomic_json(self.directory / 'pg09-admission.json', dict(request, external_admission=True))
        self._native_admitted = True

    def save(self):
        """Save the shared core's exact optimizer/sampler/constraint/RNG packet."""
        save_tensor_file(self.directory / 'pg09-resume.pt', self.game.state_dict())

    def restore(self, path):
        packet = torch.load(path, map_location='cpu', weights_only=True)
        self.game.load_state_dict(packet)

    def export(self, path):
        """Native portable live weights, usable by the existing inference paths."""
        path = Path(path)
        self.responses.identity()  # Reject changes to the frozen index/order.
        config = self.game.state_dict()['config']  # Reject incomplete failed updates before writing.
        frozen_index = self._frozen_identity['index']
        normalization = dict(method='pg09-native-field-scalar-scale',
            response_scale=self.game.response_scale, cache_fingerprint=frozen_index['fingerprint'])
        sha = self.adapter.export(path, model_identity=self._frozen_identity['runtime'],
            normalization=normalization,
            manifest_hash=frozen_index['identity']['manifest_sha256'], step=self.step)
        atomic_json(path.with_suffix('.pg09.json'), dict(schema='anima-pg09-export-v1',
            sha256=sha, weights='live', completed_steps=self.step,
            objective=config['objective'],
            shared_training_config_sha256=digest(config),
            quality_qualified=False))
        return sha

    def close(self):
        if self.runtime.mixer.adapters.get(self.variation) is getattr(self, 'adapter', None):
            self.runtime.mixer.adapters.pop(self.variation)
            self.runtime.mixer.strengths.pop(self.variation, None)
