"""Explicit fresh/resume integration; no loader, runner or optimizer math here."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from particle_sliders.lifecycle import native_cleanup
from particle_sliders.pg09_backends import AnimaRuntimeResponseAdapter
from particle_sliders.pg095 import ConditionalPG095
from lumen_studio.core_source import verify_core
from lumen_studio.learning_rates import recipe_learning_rates
from lumen_studio.particles import ParticleAdapter
from lumen_studio.pg09_training import PG09Trainer, CachedResponses, NativeFieldCritic
from . import PROFILE
from . import policy

FULL_SCHEMA = 'anima-explicit-fresh95-v2-FULL'
RAW_SCHEMA = 'anima-explicit-fresh95-v2-native-RAW-sidecar'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def source_record(path):
    path = Path(path).absolute()
    raw = path.read_bytes()
    return dict(file=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def source_identity():
    # The old full core closure is independently enforced by verify_core().
    from lumen_studio import particles, pg09_training
    from particle_sliders import pg09_backends, pg09, lifecycle
    return {name: source_record(module.__file__) for name, module in (
        ('fresh_training', __import__(__name__, fromlist=['__file__'])),
        ('fresh_policy', policy), ('particles', particles), ('pg09_training', pg09_training),
        ('pg09_backends', pg09_backends), ('pg09', pg09), ('lifecycle', lifecycle))}


@dataclass(frozen=True)
class Checkpoint:
    """Whole-file pin returned by save(), required for an explicit resume."""
    path: str
    bytes: int
    sha256: str


def file_pin(path):
    path = Path(path).absolute()
    with path.open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return Checkpoint(str(path), size, sha)


def _read_full(pin):
    if (not isinstance(pin, Checkpoint) or type(pin.bytes) is not int or pin.bytes < 1
            or not isinstance(pin.sha256, str) or len(pin.sha256) != 64
            or any(c not in '0123456789abcdef' for c in pin.sha256)):
        raise ValueError('explicit typed whole-file FULL pin required; no legacy migration')
    path = Path(pin.path).absolute()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if before.st_size != pin.bytes or hashlib.file_digest(stream, 'sha256').hexdigest() != pin.sha256:
            raise ValueError('FULL whole-file pin differs')
        stream.seek(0)
        packet = torch.load(stream, map_location='cpu', weights_only=True)
        after = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('FULL changed during read')
    keys = {'schema', 'profile', 'cohort_id', 'options', 'sources', 'initialization', 'game', 'ema_decay'}
    if (not isinstance(packet, dict) or set(packet) != keys or packet['schema'] != FULL_SCHEMA
            or packet['profile'] != PROFILE or packet['ema_decay'] is not None
            or not isinstance(packet['game'], dict)):
        raise ValueError('only this new profile FULL is supported; old cohorts retain original entrypoints')
    return packet


def equal_state(left, right):
    """Typed tensor/container equality; never serialize state through JSON."""
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def _write_new(path, writer):
    """Durable atomic exclusive publication: never overwrite an old checkpoint."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    raw = (canonical(value) + '\n').encode()
    _write_new(path, lambda stream: stream.write(raw))


class ReceiptBackend:
    """Delegate every native method to the exact original core adapter object."""
    def __init__(self, backend, initialization, sources):
        self.backend = backend  # Original class remains in core's Source traversal.
        self.network, self.particles = backend.network, backend.particles
        self._initialization, self._sources = deepcopy(initialization), deepcopy(sources)

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def metadata(self):
        return dict(self.backend.metadata(), fresh_initialization=dict(
            profile=PROFILE, receipt=deepcopy(self._initialization), sources=deepcopy(self._sources)))


class Attachment:
    """Borrow an admitted runtime; restore callbacks and only our attachment."""
    def __init__(self, runtime, variation):
        self.runtime, self.variation = runtime, variation
        self.adapter = None
        self.closed = False
        self.closing = False
        self.cleanup_pending = False
        self._native_restore_pending = True
        if not isinstance(variation, str) or not variation.strip() or variation in runtime.mixer.adapters:
            raise ValueError('explicit unused adapter name required')
        self.original_handles = set(runtime.mixer.handles)
        # Binding the original backend changes native checkpoint callbacks.
        self.restore_native = native_cleanup(runtime.transformer)

    def attach(self, adapter):
        self.adapter = adapter
        self.runtime.mixer.add(self.variation, adapter)
        return AnimaRuntimeResponseAdapter(self.runtime, adapter, self.variation)

    def close(self):
        if self.closed:
            return
        # Closing immediately prevents further training/inference, but a failed
        # cleanup remains retryable. Never discard the handle/callback owners.
        self.closing = True
        errors = []
        try:
            mixer = self.runtime.mixer
            if self.adapter is not None and mixer.adapters.get(self.variation) is self.adapter:
                mixer.adapters.pop(self.variation)
                mixer.strengths.pop(self.variation, None)
            if not mixer.adapters:
                for handle in tuple(mixer.handles):
                    if handle not in self.original_handles:
                        try:
                            handle.remove()
                            mixer.handles.remove(handle)  # Only after successful removal.
                        except BaseException as error:
                            errors.append(error)
        except BaseException as error:
            errors.append(error)
        if self._native_restore_pending:
            try:
                self.restore_native()
                self._native_restore_pending = False
            except BaseException as error:
                errors.append(error)
        self.cleanup_pending = bool(errors)
        if errors:
            raise BaseExceptionGroup('owned Anima cleanup failures; retry close()', errors)
        self.closed, self.closing = True, False


class FreshTrainer(PG09Trainer):
    """Versioned explicit new cohort; inherited update keeps original PG formula.

    No call to PG09Trainer.__init__: that historical entry auto-restores a bare
    packet after constructing a game. Here intent and FULL metadata are resolved
    before any added adapter/critic, and initialization precedes optimizers.
    """
    def __init__(self, runtime, cache, directory, variation, *, mode, cohort_id,
                 resume=None, seed=7, batch_size=2, total_steps=100,
                 response_scale=1., sigma=1., critic_width=32,
                 native_admission=None, lr=None, d_lr_mult=None,
                 prior_lr_mult=None, numerical_profile=None, ema_decay=None):
        # All intent/packet/config checks precede the first component constructor.
        if mode not in ('fresh', 'resume') or not isinstance(cohort_id, str) or not cohort_id.strip():
            raise ValueError('explicit fresh|resume mode and new-profile cohort_id required')
        if (mode == 'fresh' and resume is not None) or (mode == 'resume' and resume is None):
            raise ValueError('fresh refuses saved state; resume requires an exact new-profile FULL')
        if ema_decay is not None:
            raise ValueError('weight EMA is disabled; RAW only')
        if (type(seed) is not int or not 0 <= seed < 2**63 - 100
                or type(batch_size) is not int or not 2 <= batch_size <= 512
                or type(total_steps) is not int or total_steps < 1
                or type(critic_width) is not int or not 1 <= critic_width <= 512
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) or x <= 0 for x in (response_scale, sigma))):
            raise ValueError('invalid explicit new-profile construction options')
        self.directory = Path(directory).absolute()
        if self.directory.exists():
            raise FileExistsError('new output directory required; no implicit restore/overwrite')
        if runtime.device.type != 'cpu' and not callable(native_admission):
            raise ValueError('native runtime requires owner admission before component construction')
        learning_rates = recipe_learning_rates(lr=lr, d_lr_mult=d_lr_mult, prior_lr_mult=prior_lr_mult)
        self.options = dict(seed=seed, batch_size=batch_size, total_steps=total_steps,
            response_scale=response_scale, sigma=sigma, critic_width=critic_width,
            learning_rates=learning_rates, numerical_profile=numerical_profile, ema_decay=None,
            variation=variation)
        verify_core()  # Exact original core guard; no release/source-profile relaxation.
        self.sources = source_identity()
        packet = _read_full(resume) if mode == 'resume' else None
        if packet is not None:
            if (packet['cohort_id'] != cohort_id or packet['options'] != self.options
                    or packet['sources'] != self.sources):
                raise ValueError('FULL options/cohort/Source profile differs')
            receipt = packet['initialization']
            if (not isinstance(receipt, dict) or receipt.get('schema') != policy.SCHEMA
                    or receipt.get('policy') != policy.POLICIES['anima']
                    or receipt.get('cohort_id') != cohort_id or receipt.get('seed') != seed
                    or receipt.get('source_identity') != self.sources
                    or receipt.get('completed_steps') != 0 or receipt.get('restored_from') is not None
                    or receipt.get('selection') != 'raw' or receipt.get('ema_decay') is not None):
                raise ValueError('FULL lacks its original new-profile initialization receipt')
        self.runtime, self.variation, self.cohort_id = runtime, variation, cohort_id
        self.responses = CachedResponses(cache, runtime, variation, numerical_profile=numerical_profile)
        self._frozen_identity = self.responses.identity()
        self._native_admission, self._native_admitted = native_admission, False
        self._attachment = Attachment(runtime, variation)
        try:
            devices = [runtime.device.index or 0] if runtime.device.type == 'cuda' else []
            # Original bank→branches→critic constructor order and host laws stay.
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                self.adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
                critic = NativeFieldCritic(self.responses.response_shape[1],
                    self.responses.embedding_width + 4, critic_width).to(runtime.device)
            if mode == 'fresh':
                receipt = policy.initialize('anima', self.adapter, critic,
                    construction=policy.Construction('fresh', cohort_id,
                        hashlib.sha256(policy.canonical(self.options).encode()).hexdigest()),
                    options=self.options, seed=seed, source_identity=self.sources,
                    particles=self.adapter.particles, optimizers=())
            # Resume does not call initialize. Reuse the saved receipt unchanged.
            self.initialization = deepcopy(receipt)
            original = self._attachment.attach(self.adapter)
            self.backend = ReceiptBackend(original, self.initialization, self.sources)
            # This is the first optimizer construction, after fresh scoped init.
            self.game = ConditionalPG095(self.backend, self.responses, critic,
                particles=self.adapter.particles, preset='bispf_optimized',
                response_domain='continuous_field', batch_size=batch_size, seed=seed,
                total_steps=total_steps, response_scale=response_scale, sigma=sigma,
                sampling='balanced', recipe_overrides=learning_rates)
            if mode == 'resume':
                self.game.load_state_dict(packet['game'])
                if not equal_state(self.game.state_dict(), packet['game']):
                    raise RuntimeError('strict complete game/optimizer/RNG restore equality failed')
            self.directory.mkdir(parents=True, exist_ok=False)
            parent_fd = os.open(self.directory.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            write_json(self.directory / 'initialization.json', self.initialization)
            # A new explicit entry writes its current state before any update.
            # On resume the number remains the inherited cursor; no update replay.
            self.entry_FULL = self.save()
            self.entry_RAW = self.export(self.directory / f'RAW{self.step:06d}.safetensors')
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup('Anima entry and cleanup failed', [error, cleanup])
            raise

    def update(self):
        if self._attachment.closed or self._attachment.closing:
            raise RuntimeError('training owner is closed')
        return super().update()

    def save(self):
        self.responses.identity()
        packet = dict(schema=FULL_SCHEMA, profile=PROFILE, cohort_id=self.cohort_id,
            options=deepcopy(self.options), sources=deepcopy(self.sources),
            initialization=deepcopy(self.initialization), game=self.game.state_dict(), ema_decay=None)
        path = self.directory / f'FULL{self.step:06d}.pt'
        _write_new(path, lambda stream: torch.save(packet, stream))
        return file_pin(path)

    def restore(self, path):
        raise ValueError('restore only through explicit open_training(mode="resume", resume=Checkpoint(...))')

    def export(self, path):
        # Reuse original native export law/normalization and RAW state format.
        path = Path(path)
        if any(p.exists() for p in (path, path.with_suffix('.pg09.json'), path.with_suffix('.fresh95.json'))):
            raise FileExistsError('new RAW and sidecar paths required')
        sha = super().export(path)
        # The original native exporter owns format/math; this new API also
        # makes its RAW and original sidecar durable before publishing ours.
        for artifact in (path, path.with_suffix('.pg09.json')):
            with artifact.open('rb') as stream:
                os.fsync(stream.fileno())
        sidecar = dict(schema=RAW_SCHEMA, profile=PROFILE, sources=deepcopy(self.sources),
            initialization=deepcopy(self.initialization), options=deepcopy(self.options),
            cohort_id=self.cohort_id, native_RAW_sha256=sha, completed_steps=self.step,
            model_identity=deepcopy(self._frozen_identity['runtime']),
            cache_identity=deepcopy(self._frozen_identity), selection='raw', ema_decay=None,
            historical_LR_measurement_transferred=False, quality_qualified=False)
        write_json(Path(path).with_suffix('.fresh95.json'), sidecar)
        return file_pin(path)

    def close(self):
        owner = getattr(self, '_attachment', None)
        if owner is not None:
            owner.close()
