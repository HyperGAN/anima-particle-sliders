"""RAW loading with explicit native owner; never builds a game or initializes."""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path

import torch

from lumen_studio.core_source import verify_core
from lumen_studio.lifecycle import NativeOwner
from lumen_studio.particles import ParticleAdapter
from . import PROFILE, policy
from .training import Attachment, Checkpoint, RAW_SCHEMA, file_pin, source_identity


class InferenceOwner(Attachment):
    """Attach native product RAW after versioned sidecar validation.

    The caller owns runtime admission and the existing product renderer. This API
    loads no pretrained model, teacher cache, critic or optimizer. It offers the
    unchanged native observation and scale contexts around the caller's render.
    """
    def __init__(self, runtime, variation, *, raw, owner_admission=None):
        if not isinstance(raw, Checkpoint) or file_pin(raw.path) != raw:
            raise ValueError('exact native RAW whole-file pin required')
        verify_core()
        sidecar = json.loads(Path(raw.path).with_suffix('.fresh95.json').read_text())
        if (sidecar.get('schema') != RAW_SCHEMA or sidecar.get('profile') != PROFILE
                or sidecar.get('native_RAW_sha256') != raw.sha256
                or sidecar.get('sources') != source_identity()
                or sidecar.get('selection') != 'raw' or sidecar.get('ema_decay') is not None
                or sidecar.get('model_identity') != runtime.identity
                or not isinstance(sidecar.get('initialization'), dict)):
            raise ValueError('new-profile native RAW provenance differs')
        receipt = sidecar['initialization']
        if (receipt.get('schema') != policy.SCHEMA or receipt.get('policy') != policy.POLICIES['anima']
                or receipt.get('source_identity') != sidecar['sources']
                or receipt.get('cohort_id') != sidecar.get('cohort_id')
                or receipt.get('seed') != sidecar.get('options', {}).get('seed')
                or receipt.get('completed_steps') != 0 or receipt.get('restored_from') is not None
                or receipt.get('selection') != 'raw' or receipt.get('ema_decay') is not None
                or type(sidecar.get('completed_steps')) is not int or sidecar['completed_steps'] < 0):
            raise ValueError('native RAW original initialization receipt differs')
        if runtime.device.type != 'cpu' and not callable(owner_admission):
            raise ValueError('native inference requires an external owner admission callback')
        self.raw, self.sidecar, self.owner_admission = raw, sidecar, owner_admission
        super().__init__(runtime, variation)
        try:
            devices = [runtime.device.index or 0] if runtime.device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(sidecar['options']['seed'])
                adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
            # Product RAW format remains paired with its exact native loader.
            # No policy.initialize, game, teacher or optimizer on this route.
            self.native_metadata = adapter.load_export(raw.path, model_identity=runtime.identity)
            native = self.native_metadata
            cache_index = sidecar['cache_identity']['index']
            if (native['weights'] != 'live' or native['step'] != sidecar['completed_steps']
                    or native['manifest_sha256'] != cache_index['identity']['manifest_sha256']
                    or native['normalization']['cache_fingerprint'] != cache_index['fingerprint']):
                raise ValueError('native RAW metadata and profile sidecar differ')
            self.backend = self.attach(adapter)
            self.adapter.eval()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup('Anima inference entry and cleanup failed', [error, cleanup])
            raise

    @contextmanager
    def scaled(self, strength):
        if (isinstance(strength, bool) or not isinstance(strength, (int, float))
                or not math.isfinite(strength) or not 0 <= strength <= 1):
            raise ValueError('explicit native inference strength must be in [0,1]')
        if self.closed or self.closing:
            raise RuntimeError('inference owner is closed')
        if self.runtime.device.type != 'cpu':
            request = dict(operation='native-inference', profile=PROFILE,
                RAW=dict(path=self.raw.path, bytes=self.raw.bytes, sha256=self.raw.sha256),
                model_identity=deepcopy(self.runtime.identity), strength=float(strength))
            if self.owner_admission(request) is not True:
                raise ValueError('external owner did not admit native inference')
        # Reuse the product's exact native state/buffer/scheduler observation.
        with NativeOwner.observation(self):
            with self.backend.network.scaled(float(strength)):
                yield self.runtime
