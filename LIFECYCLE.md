# Anima image sessions: proposed new cohort

This local integration targets the exact core 0.7.0 milestone
`fd89e1c780ab5210cd1be98ec3b70977e6349379`. It is based on draft PR5
`4477716c8dbedd3a079e3299a1229adaaa2b407f`, preserving its existing shared
PG09 trainer and early legacy-dispatch safeguards. It is unpublished and
unqualified. Source inspection and static checks are the only current evidence;
installed-wheel CPU conformance must be scheduled by the parent.

The new `anima-particle-sliders` distribution registers backend `anima` in
`particle_sliders.image_backends`. It depends on Particle Sliders, which owns
the ParticleGAN dependency and GAN math. There is no reverse product import or
copied GAN objective, regularizer, prior, optimizer or stepping loop.

`requirements.txt`, `core.lock.json`, `requirements-pg09.txt` and
`core.pg09.lock.json` retain their original historical pins and qualification
states. Select one isolated new environment for this proposal. Do not install
the historical dependency selections with this package or retarget active runs.
The exact proposed shared commit is local, not yet a publicly fetchable release.
Existing native runtime environment checks remain authoritative; the optional
`native` dependencies are not a qualified full environment lock.

## Factory and operations

`lumen_studio.lifecycle.BACKEND` validates local configuration, then owns loading
the frozen local Turbo runtime, verifying the existing training cache, creating
and attaching its real routed bank and native patch critic, and constructing
the shared game. The caller supplies a configuration, not model internals.
For an admitted local integration fixture only,
`BACKEND.open(config, runtime=already_staged_cpu_runtime)` can borrow a runtime;
the factory still owns every training component. Borrowed runtime adapters,
mixes and checkpoint settings are preserved, while session-owned attachments
are removed on close or setup failure. No checkpoint is discovered implicitly.

Example configuration for a future separately admitted CPU run:

```python
from hashlib import sha256
from pathlib import Path
from particle_sliders.lifecycle import SessionConfig, open_session

cache = Path('/already/verified/anima/train-cache')
config = SessionConfig('anima', 'new-anima-image-v1-cohort',
    model_dir='/already/staged/anima-turbo-v1.1', cache_dir=str(cache),
    cache_index_sha256=sha256((cache / 'index.json').read_bytes()).hexdigest(),
    variation='finalform', device='cpu', base_precision='bfloat16',
    steps=4, batch=2, seed=7, critic_width=32, ema_decay=.9)
with open_session(config) as session:
    session.train(stop_after=2, on_step=hypergan_observer)
    session.checkpoint('new-run/training.pt')
with open_session(config, resume='new-run/training.pt') as session:
    session.train()  # Original four-update budget, not four additional updates.
    session.export('new-run/adapter.pt', selection='ema')
with open_session(config, artifact='new-run/adapter.pt', operation='evaluate') as session:
    velocity = session.evaluate(index=0, seed=11, strength=.5)
```

`hypergan_observer` is an externally supplied run observer. The product does
not create a second dashboard. The common installed `particle-slider`
capabilities/train/evaluate commands consume `config.as_dict()` as JSON.
No commands have been executed for this proposal. The small test job should
use the existing real tiny random Cosmos fixture, not a pretrained CPU load.

Public strength support is finite [0,1]. Base precision is BF16, all generator
and critic trainables FP32. Native geometry is `[1,C,1,H,W]`; evaluation returns
velocities. Conditioning, exact routed bank ownership, native time normalization,
balanced sampling and `bispf_optimized` remain as in the explicit PG09 trainer.
Other devices, signed/out-of-range strengths, reference images and rendering
are rejected before model allocation through the public preflight.

Rendering is unavailable pending native 8/10/12-step Turbo scheduler/rounding,
CFG1 and decoder parity. Original Studio/release/Comfy inference continues to
use its own selected native safetensors/alpha/LoRA entrypoints. A session packet
is a separate versioned format; do not feed it to those historical loaders.
No native callback or generic solver is substituted for missing qualification.

## Strict identity and remaining seams

Training and inference session packets differ and bind the complete prepared
configuration/cohort, capabilities, factory/core/product source bytes, native
model/environment, verified frozen cache, architecture and PG recipe. Corrupt,
legacy or changed-cohort packets are rejected; no implicit rebinding is offered.
At this milestone loading an inference packet still constructs the prepared
game/cache. Compact inference is being fixed by the core owner separately.

Core's `Binding` has no pre-update native admission callback. The original
Anima trainer requires one for exact runtime/cache teacher parity before a
non-CPU update; this factory gates non-CPU execution until that seam is fixed.
Custom `NativeFieldCritic`, low-rank `Branch` and routed bridge definitions
require a separate matched-parameter HNDL migration and PG update parity review.
The frozen pretrained backbone remains native.

See `LIFECYCLE-PLAN.md` for entrypoint audit, migration boundaries and the
parent-scheduled conformance matrix. Pretrained convergence/quality, native
memory, installed packaging and this new lifecycle's runtime behavior are not
yet qualified. No GPU work, model download, package installation, publication,
merge, deployment, system change or active pin change is part of this phase.
