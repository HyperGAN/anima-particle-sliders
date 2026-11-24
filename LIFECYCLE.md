# Anima image sessions

Anima 0.4.0 installs core 0.15.2 and ParticleGAN 0.11.1. The registered
`anima` backend is CPU-only and supports training and numerical response.
The separately admitted native training API is
`lumen_studio.engine_training.EngineTrainer`.

Both entry points accept positive `lr`, `d_lr_mult` and `prior_lr_mult`.
Omitted rates use the existing explicit `bispf_optimized` recipe. The
resolved recipe and runtime source are included in checkpoint identity.
Use `ema_decay=None` in common session options and select RAW for inference.

The shared core also exposes EMPF as an explicit preset for clients such as
Nova. For exact replay of packets created before that core update, use the
[initial ParticleGAN 0.11.1 checkout](https://github.com/HyperGAN/anima-particle-sliders/tree/5ab590b30d17423652ff851fd1d8246cc364df89).

See [scoped initialization](docs/explicit-fresh-initialization.md) for fresh
runs and exact resume. Earlier artifacts use their original checkout and
dependency pins.

For historical reproduction, use the [original cohort checkout](https://github.com/HyperGAN/anima-particle-sliders/tree/791a80a4f38a9b8cb096d719b6758a79325bd7d2).

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

Current ParticleGAN sessions set `ema_decay=None` and use RAW weights for
training, evaluation, export and inference. Do not add EMA averaging,
comparisons or selection. Historical published EMA snapshots and their legacy
loaders retain their original provenance.

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
    steps=4, batch=2, seed=7, critic_width=32, ema_decay=None)
with open_session(config) as session:
    session.train(stop_after=2, on_step=hypergan_observer)
    session.checkpoint('new-run/training.pt')
with open_session(config, resume='new-run/training.pt') as session:
    session.train()  # Original four-update budget, not four additional updates.
    session.export('new-run/adapter.pt', selection='raw')
with open_session(config, artifact='new-run/adapter.pt', operation='evaluate') as session:
    velocity = session.evaluate(index=0, seed=11, strength=.5)
```

`hypergan_observer` is an externally supplied run observer. The product does
not create a second dashboard. The common installed `particle-slider`
capabilities/train/evaluate commands consume `config.as_dict()` as JSON.
The bounded test job uses the existing real tiny random Cosmos fixture, not a
pretrained CPU load. It exercises registered `open_session` and the owned-loader
branch with a synthetic loader; native staged weight loading remains untested.

Public strength support is finite [0,1]. Base precision is BF16, all generator
and critic trainables FP32. Native geometry is `[1,C,1,H,W]`; evaluation returns
velocities. Conditioning, exact routed bank ownership, native time normalization,
balanced sampling and `bispf_optimized` remain as in the explicit ParticleGAN trainer.
Other devices, signed/out-of-range strengths, reference images and rendering
are rejected before model allocation through the public preflight.

Rendering is unavailable pending native 8/10/12-step Turbo scheduler/rounding,
CFG1 and decoder parity. Original Studio/release/Comfy inference continues to
use its own selected native safetensors/alpha/LoRA entrypoints. A session packet
is a separate versioned format; do not feed it to those historical loaders.
The new velocity-response callback does not substitute for native image sampler
or decoder qualification.

## Strict identity and remaining seams

Training and inference session packets differ and bind the complete prepared
configuration/cohort, capabilities, factory/core/product source bytes, native
model/environment, verified frozen cache, architecture and PG recipe. Corrupt,
legacy or changed-cohort packets are rejected; no implicit rebinding is offered.
The explicit v1 loader still constructs the prepared game/cache. Core 0.8.1
also supplies a distinct native-only v2 API, now wired in this proposal; see
`CORE081-ADOPTION.md`. Its bounded raw/EMA native-only checks now pass; see
`CORE081-CPU.md` for the exact installed qualification scope.

Core 0.8.1 now supplies `Binding.before_update`; this factory wires a pure
CPU/evidence check. The original Anima native trainer requires actual model/cache
teacher-parity admission for accelerators, so non-CPU execution remains gated
until separately qualified.
Custom `NativeFieldCritic`, low-rank `Branch` and routed bridge definitions
require a separate matched-parameter HNDL migration and PG update parity review.
The frozen pretrained backbone remains native.

See `LIFECYCLE-PLAN.md` for entrypoint audit, migration boundaries and the
parent-scheduled conformance matrix. Six bounded lifecycle tests and extracted
wheel registration passed at 0.7.0; the subsequent installed 0.8.1 slot passed
32 lifecycle/native-inference/PR5 regression checks without implementation edits. Full native runtime/environment loading, sampler
parity, pretrained convergence/quality and native memory remain unqualified.
No GPU work, model download, dependency installation, publication, merge,
deployment, system change or active pin change occurred.
