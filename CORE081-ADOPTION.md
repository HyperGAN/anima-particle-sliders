# Core 0.8.1 local source adoption

Proposed dependency: clean core `32f6830f6402e85aa77b6631db64fcbbb2131f6b`,
version 0.8.1. Product version advances from 0.1.0 to 0.2.0. This replaces only
this isolated derivative proposal's pyproject requirement. The original shared
0.7.0 milestone and Anima's six-check CPU receipt at `bcd519e` remain historical
and unchanged. This new candidate has not run tests, loaded models, installed
dependencies or built a wheel; a new parent CPU slot is required.

The parent explicitly reported independent verification of this exact 0.8.1
candidate (220 reviewer checks and 30 source/hash matches). The canonical owner
handoff/build receipt still carries its earlier `independent rereview pending`
wording; it is preserved verbatim as `validation/core081-canonical-receipt.json`,
not rewritten into a locally invented lock. Static checks compare its complete
30-entry `source_sha256` map to the exact local core copy. Canonical wheel SHA256:
`279aaab83da05b6ebeca030dfe115e1ae2232ca4924fe4278ea48103918fbd8d`.
This derivative has not installed or independently tested that wheel.

Anima's existing proposal is exact PG0.9.1. Canonical `pg09.py`,
`pg09_backends.py`, `formulation.py`, `reference.py` and `_training.py` hashes
match its preserved 0.7.0 milestone. The same actual routed bank, native patch
critic, conditioning, response scale, sigma, `bispf_optimized` recipe, balanced
sampling and bank-then-critic seeded initialization remain. No PG0.9.3 profile
is downgraded and no mathematical formulation is changed. Historical
`requirements*.txt`, `core*.lock.json`, PR5 files and active pins remain intact.

## Source changes

- Training `Backend.preflight` rejects a staged accelerator runtime before core
  preparation device/RNG handling or model allocation. Unknown staged components
  and unqualified training-native render requests are rejected.
- `Binding.before_update` makes a pure CPU/evidence decision before each update.
  It does no external I/O and does not cache or fabricate native admission.
  The core now has the needed admission seam; accelerator execution remains
  gated pending actual derivative admission/parity qualification.
- Shared `NativeOwner` owns local runtime loading and actual bank attachment,
  partial-load/attachment cleanup and borrowed module mode/gradient/checkpoint
  restoration using core's public `native_cleanup` helper.
- Both bindings use a product observation context for state outside the adapter
  parameter wrapper: actual adapter/native module buffers and their registrations,
  modes/gradient/checkpoint flags, scheduler state, mixer strengths and embedding
  cache membership/order. Frozen base parameter values are never cloned per
  observation. Core owns the wrapper parameter/buffer/RNG isolation.
- A separate `particle_sliders.inference_backends` entrypoint exposes
  `lumen_studio.lifecycle:NATIVE_BACKEND`. Its factory constructs only the local
  runtime, real particle adapter and shared response backend. It never calls
  `CachedResponses`, `TargetCache`, a critic, optimizer or ConditionalPG09.
  All native/training factory helpers stay in one inspectable module so the
  core's factory-file hashes cover actual construction and observation code.

## Native-only response protocol

Training's binding supplies the verified native specification automatically.
After a new training cohort is qualified:

```python
session.export_inference('new-run/native-ema.pt', selection='ema')
```

The artifact uses core's v2 native inference envelope. Its specification contains
actual base/source/architecture/protocol identity and the adapter name, BF16
precision, latent channels and embedding width. It contains no cache path,
teacher rows, critic, optimizer or training configuration. The origin receipt
retains the original cohort, training-manifest digest, step, generator arm, PG
version and selected raw/EMA label. It is not a training checkpoint.

```python
from particle_sliders.inference import open_inference
request = dict(latent=local_cpu_latent, embedding=local_cpu_embedding,
               timestep=500.)
with open_inference('new-run/native-ema.pt', operation='response',
                    options=dict(model_dir='/local/verified/anima', device='cpu',
                                 base_precision='bfloat16'),
                    native_request=request) as native:
    velocity = native.response(strength=.5, seed=11, **request)
```

`latent` is a floating CPU tensor `[1,C,1,H,W]`; `embedding` is the already
prepared floating CPU neutral-token tensor `[1,L,D]`; `timestep` is a finite
nonnegative raw native scheduler scalar. Supplied request shapes/channels/width
are checked before loading and again against the actual model. Values and upper
timestep bound are checked before prediction. Local file and device checks run
before loading. Runtime options accept only model_dir/device/base_precision;
teacher cache, training budget and resume options are rejected. The current
response callback is a programmatic tensor API; no CLI JSON-to-tensor file
protocol is claimed. Strength remains [0,1]. References and native image
rendering remain unavailable; no generic sampler or decoder replaces parity.

Legacy v1 `session.export` and its explicit game/cache loader are retained.
No old v1 or native safetensors packet is promoted/rebound to v2. Source and
capability changes require a new cohort and new receipts. The existing test's
inference-step assertion is updated for 0.8.1's earlier typed refusal and usable
session behavior; it has not been rerun against this candidate.

## Next allocated CPU checks and remaining boundaries

Recheck the existing six tests plus v2 raw/EMA export/load with the cache removed
and training constructors trapped, exact native zero/fractional response,
metadata/source/schema failure before mutation, staged accelerator refusal before
CUDA handling, before-update rejection without state/EMA/observer changes, and
success/failure native observation buffer/reference/mode/cache/scheduler/mix
restoration. Keep CPU/offline/one-thread and the parent's resource monitor.

No core defect was established by this source-only adoption. Concrete remaining
limits are the untested derivative hook/v2 factory changes, actual native
accelerator admission and sampling/decode parity, tensor-only native response
input, HNDL parity, and publication/main-integration approval. The canonical
IMAGE-LIFECYCLE.md still describes only v1; NATIVE-INFERENCE.md documents the v2
API used here. Core and all other research worktrees remain untouched.
