# Installed Anima core 0.8.1 qualification

Qualified implementation: `5ecda1b5f4b5960bc1a25f05f5ac2923ede16fb4`.
The follow-up commit adds tests, documentation and receipts only; runtime and
package metadata bytes remain exactly those of that adoption. Core stays at
`32f6830f6402e85aa77b6631db64fcbbb2131f6b`, version 0.8.1, with PG0.9.1.
No newer core or checkpoint rebinding was needed. No implementation defect was
found, so no implementation patch was made.

## Executed results

| Scope | Passed | Pytest elapsed | Process elapsed |
| --- | ---: | ---: | ---: |
| Existing lifecycle/resume/EMA/observation/cleanup checks | 6 | 5.34 s | 9.20 s |
| V2 native-only inference, hooks and installed legacy CLI | 10 | 3.86 s | 5.89 s |
| Preserved PR5 CLI/API/worker dispatch-order safeguards | 16 | 1.03 s | 3.02 s |

32 passed, no failures or skips. The first group validates exact disk resume
including complete game, EMA, metrics, coverage and RNG state; frozen base and
actual routed bank ownership; observation RNG/modes/zero/fractional response;
invalid checksum/EMA/cohort rejection before mutation; borrowed resources and
setup/update/partial-loader cleanup. It is the same six-check module previously
qualified at 0.7.0, now actually executed against the installed 0.8.1 candidate.

The ten new cases establish:

- Raw and EMA v2 exports load with the entire teacher cache removed while
  ConditionalPG09, TargetCache, CachedResponses and NativeFieldCritic constructors
  are trapped. Loaded adapter values, exact zero and fractional velocity outputs
  match their selected training raw/EMA values, with no game/rows/critic/optimizer
  on the inference session. Train/checkpoint/restore remain unavailable.
- Staged accelerator requests stop before native loading or RNG/CUDA device
  handling. Native packet checksum/factory identity, geometry, reference and
  strength failures stop before allocation; malformed network state rejects
  before mutation and closes owned native attachments.
- Before-update rejection leaves the session usable and preserves game/EMA,
  global RNG and observer calls. The original pure evidence hook accepts only
  the still-qualified CPU case and rejects accelerator drift.
- On both successful and failed observations, training and native-only sessions
  restore original native buffer objects/values/registrations/persistence,
  modes/gradient flags, scheduler state/object, mixer state/object and embedding
  cache membership/order/object. The caller RNG is preserved.
- Installed `lumen_studio` train/benchmark CLI paths still refuse legacy training
  before output creation.

All 158 files in PR5 `4477716c8dbedd3a079e3299a1229adaaa2b407f` remain
byte-identical. Its original 16 dispatch tests also passed: CLI/release/campaign
refusal before runtime/cache/filesystem/device work, HTTP 409 before training
store/cache reads, and worker lease/recovery handling that preserves inference.
Only the release/campaign wrappers use their preserved source files because they
are not packaged commands; all loaded lumen_studio modules and all core modules
in that run remained the installed wheel versions. No actual MOSS preparation
or native training dispatch occurred.

## Installation, identity and resources

A pure-Python product wheel was built from exact adoption `5ecda1b`. It and the
canonical local core wheel were installed with `pip --no-index --no-deps --target`
into this task's fresh `../evidence/core081-qualification/site` directory. The
existing compatible Python 3.11 environment supplies Torch 2.11.0+cu128,
Diffusers 0.41.0.dev0 and released PG0.9.1. No global/shared environment was
modified and no dependency resolution or downloads occurred. This is installed
wheel qualification with existing compatible dependencies, not a clean resolver
or real native environment-lock qualification.

Product wheel SHA256:
`264671c876909afb46c43f330f8b4959ea1b250ed966d98cac9d722a1ed517ec`.
Canonical core wheel SHA256:
`279aaab83da05b6ebeca030dfe115e1ae2232ca4924fe4278ea48103918fbd8d`.
Every one of the 30 canonical core Python/JSON hashes was checked on each test
process. Released PG source fingerprint remained
`04624298310cd4b28d5bf687a1eac8951efd9439fb0310709ba5b9703208ceab`.
Packaged product Python/config members were also compared to the exact adoption.

Each test process ran from `/tmp` under `python -I`, with the fresh installed
wheel target first on the import path. Source injection was disabled for
installed lifecycle/inference tests, apart from test fixture modules themselves.
Both registrations resolved to the installed product. The separately classified
PR5 wrapper regression run adds only its preserved command-wrapper source root;
post-run module-origin assertions establish installed product/core ownership.

One subprocess tree ran at a time. A 50 ms resource monitor stopped jobs above
2 GiB sampled tree RSS, below 16 GiB host available memory, or after 180 seconds.
All workloads used one operator and inter-op thread, OMP/MKL/OpenBLAS limits of
one, empty CUDA visibility and HF/Transformers offline mode. CUDA remained
uninitialized. Maximum measured child RSS was 821936128 bytes (784 MiB), and
minimum host available memory was 49588207616 bytes (46.2 GiB). No guard fired.
The pre-existing conftest's two-thread import setting was overridden before every
test workload by the runner plugin, as in the earlier bounded qualification.

Exact commands, wheel hashes, logs, resource receipts and JUnit results are
saved in `validation/core081-qualification/`; the top-level summary is
`validation/core081-qualification.json`. The initial six-check JUnit file was
superseded by the next run's default filename; its command, resource JSON and
complete passing console log remain. The ten-case and sixteen-case JUnit files
are retained separately. No result was reconstructed or claimed from an absent
JUnit file.

## Limits unchanged

Only random tiny Cosmos and synthetic teacher tensors were used. The owned
native loader is replaced by the existing tiny fixture in these tests; no
pretrained weights are loaded. Actual accelerator admission, native full-model
memory/sampling/decoder parity, convergence/quality and HNDL matched-network
parity remain unqualified. Native responses still use programmatic CPU tensors;
no JSON tensor-file input protocol is claimed. CPU/BF16 and render/reference
refusals remain intact. Historical cohorts, active pins and original PR payloads
are unchanged. No activation, publication, merge or parent-thread messaging.

A nonblocking core diagnostic rough spot was observed: a valid v2 envelope with
an invalid adapter tensor shape raises raw `ValueError: checkpoint tensor schema
differs`, whereas checksum/factory identity failures use `CheckpointError`.
The test asserts the current pinned behavior and proves complete cleanup and
rejection before any adapter mutation. No core patch, newer pin or exception
translation workaround was introduced in Anima.
