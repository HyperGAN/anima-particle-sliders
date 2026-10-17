# Explicit ParticleGAN 0.9 training

`lumen_studio.pg09_training.PG09Trainer` is the new training entrypoint. It
executes `particle_sliders.pg09.ConditionalPG09` with the public
`bispf_optimized` recipe and `particle_sliders.pg09_backends.AnimaRuntimeResponseAdapter`.
The product contains no PG0.9 loss, optimizer, constraint or sampler copy.
ParticleGAN must match the exact version and verified source profile required
by the reviewed installed shared package. The package owner is qualifying
PG0.9.1; the existing admitted lineage uses PG0.9.0. This proposal does not
retarget an admitted runtime and does not support PG0.10.

The shared package owner must publish the reviewed 0.6.0 integration containing
`particle_sliders.pg09_backends`; the prior 0.3.0 dependency does not provide
this API. The immutable dependency pin is updated only to that reviewed
integration commit. This remains an integration gate until publication.

Runtime loading, GPU admission and scheduling belong to the external
coordinator. After that coordinator has admitted and loaded the frozen
`TurboRuntime`, it may call:

```python
from lumen_studio.cache import TargetCache
from lumen_studio.pg09_training import PG09Trainer

cache = TargetCache(verified_train_cache, verify=True)
trainer = PG09Trainer(admitted_runtime, cache, new_run_directory,
                      "finalform", total_steps=100, batch_size=2,
                      native_admission=coordinator_admission_check)
try:
    trainer.update()
    trainer.save()                  # exact shared-core resume packet
    trainer.export(new_export)      # immutable native live-weight export
finally:
    trainer.close()
```

This API does not create a runtime, submit work or infer GPU availability.
For non-CPU runtimes, a mandatory coordinator callback receives the exact
model identity, cache fingerprint and native response shape before the first
update. It must return the literal `True` after the coordinator verifies the
actual staged weights, numerical frozen-teacher/native parity, source pins
and memory/admission constraints. Its boolean decision is recorded; the
callback is an admission seam and does not perform or replace those numerical
checks. CPU software fixtures do not require admission.
Use a separate run directory. It automatically restores `pg09-resume.pt`
when present and refuses changed conditioning, targets, model identity,
recipe, budget, architecture or implementation sources through the shared
core's checkpoint contract. An inference export is distinct from this full
training packet. The exported weights are explicitly `live`; no EMA is
computed or claimed by this binding.

The existing `ParticleAdapter` and its actual 128 by 4 routed bank are used
in hooked model forwards and passed to the shared game. Frozen base model
parameters are excluded from optimizers. Target records retain genuine
`[1,C,1,H,W]` velocity fields. The penalty receives that native geometry;
only the critic extracts spatial patches internally using rank-two Linear
weights supported by PG0.9. Critic logits are unbounded. Training optimizes
the PG adversarial objective and its required public constraints, with no
MSE or reconstruction objective.

Training is unipolar: the adapted neutral conditioning at scale +1 learns
the frozen positive response at the same latent and raw scheduler time.
Conditions contain mean neutral token embeddings, normalized native time,
denoise position, frozen trajectory choice and +1. Row IDs, prompt labels,
character IDs and seeds are excluded. Conditioning is read lazily from the
verified training cache; no development cache is used for training. Native
exports load through the existing `adapter_class`/`ParticleAdapter` paths
and retain CFG1 Turbo rendering with the original scheduler, VAE and
selected weights.

## Existing entrypoints

| Entry | Executed implementation | Status |
| --- | --- | --- |
| `lumen_studio.pg09_training.PG09Trainer` | installed `ConditionalPG09`, native product adapter | new PG0.9 path |
| `release_tools/train.py`, Studio `Coordinator.train_quantum`, `python -m lumen_studio train`, `scripts/anima_campaign.py` | archived `Trainer` and handwritten stamp `Game` | historical gmix reproduction; not PG0.9 |
| `benchmark.py`, `preflight.py`, `convergence.py`, `scripts/anima_verify_update.py` | archived game/reference probes | historical diagnostics; not PG0.9 |
| `release_tools/distill.py`, `prepare_expansion.py` | shared closed-form `fit_routed_down` | ordinary LoRA approximation; not slider GAN training |
| Studio render, `release_tools/infer.py`, Comfy particle node | selected native/alpha/single particle or native LoRA file | inference only |

The following inventory includes all production Python command wrappers and
the callable entrypoints they reach at the audited product base `36ad1267`:

| Files or routes | Training/inference behavior and disposition |
| --- | --- |
| `lumen_studio/__main__.py` `train`, `worker`; `api.py` `/api/training/runs`; `coordinator.py` `train_quantum` | queue/execute archived `training.Trainer` and `game.Game`; legacy gmix only |
| `scripts/anima_campaign.py` | target preparation and archived coordinator training; legacy gmix only |
| `release_tools/train.py` | archived release recipe, explicit Final Form/Afterimage/Dusk manifest dispatch; legacy gmix only |
| `lumen_studio/benchmark.py`, `preflight.py` | archived Trainer update/resume/performance probes; legacy gmix only |
| `lumen_studio/convergence.py` `response_step`; `scripts/anima_verify_update.py` | historical separate D/G/VIC/reference update diagnostics; no PG0.9 claim |
| `scripts/anima_verify_reference.py`, `anima_verify_normalization.py`, `anima_probe_training.py` | archived source, normalization and selected-EMA residual diagnostics; no new optimizer path |
| `lumen_studio/__main__.py` `serve`, `references`, `cache`, `compile`, `prepare-model`; `runtime.py` `create_runtime`; `cache.py` `prepare_targets` | API serving, queueing, frozen target preparation or official model conversion; no GAN update |
| `api.py` `/api/generate`, `/api/comparisons`, image rerender; `coordinator.py` `render_one` | selected stored original native EMA snapshot plus exact requested mix/prompt/seed; `TurboRuntime.render` |
| `release_tools/infer.py` | explicit selected native/alpha/single particle or native LoRA file; `TurboRuntime.render` |
| `release_tools/distill.py`, `prepare_expansion.py` | selected EMA teacher activations, shared closed-form LoRA approximation, explicit particle/LoRA render sweeps; not slider GAN training |
| `release_tools/audit_distill.py`, `match_bad_intent.py`, `calibrated_release.py` | selected native particle/LoRA comparison renders; no GAN update |
| `comfy_particle.py`, `single_particle.py`, `alpha_adapter.py`, `particle_io.py` | Comfy/native checkpoint loading, format dispatch, alpha application and routed forward hooks; no training objective |
| `release_tools/queue_samples.py`, `refresh_examples.py` | enqueue/render selected checkpoint examples; no GAN update |
| `release_tools/prepare_model.py`; `scripts/anima_reuse_targets.py` | model conversion or byte-verified frozen cache reuse; no GAN update |
| `scripts/anima_cuda_host.py` | kernel-policy host wrapper for the supplied script/module; does not change its training objective; requires separate native admission |
| `scripts/anima_submit_audit_review.py`; `release_tools/verify_live.py`, `verify_comfy.py`, `verify_distillation.py`, `verify_samples.py` | review/source/math/sample checks; no PG0.9 trainer dispatch |
| `release_tools/build.py`, `stage.py`, `stage_expansion.py`, `add_bad_intent.py`, `rescale_lora_alpha.py`, `publish.py` | release assembly, alpha rescaling or publishing commands never invoked; only the ZIP packaging helper runs in a CPU regression |

`configs/anima/model.lock.json` and `converted-model.lock.json` pin the actual
runtime to `circlestone-labs/Anima:turbo-v1.1`, repository revision
`f973fc41ec7545364ac9776c2440285f43ff2a30`, with official
`anima-turbo-v1.1.safetensors`, `qwen_3_06b_base.safetensors` and
`qwen_image_vae.safetensors`. The Base-v1.0 model identifier in that lock is
the tokenizer source, not the active diffusion weights. `TurboRuntime`
rejects a different model label, verifies staged file hashes and loads local
components only. It uses frozen Qwen/conditioner/VAE and the selected routed
DiT adapter, native scheduler times divided by the training timestep count,
CPU-seeded noise, 8/10/12 Euler steps and CFG1.

`data/expansion-selection.json` chooses Final Form EMA step 1200, particle
alpha 12 and ordinary-LoRA alpha 24. `prepare_expansion.py` reads that exact
EMA for export rather than choosing the latest 1600 snapshot. No selected
weights or alpha values were changed by this audit.

Outside this product, `conceptmod/textsliders/train_lora_anima.py` and the
Anima smoke/embedding scripts in the adapter repository run legacy Base
UNI/trajectory/MSE experiments, not this Turbo PG0.9 path. The external
`/mnt/nvme/sliders/scripts/train_verified_anima_concept{,_duty_cycle}.py`
names describe Anima-derived prompts, but they execute MOSS
`moss_sliders.training.train_step`; their paired monitor scripts launch those
MOSS pilots. They were read only and are not Anima runtime evidence.

Historical Game helpers remain available only with `legacy_cpu_fixture=True`
on a CPU runtime. Default execution through the old Trainer fails before
Trainer allocation or any historical optimizer update, and CUDA is always refused even with that fixture flag, so
CLI, release and coordinator wrappers cannot optimize the historical gmix
objective on a live runtime. Previously trained caches and checkpoints are
read-only inputs; they are not resumed under the new PG0.9 objective. Old CLI wrappers may load a model or prepare caches before reaching Trainer;
this guard is not an admission or GPU dispatch mechanism. New coordinator
routing must select this explicit callable after admission; it is not enabled in a resident service.

## CPU qualification and remaining gates

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/particle-sliders/packages/particle-sliders-core/src:. \
/path/to/pg090/bin/python -m pytest tests/lumen_studio/test_pg09_training.py \
  -q -o pythonpath='' -p no:cacheprovider
```

A supplemental CPU API fixture passed eleven tests with immutable core `1e960d9b`
(PG0.9.0) plus the sole owner's packaged backend definitions. That process-only
composition does not qualify the release wheel. Final release-target tests
remain blocked while the owner investigates PG0.9.1; the qualified test
environment currently contains PG0.9.0. No installed environment was changed.

The tests use a real Cosmos transformer at tiny random dimensions, the
product runtime methods and the actual routed bank. They verify shared
PG0.9 execution, nonzero response gradient to the same bank, bank/branch
updates, frozen base weights, native geometry, condition features, exact
resume, cache rejection and portable native live-weight reload. No
pretrained model is loaded and CUDA remains uninitialized.

Native image quality, full-model memory, BF16 CUDA numerical parity,
admitted checkpoint recomputation and image/export parity require the
coordinator's separate native-runtime gate. Published adapters and archived
example selection are not changed or republished by this work.
