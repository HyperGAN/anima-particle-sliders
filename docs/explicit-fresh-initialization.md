# Opt-in fresh initialization

`lumen_studio_fresh95` adds the explicit `anima-explicit-fresh95-v2` policy for **new cohorts**. Use it when you choose fresh scoped initialization; existing trainers, registered backends, saved cohorts, released files and [finished-output LR choices](finished-output-learning-rates.md) retain their original behavior.

The policy uses the actual existing adapter Parameters. Down matrices receive the public ParticleGAN QR initializer at Anima's original a=√5 scale, Uniform ±1/√fan_in. The actual 128×4 bank receives R2Normal(0,1). UP stays exactly zero, and every router/routed-MLP and critic parameter keeps its original constructor values. Initialization runs once before optimizer construction. Model weights and forward methods are unchanged.

## Installation and ownership

The namespace is part of the normal Anima package discovery (`lumen_studio*`), alongside `lumen_studio`. It adds no dependency, extra, CLI, common-backend registration, or core overlay. Install this source in an isolated product environment with the exact current [core0.13.0 / PG0.9.5 cohort](../PG095-ADOPTION.md). The original core commit/source guards remain active. The new helper calls the public PG0.9.5 initializer directly after checking its Source; it does not import a later core initialization helper.

The application supplies an already-loaded runtime, completed matching teacher cache, and its resource/admission owner. These APIs neither load a pretrained model nor collect teachers, schedule a job, or issue admission. For native accelerators, `native_admission` retains the original model/cache teacher-parity admission contract. The application also owns and serializes use of the borrowed runtime. See the existing [explicit native training interface](../PG09_TRAINING.md) for its runtime/cache boundary.

## Fresh training

Choose a new output directory and explicit cohort ID. `mode="fresh"` refuses a resume packet. The existing seeded bank→branches→critic constructors run first, followed by scoped initialization and then the unchanged shared game. The receipt is bound into the game signature before optimizers exist. FULL0, native RAW0 and the initialization receipt are durable before the entry returns.

```python
from lumen_studio_fresh95 import open_training

# runtime, cache and application_owner are supplied by the admitted application.
options = dict(seed=7, batch_size=2, total_steps=128, ema_decay=None,
               native_admission=application_owner.admit_native_update)
trainer = open_training(runtime, cache, "new-fresh95-run", variation,
    mode="fresh", cohort_id="my-new-fresh95-v2-cohort", **options)
try:
    # Request only genuinely missing work. Save FULL and RAW after each update.
    metrics = trainer.update()
    full_pin = trainer.save()
    raw_pin = trainer.export("new-fresh95-run/RAW-after-update.safetensors")
finally:
    trainer.close()
```

Optional `lr`, `d_lr_mult` and `prior_lr_mult` have the same meaning as the existing trainer: G=lr, D=lr×d_lr_mult and bank=lr×prior_lr_mult. Retain the same values on resume. Omitting them preserves the official preset; it does not select a measured fresh95 optimum. The existing Candlelit C2 RAW960 result belongs to its original cohort and is not transferred to this policy.

No weight EMA path is enabled. Entries reject every non-None `ema_decay`; updates, exports and inference use raw current weights. ParticleGAN's scalar optimizer statistics remain unchanged.

## Strict FULL resume

`save()` returns a `Checkpoint(path, bytes, sha256)`. Preserve that whole-file pin together with the FULL packet. Reopening requires the same cohort, options, frozen cache and Source, plus another new output directory. The initializer is skipped entirely, and the complete shared game—including bank/network/critic, both optimizers, streams, RNG, row orders/cursors/exposures and completed count—is strictly restored and compared before any update.

```python
resumed = open_training(runtime, cache, "new-resume-output", variation,
    mode="resume", cohort_id="my-new-fresh95-v2-cohort",
    resume=full_pin, **options)
try:
    # Starts at the saved cursor; no completed updates or teachers are replayed.
    metrics = resumed.update()
    next_full_pin = resumed.save()
    next_raw_pin = resumed.export("new-resume-output/RAW-after-update.safetensors")
finally:
    resumed.close()
```

Only this policy's versioned FULL envelope is accepted. Historical bare packets and saved cohorts keep their original loaders; there is no migration or reinitialization. This also avoids the historical trainer's implicit `pg09-resume.pt` discovery.

## Native RAW inference

Keep the native safetensors file with its `.pg09.json` and `.fresh95.json` sidecars. `open_inference` validates the policy provenance, then uses the original product `ParticleAdapter.load_export` format. It builds no critic, optimizer, game or teacher cache and never calls the initializer.

```python
from lumen_studio_fresh95 import open_inference

owner = open_inference(runtime, "unused-preview-adapter-name", raw=raw_pin,
    owner_admission=application_owner.admit_native_inference)
try:
    with owner.scaled(1.0) as native:
        # Existing TurboRuntime API; native execution needs application admission.
        image = native.render(prompt, seed=7, width=768, height=768, steps=10)
finally:
    owner.close()
```

Strength remains in [0,1]. The scale/observation context preserves the existing native scheduler, buffers and modes. Close removes only the owned attachment, preserves foreign adapters/hooks, restores checkpoint callbacks, and leaves the borrowed runtime alive. If cleanup fails, further updates/inference are blocked while `close()` can retry the unfinished cleanup actions.

## Verified scope

On 2026-11-10, all four new CPU software cases passed (4 tests, 0 failures/errors/skips) with the exact guarded core0.13.0 / PG0.9.5 environment, Torch2.9.1+cu128, Diffusers0.36.0 and Transformers4.57.1. They verified initialization-before-optimizers, deterministic declarations and same Parameters/zero-UP/critic law; exact FULL/optimizer/RNG restore; native RAW loading with initialization forbidden; refusal paths; and retryable cleanup. The test Source is retained verbatim in [test_explicit_fresh95.py](../tests/test_explicit_fresh95.py).

Those were random tiny Cosmos and synthetic-cache software checks. No pretrained model, real teacher, GPU, old proof or completed LR work was consumed or replayed. Native pretrained training/rendering with this new policy and the new package's installed-wheel build have not been exercised by those checks. Existing optional dependency metadata and native qualification limits remain authoritative; no new released weights, rate default or quality claim follows from adding this opt-in API.
