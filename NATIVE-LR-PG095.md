# Historical PG0.9.5 native512 teacher-response SSE measurement

Current use follows the [Candlelit C2 RAW960 finished-output choice](docs/finished-output-learning-rates.md).
The older screen below used a different batch, horizon and objective. Its
numbers, options preset and source receipts remain unchanged; they are
diagnostics, not the current LR selector or a request to repeat the search.

**Best-tested development-selected setting: G=1e-5, D=0.05399999999999999,
P=0.030000000000000002, with a declared 128-update horizon.** Independent
held-out RAW relative SSE was **1.0649726878176227**, compared with OFF=1.
This was a source/mechanics-qualified historical measurement. It establishes
**no held-out improvement over OFF** under that diagnostic.

This setting belongs to the historical cohort, not the current best-tested
finished-output recommendation. Defaults and existing released slider weights
are unchanged. The [machine-readable evidence](docs/pg095-native512-lr-evidence.json)
records exact source pins, actual rate bits, results and retained receipt hashes.

## Exact measured scope

The run used Anima 0.4.0 at
`fa0f7076b7c723d53080343f8adc3a99b64b744d`, core 0.13.0 at
`3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`, and official ParticleGAN 0.9.5
with `release-0.9.5`. It created fresh teacher values, initialization and state;
no PG0.9.4/earlier checkpoint or teacher result was relabeled.

- `circlestone-labs/Anima:turbo-v1.1`, Candlelit, native512 velocity fields
  shaped `[1,16,1,64,64]`; BF16 base/conditioning and FP32 responses/trainables.
- Actual rank-8 routed adapter, 128 by 4 particle bank, batch 2, seed 7,
  common horizon 128, live weights without EMA.
- Torch `2.13.0+cu126`, Transformers `5.15.0`, Diffusers `0.40.0.dev0`,
  NumPy `2.5.2`; `deterministic-math-cuda-v1`, TF32 off, math SDPA,
  deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Development: 4 equal examples, each containing all 10 positions, for 40 fields.
  Independent final: 3 previously unused subjects by 2 seeds by 2 frozen
  trajectories, giving 12 examples and 120 fields. All 12 had nonzero teacher
  energy; none were excluded.

The historical screen's metric promoted student, neutral and positive teacher operands to
FP64 **before subtraction**, pooled SSE and teacher-edit energy within each
example, then averaged examples equally. Teacher-only zero-energy exclusions
were frozen before adapter application; there was no epsilon denominator, and
nonfinite students could not remove examples. Current rate selection instead
uses one equal-case mean intended-effect ON−OFF score in finished outputs.
SSE, preservation and generic beauty cannot select, reject or break ties.

## Equal-budget search and held-out measurement

The initial four G values were `1e-5`, `3e-5`, `1e-4`, `3e-4` at 32 updates.
The first two development survivors continued to 64. Because G=1e-5 was the
lower boundary, `3e-6` and `1e-6` were prospectively added, each completing
32 and then 64 updates with the exact same actual D/P, initialization,
teachers and horizon. The ultimate comparison used all four measured 64-step
endpoints; completed original cells were not retrained.

| G | RAW at 32 | RAW at 64 |
| --- | ---: | ---: |
| 1e-6 | 0.975247842493776 | 0.9583045106250359 |
| 3e-6 | 0.9356771346281453 | 0.868846797736529 |
| **1e-5** | **0.8172587901035525** | **0.7339753759110534** |
| 3e-5 | 0.8993422321566498 | not selected |
| 1e-4 | 0.8370187316394356 | 0.79757146744406 |
| 3e-4 | 1.584595754303189 | not selected |

Every admitted finite stage completed its declared budget, including scores
above OFF. The selected G=1e-5 continued from its exact 64-step packet to 128
without prefix retraining. Its development RAW at 128 was
**0.7648770712839992**. Selection was frozen before final conditioning and
teacher/student evaluation. The independent final score
**1.0649726878176227** is about **6.4973% above OFF**, and no held-out improvement
is claimed. The opened final data did not drive a retune or another selection.

Actual new-source checks covered full2 and fresh1+1 state/metric/response
identity, exact full-state continuation, public RAW tensor mapping/replay,
OFF/base/caller-RNG identity, actual optimizer groups and finite updates.
An independent CPU audit recomputed all 120 saved final fields in FP64 and
reproduced the reported score exactly. Final evaluation performed zero updates.
Source/observer failures and the resource interruption remain in the evidence;
none were replaced with a successful native claim.

## Historical options and API

[The options overlay](configs/anima/pg095-native512-best-tested.options.json)
contains the historical common-session option keys; it is not a complete
runnable session or the current rate choice. Its original reproduction scope
requires its own verified model/cache/cohort configuration.
It sets `steps=128`, `batch=2`, `seed=7`, and disables EMA. Its LR multipliers
resolve to these exact binary64 products:

```text
G = 1e-5                         (0x1.4f8b588e368f1p-17)
D = G * 5399.999999999999        (0x1.ba5e353f7ced8p-5)
P = G * 3000.0                  (0x1.eb851eb851eb9p-6)
```

The measured accelerator route was the explicit `PG09Trainer` API with an
external native-admission callback and activated deterministic numerical profile:

```python
trainer = PG09Trainer(
    admitted_runtime, verified_train_cache, new_run_directory, "candlelit",
    total_steps=128, batch_size=2, seed=7, critic_width=32,
    response_scale=1.0, sigma=1.0,
    lr=1e-5, d_lr_mult=5399.999999999999, prior_lr_mult=3000.0,
    numerical_profile="deterministic-math-cuda-v1",
    native_admission=coordinator_admission_check,
)
```

Activate that profile before CUDA startup, and use the exact fresh cache and
runtime/source checks described in [the API guide](PG09_TRAINING.md).
Common lifecycle CPU/render capability gates remain in place; this measurement
does not enable a different native lifecycle or rendering operation. A changed
rate, data set, horizon, model or numerical profile needs a new cohort. Historical
PG0.9.4/earlier evidence keeps its original source and environments.
