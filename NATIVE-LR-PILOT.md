# Native PG094 learning-rate pilot

`scripts/anima_pg094_lr_pilot.py` runs the current real, frozen Turbo v1.1
runtime and actual routed particle bank with official PG094 loss, BISPF,
SBNP and DualNorm. It accepts explicit base rates and holds the public critic
and prior multipliers at 1.5 and 2.5. New configuration options `lr`,
`d_lr_mult`, and `prior_lr_mult` are identical in `PG09Trainer` and the common
session JSON/API/CLI. An omitted override keeps ParticleGAN's resolved default.

The pilot requires an independently admitted GPU role slot, local converted
model, historical native target inputs, a new output directory and a hard
process memory budget. It never changes historical cache, checkpoint, model,
environment lock, or qualification receipt bytes. It verifies the original
native environment and model-file lock, then freshly computes both teacher
responses from selected original latents and prompt embeddings into a new
content-bound cohort. A repeated neutral teacher prediction must match exactly
before accelerator updates. It verifies that training and heldout characters
are disjoint; heldout fields never update weights or normalization.

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -u scripts/anima_pg094_lr_pilot.py \
  --model-dir /local/anima-turbo-v1.1 \
  --historical-targets /local/original-targets \
  --output /local/new-experiments/anima-pg094-lr \
  --lrs .00012 .0012 .004 .012 --steps 32 --shards 2 --eval-rows 12 \
  --max-memory-gib 18 --render
```

The shared game owns every update and exact optimizer/RNG checkpoint. The
pilot saves and reconstructs the trainer midway, exports independently loaded
EMA safetensors, compares heldout response scores after reload, optionally
renders a heldout 512px/10-step sample, and records peak allocated memory,
step times, losses and relative parameter update size. Nonfinite values,
backbone writes, teacher mismatch, changed cohort and exceeded memory budget
are errors. The output `report.json` is provisional screening evidence; the
small character subset, one seed and short update budget do not establish a
model-wide recommended LR or qualify common native sampler/render interfaces.

The base-rate grid is a coupled G/D/P screen. `--fixed-d-lr` and
`--fixed-prior-lr` hold those absolute rates while screening generator rates;
their multipliers are calculated as absolute/base. Each new confirmation
attempt captures an immutable actual step-zero packet and network, critic,
RNG, configuration and recipe hashes before updates, plus actual optimizer
group learning rates. Run from a fixed, read-only complete source snapshot.
`scripts/anima_pg094_lr_render.py` provides a separate portable-artifact-only
matched native rendering window, after training unloads. It uses heldout cached
prompt embeddings, loads only the transformer, scheduler and decoder, and
creates neutral, positive-teacher and half/full-strength EMA views with matched
seed/sampler and image hashes. No critic or optimizer is constructed.

The common lifecycle's prior CPU-only gate remains separate from this explicit,
fresh-teacher native experiment admission. Historical cohorts retain their
original source and optimizer identities. No published weights are replaced.
