# Fresh full-coverage PG0.9.5 repair

The measured native512 run used a pilot's first two prompt rows and one seed per
row: 40 training fields and 40 development fields. The full declared Candlelit
manifests contain 7,680 training fields and 480 development fields. This repair
creates a separate fresh cohort. The existing measured recipe, opened final,
checkpoints, source snapshots and historical pilot scripts remain immutable.

`scripts/anima_pg095_coverage_plan.py` defaults to production coverage. It derives
every row, seed, trajectory and denoise position from validated unsealed
manifests. It has no first-N or shard-count option. Pilot mode requires explicit
row IDs and per-split seeds and always sets `production_rate_eligible=false`.
Production rate consumers must call `require_production_rate_coverage` with the
authentic declared manifests and actual source-bound index/field metadata; a
pilot or even a rehashed pilot labeled production refuses that gate.

The full development set repeats each character in multiple prompt pairs.
Examples therefore use `[row_id, seed, trajectory]`, with character retained as
a separate stratum. Every example has exactly ten distinct native positions.
The metric remains operand-first FP64 SSE against the positive teacher, pooled
over coordinates and positions inside each example, then averaged equally over
examples. Teacher-only exact-zero membership is fixed before student scoring;
all student fields must be finite, including excluded zero-energy examples.

The staged preparation order is:

1. Compile complete train/dev coverage and authenticate declared manifests.
2. In a separately admitted CPU phase, compile all neutral/positive conditions
   through one frozen native CPU-BF16 encoder law. Unload the encoder. The same
   compiler and identity law must also serve a prospectively unused fresh final.
3. In a separately admitted GPU phase, call
   `fullcoverage_prepare.prepare_fullcoverage` with an already loaded pristine
   runtime, immutable conditioning provider and actual source/owner admission.
   It generates fresh native512 ten-step neutral and positive trajectories from
   each declared seed and evaluates both pristine teachers at every state.
   Historical teachers, embeddings, latent paths and cached neutral trajectories
   are not reused. A partial producer output has no completion/coverage authority.
4. Fit normalization from the complete authenticated training cache only. The
   separate source-qualified transform uses `(S-N)/S_t` and `(T-N)/S_t`, with
   train-only per-coordinate/position scales. Its signature must bind full game
   config/state and refuse incompatible resume. Native RAW scoring, export and
   inference stay in the original native field units.
5. Train a fresh official core0.13.0 / ParticleGAN0.9.5 cohort. The prospective
   matched budget is 1,600 updates and effective batch8, subject to separately
   reviewed native execution and actual resource fit. It cannot resume the old
   128-step horizon or silently call a batch2 run equivalent. No actual training
   worker/driver or resource admission is authorized by these metadata helpers.
   The core may retain all eight generator graphs in one update. A genuine
   native full2 and saved1-to2 continuation must qualify this new batch/transform
   cohort's actual resource fit and exact state before the grid. If it does not
   fit, report the actual refusal and predeclare a separate alternative cohort;
   do not silently add microbatching, shrink geometry or change precision.
6. Select using full unsealed development coverage. Freeze the real selected
   endpoint before a single observation of a newly declared unused final.
   The old opened final must never choose rates, scales, architecture or budget.

Weight EMA is disabled throughout: `ema_decay=None`, no materialized shadow, RAW
training observation/export/inference only. The official optimizer's scalar EMA
statistics are unchanged. This source proposal does not restore historical
weight EMA, historical Adam/critic math or gallery strength calibration.

The native producer library is source preparation, not a hardware admission
controller. A future one-phase driver must bind the exact new product/core/
conditioning/normalization sources and actual owner receipts, freshly account
known live whole-phase growth once, retain all resource/lifetime guards and own
its actual child through unload/reap. No future capacity reservation, native
qualification, convergence or final improvement follows from metadata coverage.
