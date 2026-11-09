# Anima: ParticleGAN 0.9.5 source cohort

The current installer pins `particle-sliders-core==0.13.0` to public commit
`3dd672a7f9b8c169daef90fd63e4bb42b9ddc423` and the published `particlegan==0.9.5`.
Training explicitly uses `particle_sliders.pg095.ConditionalPG095` with
`release-0.9.5`. Native image/field geometry keeps the released `auto` penalty
domain. Learning-rate defaults, numerical profiles, model geometry and native
admission boundaries are unchanged.

Install in a **new environment** with `python -m pip install .` (or the existing
native extra where provided). The common CLI and Python factory share this
engine and source receipt. Installed recursive core source/data and official
ParticleGAN source checks remain enforced. This migration requires fresh caches,
SessionConfig/cohort identities and training runs; no old training packet is
rebound or rewritten. The core rejects PG0.9.4 checkpoint identities.

Previous PG0.9.4 and earlier native proofs, learning-rate searches, recommendations,
weights and source locks remain historical and require their original pinned
environments. They are **not PG0.9.5 calibration evidence**. CPU update/resume,
export and source-refusal checks establish software contracts only. Current
rate guidance uses [Candlelit C2 G=0.00075 at RAW960](docs/finished-output-learning-rates.md),
the best of three already scored arms on four examined DEV cases, with mean
intended-effect ON−OFF gain 0.0038009807467460632. It preserves the original
captured native source/model/conditioning identities. The [earlier G=1e-5
teacher-response screen](NATIVE-LR-PG095.md) and its held-out RAW SSE
1.0649726878176227 versus OFF=1 remain historical diagnostics. Neither record
changes common lifecycle CPU/render gates or package defaults.

Use one selector: intended-effect intensity ON minus matched OFF in finished
outputs, averaged equally across the declared cases. Reuse completed searches,
saved RAW checkpoints, outputs and first scores; do not replay completed updates
or add fresh confirmation for this existing choice. SSE, preservation and generic
beauty are diagnostics only, never selectors or tiebreaks. Use RAW weights with
`ema_decay=None`; do not enable weight EMA.
