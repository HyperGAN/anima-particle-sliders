# Anima Final Boss leaderboard

**Current research default.** Soft foreground weighting with a 0.45 background floor, trained to RAW192 and replayed at early α 0.95. It scored **33.71** on the eight development cases and **28.55** on eight separate confirmation cases. [Default and exact gains](final-boss-raw192/default.json) · [Matched confirmation images](final-boss-raw192/confirmed-default.jpg) · [Download RAW checkpoints](https://huggingface.co/ntc-ai/anima-final-boss-research)

## Training techniques

The ranking uses the best measured variant of each technique after 192 updates with stock ParticleGAN. All rows share the same eight development cases and the early α 2/3 setting. The constant α 0.6375 column shows how the same checkpoints perform with a fixed gain through denoising.

| Rank | Training change | Early α 2/3 | Constant α 0.6375 | Research source |
| ---: | --- | ---: | ---: | --- |
| 1 | Soft foreground weighting, 0.45 background floor | **25.66** | 13.06 | [Anima PR #31](https://github.com/HyperGAN/anima-particle-sliders/pull/31) |
| 2 | Soft foreground weighting plus 35% stronger Boss cues | 24.93 | **16.90** | [Anima PR #31](https://github.com/HyperGAN/anima-particle-sliders/pull/31), related [teacher mixing #278](https://github.com/HyperGAN/particle-sliders/pull/278) |
| 3 | Paired scene and lighting contexts, 50% blend, 0.65 floor | 22.23 | 11.58 | [Concept Sliders](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05660.pdf), related [contrast isolation #216](https://github.com/HyperGAN/particle-sliders/pull/216) |
| 4 | Background-only scene and lighting projection, 0.45 floor | 22.22 | 11.51 | Local Anima adaptation of [nuisance deflation #254](https://github.com/HyperGAN/particle-sliders/pull/254) |
| 5 | Whole-image scene and lighting projection, 0.65 floor | 18.74 | 13.72 | Local Anima adaptation of [nuisance deflation #254](https://github.com/HyperGAN/particle-sliders/pull/254) |
| 6 | Train on adapter rollout states, 0.45 floor | 11.61 | 5.49 | Local Anima experiment, related [rollout protocol #246](https://github.com/HyperGAN/particle-sliders/pull/246) |

The prompt-only control scored **18.32** at shared early strength. Its constant result is unmeasured. The 35% cue blend gained **3.84 points** at constant strength over the default training recipe. It improved seven of eight matched cases at constant strength and one of eight at early strength. The 85% cue blend scored lower at both shared strengths. Each technique used one initialization seed. [All runs and per-case scores](final-boss-raw192/results.json)

[Shared early samples](final-boss-raw192/shared-early.jpg) · [Shared constant samples](final-boss-raw192/shared-constant.jpg) · [Training and replay source](final-boss-raw192/README.md#source-for-the-trained-targets)

## Comparing a new run

Compare the saved RAW checkpoint with the incumbent on the same prompts, seeds, sampler, and shared strengths. At 192 updates the early score to beat is **25.66**. The 35% cue blend sets the constant-strength comparison at **16.90**. Report tuned strength and its confirmation panel alongside the shared-strength comparison.

Jev measures Boss character distinction and visible image damage separately. For each case the score is `max(0, Boss_ON − Boss_OFF) × (1 − image_damage)²`. The leaderboard averages the per-case scores on a 0–100 scale. Catastrophic structural failures and unmeasured rows carry separate statuses. [Jev rubrics](final-boss-raw192/rubrics.json)

The Particle Sliders CPU study ranked paired contexts first at early strength and background-only projection first at constant strength. Both winners differ from the native Anima rankings above. Use matched native renders to rank changes to the training recipe. [Particle Sliders PR #321](https://github.com/HyperGAN/particle-sliders/pull/321)
