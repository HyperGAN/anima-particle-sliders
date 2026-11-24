# Anima Final Boss leaderboard

## Summary

**Current research default.** Soft foreground weighting with a 0.45 background floor, trained to RAW192 and replayed at early α 0.95. It scored **33.71** on eight development cases and **28.55** on eight separate confirmation cases. [Default and exact gains](final-boss-raw192/default.json) · [Matched confirmation images](final-boss-raw192/confirmed-default.jpg) · [Download RAW checkpoints](https://huggingface.co/ntc-ai/anima-final-boss-research)

**What worked in training.** Soft foreground weighting concentrates the teacher edit on the character while retaining some signal in the background. It leads the matched early-strength comparison at **25.66**. A 35% blend of stronger Boss cues scored **24.93** at early strength and led at constant strength with **16.90**. Paired contexts and projection sometimes improved readability at the cost of a weaker character change. [Training comparison](#training-techniques) · [Target-generation source](final-boss-raw192/README.md#source-for-the-trained-targets)

**What earns a high score.** The slider should make the subject visibly read as a Final Boss while keeping the image coherent. An edit with zero Boss gain scores zero. Visible damage lowers the score even when the Boss effect is strong. [Scoring and comparison protocol](#comparing-a-new-run) · [Jev rubrics](final-boss-raw192/rubrics.json)

**Latest challenger.** Applying stronger cues to the final three teacher positions led the four-case update-144 preview, **8.92** to **7.75** under the leaderboard rubric. The eight-case RAW192 comparison is pending. [Matched update-144 results](#update-144-preview)

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

Stronger Boss distinction accounts for **3.52** of the cue blend’s constant-strength gain. At early strength, paired contexts and whole-image projection improved readability, while weaker Boss distinction cost them **4.61** and **8.02** points respectively. These contributions compare each variant to soft foreground weighting. For each matched case, the calculation multiplies each factor’s change by the other factor’s average across the two renders, then averages across cases. [Per-case Boss and readability scores](final-boss-raw192/results.json)

[Shared early samples](final-boss-raw192/shared-early.jpg) · [Shared constant samples](final-boss-raw192/shared-constant.jpg) · [Training and replay source](final-boss-raw192/README.md#source-for-the-trained-targets)

## Update-48 preview

The late-cue and two-cue-average runs each passed all four image checks. Mean Jev Boss gain at shared early inference strength was **+0.186** for soft foreground weighting, **+0.176** for late cues, and **+0.174** for the cue average. Both challengers gained on bookkeeper, botanist, and seamstress. Both scored lower on archivist, where soft foreground weighting gave the coat a sharper silhouette. [Four matched cases across all three recipes](final-boss-raw192/ablations/cue-tests-raw48-matched.jpg)

## Update-96 preview

The resumed late-cue run passed all four image checks. Mean Jev Boss gain reached **+0.353**, compared with **+0.373** for soft foreground weighting at the same update. The late-cue archivist gained **+0.427** versus **+0.260** for soft foreground weighting. Soft foreground weighting led on the other three cases. [Four matched originals and renders](final-boss-raw192/ablations/late-cues-raw96-matched.jpg)

Each cue-timing recipe replaces 36 of the 120 teacher fields. The replacement moves the edit by an average of **22.0%** of its original L2 norm at the first three positions and **16.8%** at the last three. The source and replacement fields have matched L2 norms. [Teacher field distances by position](final-boss-raw192/ablations/cue-field-distances.json)

## Update-144 preview

Late cues passed all four structural checks. The quarter-check question gave late cues **+0.453** mean gain and soft foreground weighting **+0.401** at the same early strength. [Four matched originals and renders](final-boss-raw192/ablations/late-cues-raw144-matched.jpg)

Scoring those same renders with the leaderboard’s separate character and damage questions gave late cues **8.92** and soft foreground weighting **7.75**. Late cues had higher mean Boss distinction gain (**+0.183** versus **+0.168**) and slightly lower mean image damage (**0.287** versus **0.302**). Bookkeeper accounted for most of the score increase. Botanist and seamstress favored soft foreground weighting after the damage adjustment. [Per-case scores and matched image hashes](final-boss-raw192/ablations/late-cues-raw144-scores.json)

## Qualified teacher targets

On eight archived teacher cases, guiding toward a concrete costume more strongly early in denoising scored **29.50–32.38**, versus **26.70** for the same costume prompt at fixed teacher strength. The range covers every possible outcome for one missing damage judgment. These are frozen Anima renders made before adapter training. [Per-case scores and gain schedule](final-boss-raw192/ablations/vibe-teacher-readability.json)

A prepared training recipe pairs all **80** fields from four qualified costume teachers with **40** fields from two original soft-foreground contexts. The original and new recipes each use 120 fields across six contexts. The new recipe changes both the contexts and the teacher guidance. Its planned fit uses 192 stock ParticleGAN updates and the same development comparisons. The timing test draws on observations that text conditioning has different effects across denoising stages. [eDiff-I](https://arxiv.org/abs/2211.01324) · [Prompt-Conditioned Interventions](https://arxiv.org/abs/2512.08486)

## Structural stops

The 35% cue blend that kept only components with matching signs across its two alternate cue edits stopped at update 48. The bookkeeper render collapsed to a texture and scored 0.83 on Jev’s unreadable-image check. Training stopped before update 49. [Saved RAW48 render](final-boss-raw192/ablations/diverse-consensus35-raw48-bookkeeper.png) · [Consensus research](https://github.com/HyperGAN/particle-sliders/pull/315)

## Comparing a new run

Compare the saved RAW checkpoint with the incumbent on the same prompts, seeds, sampler, and shared strengths. At 192 updates the early score to beat is **25.66**. The 35% cue blend sets the constant-strength comparison at **16.90**. Report tuned strength and its confirmation panel alongside the shared-strength comparison.

Jev measures Boss character distinction and visible image damage separately. For each case the score is `max(0, Boss_ON − Boss_OFF) × (1 − image_damage)²`. The leaderboard averages the per-case scores on a 0–100 scale. Catastrophic structural failures and unmeasured rows carry separate statuses. [Jev rubrics](final-boss-raw192/rubrics.json)

The Particle Sliders CPU study ranked paired contexts first at early strength and background-only projection first at constant strength. Both winners differ from the native Anima rankings above. Use matched native renders to rank changes to the training recipe. [Particle Sliders PR #321](https://github.com/HyperGAN/particle-sliders/pull/321)
