# Anima Final Boss leaderboard

## Summary

**Current research default.** Soft foreground weighting with a 0.45 background floor, trained to RAW192 and replayed at early α 0.95. It scored **33.71** on eight development cases and **28.55** on eight separate confirmation cases. [Default and exact gains](final-boss-raw192/default.json) · [Matched confirmation images](final-boss-raw192/confirmed-default.jpg) · [Download RAW checkpoints](https://huggingface.co/ntc-ai/anima-final-boss-research)

**What worked in training.** Soft foreground weighting concentrates the teacher edit on the character while retaining some signal in the background. It leads the matched early-strength comparison at **25.66**. A 35% blend of stronger Boss cues scored **24.93** at early strength and led at constant strength with **16.90**. Paired contexts and projection sometimes improved readability at the cost of a weaker character change. [Training comparison](#training-techniques) · [Target-generation source](final-boss-raw192/README.md#source-for-the-trained-targets)

**What earns a high score.** The slider should make the subject visibly read as a Final Boss while keeping the image coherent. An edit with zero Boss gain scores zero. Visible damage lowers the score even when the Boss effect is strong. [Scoring and comparison protocol](#comparing-a-new-run) · [Jev rubrics](final-boss-raw192/rubrics.json)

**Latest challengers.** The two-cue average led the update-96 preview and is training toward RAW192. Early and late cue timing finished below the shared early leader. The next run uses four qualified costume teachers and two soft-foreground contexts. [Update-144 results](#update-144-preview) · [Completed early-cue comparison](#early-cues-at-update-192) · [Qualified teacher targets](#qualified-teacher-targets)

## Training techniques

The ranking uses the best measured variant of each technique after 192 updates with stock ParticleGAN. All rows share the same eight development cases and the early α 2/3 setting. The constant α 0.6375 column shows how the same checkpoints perform with a fixed gain through denoising.

| Rank | Training change | Early α 2/3 | Constant α 0.6375 | Research source |
| ---: | --- | ---: | ---: | --- |
| 1 | Soft foreground weighting, 0.45 background floor | **25.66** | 13.06 | [Anima PR #31](https://github.com/HyperGAN/anima-particle-sliders/pull/31) |
| 2 | Soft foreground weighting plus 35% stronger Boss cues | 24.93 | **16.90** | [Anima PR #31](https://github.com/HyperGAN/anima-particle-sliders/pull/31), related [teacher mixing #278](https://github.com/HyperGAN/particle-sliders/pull/278) |
| 3 | Paired scene and lighting contexts, 50% blend, 0.65 floor | 22.23 | 11.58 | [Concept Sliders](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05660.pdf), related [contrast isolation #216](https://github.com/HyperGAN/particle-sliders/pull/216) |
| 4 | Background-only scene and lighting projection, 0.45 floor | 22.22 | 11.51 | Local Anima adaptation of [nuisance deflation #254](https://github.com/HyperGAN/particle-sliders/pull/254) |
| 5 | Stronger cues at the final three teacher positions | 21.20 | 10.41 | [Teacher timing adaptation and matched renders](#late-cues-at-update-192) |
| 6 | Stronger cues at the first three teacher positions | 20.32 | 14.99 | [Teacher timing adaptation and matched renders](#early-cues-at-update-192) |
| 7 | Whole-image scene and lighting projection, 0.65 floor | 18.74 | 13.72 | Local Anima adaptation of [nuisance deflation #254](https://github.com/HyperGAN/particle-sliders/pull/254) |
| 8 | Train on adapter rollout states, 0.45 floor | 11.61 | 5.49 | Local Anima experiment, related [rollout protocol #246](https://github.com/HyperGAN/particle-sliders/pull/246) |

The prompt-only control scored **18.32** at shared early strength. Its constant result is unmeasured. The 35% cue blend gained **3.84 points** at constant strength over the default training recipe. It improved seven of eight matched cases at constant strength and one of eight at early strength. The 85% cue blend scored lower at both shared strengths. Each technique used one initialization seed. [All runs and per-case scores](final-boss-raw192/results.json)

Stronger Boss distinction accounts for **3.52** of the cue blend’s constant-strength gain. At early strength, paired contexts and whole-image projection improved readability, while weaker Boss distinction cost them **4.61** and **8.02** points respectively. These contributions compare each variant to soft foreground weighting. For each matched case, the calculation multiplies each factor’s change by the other factor’s average across the two renders, then averages across cases. [Per-case Boss and readability scores](final-boss-raw192/results.json)

[Shared early samples](final-boss-raw192/shared-early.jpg) · [Shared constant samples](final-boss-raw192/shared-constant.jpg) · [Training and replay source](final-boss-raw192/README.md#source-for-the-trained-targets)

## Update-48 preview

All four recipes passed the structural checks. Stronger cues at the first three teacher positions led under the leaderboard rubric with **3.23** at shared early α 2/3. Soft foreground weighting scored **2.34**, the two-cue average **1.94**, and late cues **1.74**. Early cues gained more Boss distinction and also had the highest mean image damage (**0.305**). The archivist and botanist renders show visible speckling. [Four matched cases across all recipes](final-boss-raw192/ablations/cue-timing-raw48-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/cue-timing-raw48-scores.json)

Mean gain under the quarter-check question was **+0.228** for early cues, **+0.186** for soft foreground weighting, **+0.176** for late cues, and **+0.174** for the two-cue average.

## Update-96 preview

The two-cue average blends 35% of the mean of two alternate teacher edit directions into the original edit and keeps the 0.45 background floor and original per-field L2 norm. It led every case and scored **10.34** at shared early α 2/3. Early cues scored **6.89**, late cues **6.76**, and soft foreground weighting **5.73**. All four recipes passed the structural checks. Two-cue average had the lowest mean damage (**0.268**), though some edits visibly change the face, hair, or pose. [Matched originals and renders](final-boss-raw192/ablations/cue-timing-raw96-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/cue-timing-raw96-scores.json) · [Training source and commands](final-boss-raw192/README.md#source-for-the-trained-targets)

The quarter-check question gave mean Jev gains of **+0.526** for the two-cue average, **+0.408** for early cues, **+0.373** for soft foreground weighting, and **+0.353** for late cues. The full training comparison uses the eight-case RAW192 panel.

Each cue-timing recipe replaces 36 of the 120 teacher fields. The replacement moves the edit by an average of **22.0%** of its original L2 norm at the first three positions and **16.8%** at the last three. The source and replacement fields have matched L2 norms. [Teacher field distances by position](final-boss-raw192/ablations/cue-field-distances.json)

## Update-144 preview

At shared early α 2/3, early cues scored **9.78**, the two-cue average **9.55**, late cues **8.92**, and soft foreground weighting **7.75**. All four passed the structural checks. Early cues led on three cases. The two-cue average led on seamstress and tied early cues for mean Boss distinction gain (**+0.200**). Its higher mean damage (**0.329** versus **0.289**) contributed to a lower adjusted score. [Matched originals and renders](final-boss-raw192/ablations/cue-timing-raw144-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/cue-timing-raw144-scores.json)

The quarter-check question gave mean gains of **+0.513** for early cues, **+0.497** for the two-cue average, **+0.453** for late cues, and **+0.401** for soft foreground weighting.

## Late cues at update 192

The late-cue recipe used the 35% stronger-cue teacher at the final three denoising positions and retained the soft-foreground teacher at the other seven. It completed 192 updates and passed the structural checks at both shared strengths. At early α 2/3 it finished **4.46 points** behind soft foreground weighting. It had weaker mean Boss distinction gain (**+0.358** versus **+0.423**) and more visible damage (**0.233** versus **0.214**). Stationer, guide, and violinist favored late cues. Ranger, ferryman, and guard accounted for most of the decline. [Eight matched renders](final-boss-raw192/ablations/late-cues-raw192-matched.jpg) · [Both strengths and per-case scores](final-boss-raw192/ablations/late-cues-raw192-scores.json) · [Training source and commands](final-boss-raw192/README.md#source-for-the-trained-targets) · [Timing research](https://arxiv.org/abs/2512.08486)

## Early cues at update 192

The early-cue recipe used the same 35% blend at the first three denoising positions. It completed 192 updates and passed the structural checks at both shared strengths. At early α 2/3 it scored **20.32**, with Boss distinction gain of **+0.409** and mean damage of **0.288**. Compared with soft foreground weighting, **4.73** of the **5.34** lost points came from the damage term using the same per-case decomposition as above. Stationer improved, and watchmaker lost the most. At constant strength early cues scored **14.99**, between soft foreground weighting at **13.06** and full-trajectory stronger cues at **16.90**. [Eight matched renders](final-boss-raw192/ablations/early-cues-raw192-matched.jpg) · [Both strengths and per-case scores](final-boss-raw192/ablations/early-cues-raw192-scores.json) · [Training source and commands](final-boss-raw192/README.md#source-for-the-trained-targets)

## Qualified teacher targets

On eight archived teacher cases, guiding toward a concrete costume more strongly early in denoising scored **29.50–32.38**, versus **26.70** for the same costume prompt at fixed teacher strength. The range covers every possible outcome for one missing damage judgment. These are frozen Anima renders made before adapter training. [Per-case scores and gain schedule](final-boss-raw192/ablations/vibe-teacher-readability.json)

The training recipe pairs all **80** fields from four qualified costume teachers with **40** fields from two original soft-foreground contexts. The original and new recipes each use 120 fields across six contexts. The new recipe changes both the contexts and the teacher guidance. Its 192-update stock ParticleGAN run has started, with the shared development comparisons planned. The timing test draws on observations that text conditioning has different effects across denoising stages. [eDiff-I](https://arxiv.org/abs/2211.01324) · [Prompt-Conditioned Interventions](https://arxiv.org/abs/2512.08486)

## Structural stops

The 35% cue blend that kept only components with matching signs across its two alternate cue edits stopped at update 48. The bookkeeper render collapsed to a texture and scored 0.83 on Jev’s unreadable-image check. Training stopped before update 49. [Saved RAW48 render](final-boss-raw192/ablations/diverse-consensus35-raw48-bookkeeper.png) · [Consensus research](https://github.com/HyperGAN/particle-sliders/pull/315)

## Comparing a new run

Compare the saved RAW checkpoint with the incumbent on the same prompts, seeds, sampler, and shared strengths. At 192 updates the early score to beat is **25.66**. The 35% cue blend sets the constant-strength comparison at **16.90**. Report tuned strength and its confirmation panel alongside the shared-strength comparison.

Jev measures Boss character distinction and visible image damage separately. For each case the score is `max(0, Boss_ON − Boss_OFF) × (1 − image_damage)²`. The leaderboard averages the per-case scores on a 0–100 scale. Catastrophic structural failures and unmeasured rows carry separate statuses. [Jev rubrics](final-boss-raw192/rubrics.json)

The Particle Sliders CPU study ranked paired contexts first at early strength and background-only projection first at constant strength. Both winners differ from the native Anima rankings above. Use matched native renders to rank changes to the training recipe. [Particle Sliders PR #321](https://github.com/HyperGAN/particle-sliders/pull/321)
