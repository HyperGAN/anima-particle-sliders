# Anima Final Boss leaderboard

## Summary

**Current research default.** Soft foreground weighting with a 0.45 background floor, trained to RAW192 and replayed at early α 0.95. It scored **33.71** on eight development cases and **28.55** on eight separate confirmation cases. [Default and exact gains](final-boss-raw192/default.json) · [Matched confirmation images](final-boss-raw192/confirmed-default.jpg) · [Download RAW checkpoints](https://huggingface.co/ntc-ai/anima-final-boss-research)

**What worked in training.** Soft foreground weighting concentrates the teacher edit on the character while retaining some signal in the background. It leads the matched early-strength comparison at **25.66**. A 35% blend of stronger Boss cues scored **24.93** at early strength and led at constant strength with **16.90**. Paired contexts and projection sometimes improved readability at the cost of a weaker character change. [Training comparison](#training-techniques) · [Target-generation source](final-boss-raw192/README.md#source-for-the-trained-targets)

**What earns a high score.** The slider should make the subject visibly read as a Final Boss while keeping the image coherent. An edit with zero Boss gain scores zero. Visible damage lowers the score even when the Boss effect is strong. [Scoring and comparison protocol](#comparing-a-new-run) · [Jev rubrics](final-boss-raw192/rubrics.json)

**Latest challengers.** The all-step percentile filter scored **19.52** on the eight-case RAW192 leaderboard. The early-position filter's update-96 diagnostic scored **5.76** with one unresolved structure check. Matched ordinary attire scored **11.57** at RAW192 on the four screening cases versus **19.16** for soft foreground weighting. Its eight-case comparison is running. In-scene costumes, denoising-position exposure, regional cues, and an exterior nuisance cap are queued. [Percentile filtering](#percentile-filtered-teacher-edits) · [In-scene prompts](#in-scene-costume-prompts) · [Denoising-position exposure](#denoising-position-exposure) · [Regional stronger cues](#regional-stronger-cues) · [Matched ordinary attire](#matched-ordinary-attire) · [Exterior nuisance cap](#cap-added-exterior-nuisance)

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
| 7 | Percentile-filtered blend at all ten teacher positions | 19.52 | Unmeasured | [Anima comparison](#percentile-filtered-teacher-edits), based on [SEGA](https://arxiv.org/html/2301.12247) |
| 8 | Whole-image scene and lighting projection, 0.65 floor | 18.74 | 13.72 | Local Anima adaptation of [nuisance deflation #254](https://github.com/HyperGAN/particle-sliders/pull/254) |
| 9 | Soft foreground weighting plus a 35% blend of two averaged cues | 17.34 | 8.08 | [Anima comparison](#two-cue-average-at-update-192), related [cue diversification #313](https://github.com/HyperGAN/particle-sliders/pull/313) |
| 10 | Train on adapter rollout states, 0.45 floor | 11.61 | 5.49 | Local Anima experiment, related [rollout protocol #246](https://github.com/HyperGAN/particle-sliders/pull/246) |
| — | Four qualified costume teachers and two soft-foreground contexts | Unmeasured | 10.84 | [Matched renders and diagnostic scores](#costume-teachers-at-update-192) |

The prompt-only control scored **18.32** at shared early strength. Its constant result is unmeasured. The 35% cue blend gained **3.84 points** at constant strength over the default training recipe. It improved seven of eight matched cases at constant strength and one of eight at early strength. The 85% cue blend scored lower at both shared strengths. Each technique used one initialization seed. [All runs and per-case scores](final-boss-raw192/results.json)

Stronger Boss distinction accounts for **3.52** of the cue blend’s constant-strength gain. At early strength, paired contexts and whole-image projection improved readability, while weaker Boss distinction cost them **4.61** and **8.02** points respectively. These contributions compare each variant to soft foreground weighting. For each matched case, the calculation multiplies each factor’s change by the other factor’s average across the two renders, then averages across cases. [Per-case Boss and readability scores](final-boss-raw192/results.json)

[Shared early samples](final-boss-raw192/shared-early.jpg) · [Shared constant samples](final-boss-raw192/shared-constant.jpg) · [Training and replay source](final-boss-raw192/README.md#source-for-the-trained-targets)

## Update-48 preview

All five recipes passed the structural checks. Stronger cues at the first three teacher positions led with **3.23** at shared early α 2/3. Soft foreground weighting scored **2.34**, the two-cue average **1.94**, late cues **1.74**, and the qualified costume teachers **0.90**. The costume-teacher recipe had the highest mean image damage (**0.403**) and the weakest Boss distinction gain (**+0.020**). [Four matched cases across all recipes](final-boss-raw192/ablations/cue-timing-raw48-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/cue-timing-raw48-scores.json)

Mean gain under the quarter-check question was **+0.228** for early cues, **+0.186** for soft foreground weighting, **+0.176** for late cues, **+0.174** for the two-cue average, and **+0.149** for the costume teachers.

## Update-96 preview

The two-cue average blends 35% of the mean of two alternate teacher edit directions into the original edit and keeps the 0.45 background floor and original per-field L2 norm. It scored **10.34** at shared early α 2/3 and led three cases. Early cues scored **6.89**, late cues **6.76**, soft foreground weighting **5.73**, and qualified costume teachers **5.08**. All five passed the structural checks. The costume teachers led on seamstress and had the lowest mean damage (**0.263**), with weaker mean Boss distinction gain than soft foreground weighting (**+0.098** versus **+0.125**). [Matched originals and renders](final-boss-raw192/ablations/cue-timing-raw96-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/cue-timing-raw96-scores.json) · [Training source and commands](final-boss-raw192/README.md#source-for-the-trained-targets)

The quarter-check question gave mean Jev gains of **+0.526** for the two-cue average, **+0.408** for early cues, **+0.373** for soft foreground weighting, **+0.353** for late cues, and **+0.306** for costume teachers. The full training comparison uses the eight-case RAW192 panel.

Each cue-timing recipe replaces 36 of the 120 teacher fields. The replacement moves the edit by an average of **22.0%** of its original L2 norm at the first three positions and **16.8%** at the last three. The source and replacement fields have matched L2 norms. [Teacher field distances by position](final-boss-raw192/ablations/cue-field-distances.json)

The public capture command reproduced all **360** archived fields for the early-cue, late-cue, and two-cue-average recipes in fresh CUDA runs. [Source verification](final-boss-raw192/ablations/cue-native-source-verification.json)

## Update-144 preview

At shared early α 2/3, early cues scored **9.78**, the two-cue average **9.55**, late cues **8.92**, soft foreground weighting **7.75**, and qualified costume teachers **5.13**. All five passed the structural checks. Early cues led on three cases, and the two-cue average led on seamstress. The costume teachers had the lowest mean damage (**0.258**) and the weakest mean Boss distinction gain (**+0.090**). [Matched originals and renders](final-boss-raw192/ablations/cue-timing-raw144-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/cue-timing-raw144-scores.json)

The quarter-check question gave mean gains of **+0.513** for early cues, **+0.497** for the two-cue average, **+0.453** for late cues, **+0.401** for soft foreground weighting, and **+0.318** for the costume teachers.

## Late cues at update 192

The late-cue recipe used the 35% stronger-cue teacher at the final three denoising positions and retained the soft-foreground teacher at the other seven. It completed 192 updates and passed the structural checks at both shared strengths. At early α 2/3 it finished **4.46 points** behind soft foreground weighting. It had weaker mean Boss distinction gain (**+0.358** versus **+0.423**) and more visible damage (**0.233** versus **0.214**). Stationer, guide, and violinist favored late cues. Ranger, ferryman, and guard accounted for most of the decline. [Eight matched renders](final-boss-raw192/ablations/late-cues-raw192-matched.jpg) · [Both strengths and per-case scores](final-boss-raw192/ablations/late-cues-raw192-scores.json) · [Training source and commands](final-boss-raw192/README.md#source-for-the-trained-targets) · [Timing research](https://arxiv.org/abs/2512.08486)

## Early cues at update 192

The early-cue recipe used the same 35% blend at the first three denoising positions. It completed 192 updates and passed the structural checks at both shared strengths. At early α 2/3 it scored **20.32**, with Boss distinction gain of **+0.409** and mean damage of **0.288**. Compared with soft foreground weighting, **4.73** of the **5.34** lost points came from the damage term using the same per-case decomposition as above. Stationer improved, and watchmaker lost the most. At constant strength early cues scored **14.99**, between soft foreground weighting at **13.06** and full-trajectory stronger cues at **16.90**. [Eight matched renders](final-boss-raw192/ablations/early-cues-raw192-matched.jpg) · [Both strengths and per-case scores](final-boss-raw192/ablations/early-cues-raw192-scores.json) · [Training source and commands](final-boss-raw192/README.md#source-for-the-trained-targets)

## Two-cue average at update 192

The two-cue average completed 192 updates and passed the structural checks at both strengths. At shared early α 2/3 it scored **17.34**, **8.32** points behind soft foreground weighting. It trailed on all eight development cases. Weaker Boss distinction accounted for **4.32** lost points and greater image damage **4.00**. Ranger and watchmaker had the largest declines. The four-case update-96 lead did not carry through to the full comparison. [Eight matched renders](final-boss-raw192/ablations/two-cue-average-raw192-matched.jpg) · [Both strengths and per-case scores](final-boss-raw192/ablations/two-cue-average-raw192-scores.json) · [Training source and commands](final-boss-raw192/README.md#source-for-the-trained-targets)

## Qualified teacher targets

On eight archived teacher cases, guiding toward a concrete costume more strongly early in denoising scored **29.50–32.38**, versus **26.70** for the same costume prompt at fixed teacher strength. The range covers every possible outcome for one missing damage judgment. These are frozen Anima renders made before adapter training. [Per-case scores and gain schedule](final-boss-raw192/ablations/vibe-teacher-readability.json)

The training recipe pairs all **80** fields from four qualified costume teachers with **40** fields from two original soft-foreground contexts. The original and new recipes each use 120 fields across six contexts. The new recipe changes both the contexts and the teacher guidance. Its first attempt stopped before any updates because the cache lacked the RAW export manifest digest. The retry preserved all 120 fields, reproduced six fields exactly in CUDA admission, and started the 192-update stock ParticleGAN fit. The timing test draws on observations that text conditioning has different effects across denoising stages. [Cache and score evidence](final-boss-raw192/ablations/vibe-teacher-readability.json) · [eDiff-I](https://arxiv.org/abs/2211.01324) · [Prompt-Conditioned Interventions](https://arxiv.org/abs/2512.08486)

## Costume teachers at update 192

On the four checkpoint cases, the costume-teacher adapter scored **25.70** versus **19.16** for soft foreground weighting. It won three cases. Mean Boss distinction gain rose from **+0.368** to **+0.418**, and mean damage fell from **0.266** to **0.206**. [Four matched checkpoint cases](final-boss-raw192/ablations/costume-teacher-raw192-preview-matched.jpg) · [Checkpoint scores](final-boss-raw192/ablations/costume-teacher-raw192-preview-scores.json)

On the full eight-case panel at early α 2/3, Jev left the violinist's armor structure unresolved. The other seven passed. Separately scoring all eight images for Boss effect and damage gave **21.47** versus **25.66** for soft foreground weighting. Mean Boss distinction gain rose from **+0.423** to **+0.436**, but mean damage rose from **0.214** to **0.291**. Ranger and guide improved. The constant-strength comparison passed and scored **10.84**. [Eight matched renders](final-boss-raw192/ablations/costume-teacher-raw192-matched.jpg) · [Full diagnostic scores](final-boss-raw192/ablations/costume-teacher-raw192-diagnostic.json) · [Qualified score records](final-boss-raw192/ablations/costume-teacher-raw192-scores.json)

## Percentile-filtered teacher edits

SEGA uses a percentile of the absolute prompt-response difference to select components for guidance. The Anima training adaptation blends 35% of the norm-matched top 10% of the soft-foreground edit into the dense edit, then restores the original per-field L2 norm. One run applies the blend at the first three denoising positions. The other applies it at all ten. Both passed native target admission. The all-step comparison reached RAW192 and the early-position run is still training. [SEGA paper](https://arxiv.org/html/2301.12247)

At update 48, the all-ten-position blend passed all four structural checks and scored **2.22**, close to soft foreground weighting at **2.34**. It led on three cases and trailed on archivist. Mean image damage was **0.208** versus **0.268** for soft foreground weighting. Both showed weak Boss distinction at this checkpoint. [Matched update-48 renders](final-boss-raw192/ablations/sega-all-raw48-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/sega-all-raw48-scores.json)

The early-position filter also passed all four checks at update 48. It scored **3.22** and improved bookkeeper and seamstress relative to soft foreground weighting at **2.34**. Mean Boss distinction gain rose from **+0.048** to **+0.063**, and mean damage rose from **0.268** to **0.279**. [Both filters at update 48](final-boss-raw192/ablations/sega-early-raw48-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/sega-early-raw48-scores.json)

At update 96, the all-step filter passed all four structural checks and scored **4.89** at shared early α 2/3. Soft foreground weighting scored **5.73**, and the two-cue average scored **10.34** at this checkpoint. Percentile filtering lowered mean damage from **0.315** to **0.265**, while mean Boss distinction gain fell from **+0.125** to **+0.088**. Archivist and seamstress improved, and bookkeeper dropped the most. [Matched update-96 renders](final-boss-raw192/ablations/sega-all-raw96-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/sega-all-raw96-scores.json)

The early-position filter's diagnostic scored **5.76** on the same four update-96 cases. Jev left seamstress's structure unresolved. Mean Boss distinction gain was **+0.133** versus **+0.125** for soft foreground weighting, and mean damage was **0.342** versus **0.315**. Archivist and botanist improved, and bookkeeper accounted for the largest decline. [Both filters at update 96](final-boss-raw192/ablations/sega-early-raw96-matched.jpg) · [Scores and structural statuses](final-boss-raw192/ablations/sega-early-raw96-scores.json)

At update 144, all four structural checks passed. The all-step filter scored **4.57** versus **7.75** for soft foreground weighting, with lower scores on all four cases. Mean Boss distinction gain was **+0.128** versus **+0.168**, and mean visible damage was **0.375** versus **0.302**. The earlier damage advantage disappeared at this checkpoint. [Matched update-144 renders](final-boss-raw192/ablations/sega-all-raw144-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/sega-all-raw144-scores.json)

The all-step filter completed **192 updates** and passed the four screening cases at shared early α 2/3. It scored **11.22** versus **19.16** for soft foreground weighting on those four cases. Boss distinction gain fell from **+0.368** to **+0.218** on average. Mean damage rose from **0.266** to **0.283**. Bookkeeper accounted for the biggest score loss. [RAW192 screening renders](final-boss-raw192/ablations/sega-all-raw192-screen-matched.jpg) · [Per-case screening scores](final-boss-raw192/ablations/sega-all-raw192-screen-scores.json)

On all eight leaderboard cases, the all-step filter passed the structural checks and scored **19.52** at shared early α 2/3. It trailed soft foreground weighting on every case. The per-case score decomposition attributes **5.60** of the **6.13** lost points to weaker Boss distinction and **0.53** to visible damage. At shared constant α 0.6375, all eight structural checks passed and Jev left guard's Boss distinction unmeasured, so the constant aggregate is unmeasured. The corrected evaluation used the saved RAW192 and the pinned leaderboard rubric. [Matched RAW192 renders](final-boss-raw192/ablations/sega-all-raw192-matched.jpg) · [Both strengths and per-case scores](final-boss-raw192/ablations/sega-all-raw192-scores.json)

## In-scene costume prompts

On eight matched frozen-teacher cases, the in-scene armor-and-crown prompt scored **36.96** under the leaderboard rubric. The worn helmet scored **25.35** and the amplified mask **25.03**. In-scene costume beat amplified mask on six cases. It had lower mean Boss distinction gain (**+0.729** versus **+0.826**) and much lower mean damage (**0.292** versus **0.464**). Several amplified-mask renders introduced inset portraits or multiple views. All three recipes used the frozen model with zero optimizer updates. [Matched teacher renders](final-boss-raw192/ablations/grounded-teacher-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/grounded-teacher-scores.json)

The queued training test gives the original six contexts in-scene prompts that describe the existing costume as worn by the subject under the original light. Both arms use the original trajectories, 0.45 background floor, and per-field L2 normalization. One mixes in the grounded contrast at 35% throughout. The other tapers the mix from 39.2% to 27.3% across the ten teacher positions. Both are planned for 192 stock ParticleGAN updates and the shared-strength evaluation. [Exact prompts and mixing weights](final-boss-raw192/ablations/grounded-training-prompts.json)

Both caches are prepared on CUDA. The 120 latents, neutral velocities, and embeddings match the soft-foreground cache. The taper changes the target direction by up to **4.58%** of the edit norm on average at a denoising position and preserves the per-field L2 norm. The fixed blend is reserved for the next released training lane, ahead of the taper comparison. [Cache and field comparison](final-boss-raw192/ablations/grounded-training-prompts.json)

## Denoising-position exposure

Diffusion studies found that the training signal from different noise levels can emphasize different visual concepts or pull optimization in different directions. This Anima adaptation tests whether spending more of the same training budget at certain denoising positions changes the finished slider. The opening positions test larger character changes. The middle positions test whether costume and silhouette can become more distinct with less damage. [Perception Prioritized Training](https://arxiv.org/abs/2204.00227) · [Min-SNR Training](https://arxiv.org/abs/2303.09556)

Both arms retain all 120 soft-foreground teacher fields and repeat 48 of them once. The opening arm repeats zero-indexed positions 0–3. The middle arm repeats positions 3–6. Each arm uses the stock balanced shuffle, 192 updates, and 384 row presentations per optimizer side. The original and repeated tensors matched the source bit for bit. The sampler replay drew every row in both arms, with identical row ID sequences across arms. Training is queued behind the in-scene comparison. [Cache and sampler verification](final-boss-raw192/ablations/phase-exposure-protocol.json) · [Reproduction commands](final-boss-raw192/README.md#denoising-position-exposure)

## Regional stronger cues

The 35% stronger-cue recipe moved an average of **16.9%** of its squared change from the baseline into locations outside the subject matte. Blended Diffusion preserves the surrounding image by applying edits locally at successive denoising steps. This training adaptation applies that idea to the added cue in each frozen teacher field. It keeps the soft-foreground edit unchanged outside the allowed region. It also matches both the total baseline edit norm and the distance from the baseline to the whole-field stronger cue. [Blended Diffusion](https://arxiv.org/abs/2111.14818) · [Field audit and recipe](final-boss-raw192/ablations/regional-cue-protocol.json)

The original matte and a two-site latent halo passed the norm and support checks on all 120 archived fields. The matte contained the entire change within the original subject. The halo left **0.7%** outside it, with the remainder concentrated on the subject. Independent CUDA capture reproduced all **240** teacher fields bit for bit and both caches passed native admission. Both training runs are queued with the same RAW192 budget and shared strengths. [Frozen-field checks](final-boss-raw192/ablations/regional-cue-protocol.json) · [CUDA verification](final-boss-raw192/ablations/regional-cue-native-verification.json) · [Capture commands and source](final-boss-raw192/README.md#regional-stronger-cues)

### Halo from teacher contrast

DiffEdit estimates where to edit by comparing a diffusion model's predictions under two prompts. The training adaptation averages the stronger-cue response across positions 3–6 of both saved trajectories and uses it to relocate the fixed halo's exterior pixels within four latent sites of the subject. The mask keeps its interior values, exterior area, and exterior opacities. It uses the same norm and cue-distance constraints as the fixed halo. [DiffEdit, §3.2](https://openreview.net/pdf?id=3lge0p5o-M-) · [Mask source](../release_tools/final_boss_halo.py)

At the 72 denoising fields outside the positions used to build the masks, the inferred halo covered **61.6%** of the stronger cue's available exterior energy versus **51.6%** for the fixed halo. Mean full-field directional agreement rose from **0.902** to **0.904**. All 120 prepared fields retained the original soft-foreground values outside their selected support. Native admission and a trained-image comparison are pending. [Field audit](final-boss-raw192/ablations/contrast-halo-field-audit.json)

## Matched ordinary attire

Concept Sliders uses a contrast between prompts to emphasize one attribute and suppress another. The Anima test keeps the exact stronger Boss prompt and compares it with an ordinary outfit that echoes its colors and materials, with the same subject, framing, background, and lighting. This asks whether the stronger cue can teach the Boss silhouette and expression with a smaller contribution from costume palette and fabric changes. The stock fit uses the same 35% blend, 0.45 background floor, and per-field edit norm. [Concept Sliders method](https://arxiv.org/html/2311.12092v2#S4) · [Six exact prompt pairs](final-boss-raw192/ablations/matched-attire-prompts.json)

The prepared CUDA cache retains all 120 original latents, neutral velocities and embeddings. The matched reference changes the edit direction by an average of **31.0%** of the baseline edit norm relative to the original stronger-cue recipe. The per-field edit norm matched the baseline within **0.000021%**. RAW192 training has started. [Native field comparison](final-boss-raw192/ablations/matched-attire-targets.json) · [Capture commands and source](final-boss-raw192/README.md#matched-ordinary-attire)

At update 48, all four structural checks passed. Matched attire scored **2.14**, soft foreground weighting scored **2.34**, and the original 35% stronger cue scored **3.74** at the same strength. Mean Boss distinction gain was similar across the three recipes (**+0.048** to **+0.058**). The matched-attire renders showed a warm cast and fine texture in the backgrounds. Mean damage was **0.361** versus **0.268** for soft foreground weighting and **0.204** for the stronger cue. Bookkeeper improved over soft foreground weighting. [Matched update-48 renders](final-boss-raw192/ablations/matched-attire-raw48-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/matched-attire-raw48-scores.json)

The matched-attire edit moved farther outside the subject matte in the training fields. Relative to soft foreground weighting, the exterior change averaged **12.2%** of the baseline edit norm, versus **7.2%** for the original stronger cue. The overall edit moved farther, so the exterior's share of the squared change fell from **16.9%** to **12.8%**. [Spatial field audit](final-boss-raw192/ablations/matched-attire-spatial-audit.json)

At update 96, Jev left the matched-attire seamstress's structure unresolved. The original stronger cue had one unresolved structure check on botanist. On the four screening cases, the diagnostic scores were **5.19** for matched attire and **4.05** for the stronger cue. Soft foreground weighting passed the four checks and scored **5.73**. Matched attire raised mean Boss distinction gain from **+0.125** to **+0.130** and mean damage from **0.315** to **0.374** relative to soft foreground weighting. [Matched update-96 renders](final-boss-raw192/ablations/matched-attire-raw96-matched.jpg) · [Scores and structural statuses](final-boss-raw192/ablations/matched-attire-raw96-scores.json)

At update 144, all three recipes passed all four structural checks. Matched attire scored **8.27**, soft foreground weighting scored **7.75**, and the original stronger cue scored **8.43**. Matched attire improved bookkeeper and seamstress relative to soft foreground weighting. Lower damage contributed **+0.94** points, while weaker Boss distinction contributed **−0.42** points. [Matched update-144 renders](final-boss-raw192/ablations/matched-attire-raw144-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/matched-attire-raw144-scores.json)

At RAW192, all three recipes passed the four structural checks. At shared early α 2/3, matched attire scored **11.57**, soft foreground weighting scored **19.16**, and the original stronger cue scored **21.41**. Versus soft foreground weighting, lower damage added **1.61** points and weaker Boss distinction cost **9.19** points. The largest decline was on seamstress, where Boss gain fell from **+0.54** to **+0.15**. [Matched RAW192 screening renders](final-boss-raw192/ablations/matched-attire-raw192-screen-matched.jpg) · [Per-case scores](final-boss-raw192/ablations/matched-attire-raw192-screen-scores.json)

The queued regional-attire test applies the matched-reference change inside the subject matte, starting from the 35% stronger-cue target. It preserves the full edit norm and the size of the change introduced by the matched reference. On all **120** archived fields the exterior matched the starting target bit for bit. The localized change had a mean directional cosine of **0.929** with the full matched-reference change. Native recapture and RAW192 training are pending. [Field audit](final-boss-raw192/ablations/regional-attire-field-audit.json) · [Regional target method](#regional-stronger-cues)

## Cap added exterior nuisance

The 35% stronger cue improves the constant-strength score and changes the training signal along the measured scene and lighting direction. Earlier exterior orthogonalization removed a component averaging **29.1%** of the baseline exterior edit norm. The stronger cue exceeds the baseline along that direction in **101 of 120** fields, with a mean positive excess of **2.9%** of the baseline exterior norm across all fields. This test adapts the idea of correcting only the excess above a bound from Concept Removal Guidance. [Concept Removal Guidance](https://arxiv.org/html/2606.29801v1#S3) · [Older nuisance projection](https://github.com/HyperGAN/particle-sliders/pull/254) · [Field audit](final-boss-raw192/ablations/exterior-nuisance-cap-audit.json)

The candidate starts from the 35% stronger-cue target. It caps the magnitude of the exterior projection at the soft-foreground baseline for that exact field and preserves the exterior edit norm. The stronger cue's foreground values match bit for bit in all 120 archived fields. The full edit changed by **1.0%** of its original norm on average, with 19 fields left intact. Native cache preparation and the shared RAW192 comparison are queued. [Exact rule and checks](final-boss-raw192/ablations/exterior-nuisance-cap-audit.json)

## Structural stops

The 35% cue blend that kept only components with matching signs across its two alternate cue edits stopped at update 48. The bookkeeper render collapsed to a texture and scored 0.83 on Jev’s unreadable-image check. Training stopped before update 49. [Saved RAW48 render](final-boss-raw192/ablations/diverse-consensus35-raw48-bookkeeper.png) · [Consensus research](https://github.com/HyperGAN/particle-sliders/pull/315)

## Comparing a new run

Compare the saved RAW checkpoint with the incumbent on the same prompts, seeds, sampler, and shared strengths. At 192 updates the early score to beat is **25.66**. The 35% cue blend sets the constant-strength comparison at **16.90**. Report tuned strength and its confirmation panel alongside the shared-strength comparison.

Jev measures Boss character distinction and visible image damage separately. For each case the score is `max(0, Boss_ON − Boss_OFF) × (1 − image_damage)²`. The leaderboard averages the per-case scores on a 0–100 scale. Catastrophic structural failures and unmeasured rows carry separate statuses. [Jev rubrics](final-boss-raw192/rubrics.json)

The Particle Sliders CPU study ranked paired contexts first at early strength and background-only projection first at constant strength. Both winners differ from the native Anima rankings above. Use matched native renders to rank changes to the training recipe. [Particle Sliders PR #321](https://github.com/HyperGAN/particle-sliders/pull/321)
