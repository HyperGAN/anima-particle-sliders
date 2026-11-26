# Post-training the 512-update Final Boss slider

No arm met the proposal's bar for a win, and the top-scoring forks collapsed toward one design. The result at the published generator rate depends more on which checkpoint is measured than on the targets. The highest scores came from a short fork on cleaner teachers. Twelve additional updates on five less damaged Boss teachers gave the top two rows in two seeds, 30.83 and 29.49 distinct Boss points against the untouched parent's 27.39. The second seed has the least damage measured, 22.0 against 25.2 out of 100, at equal Boss gain and with the best structural record. Twelve updates of exact continuation on unchanged targets scored 29.41, so these rows do not separate cleaner teachers from a few more updates. Both top rows also dress every character as the same crimson spiked figure and tint the scene red, which the damage question does not count. By 36 and 48 updates every arm at this rate fell below the parent with more damage. One arm stopped on a structural failure. [Matched samples and every searched strength](http://100.90.104.57:8797/post-training-leaderboard.html?metric=design) · [The proposal this tests](../final-boss-raw192/vibe12-post-training.md)

Every row starts from the PG 0.12.0 Final Boss RAW512, SHA-256 `d3b828cfa4bccf0c20dc11587ed1080cc3eda2684a424e0079b334ed15e2b377`, trained on the late-cues targets at G 0.00075, D 0.009 and particle 0.03. Each RAW had its own constant-alpha search on four characters. That alpha was frozen before the eight ranking characters. This is the measurement behind the [constant-alpha leaderboard](../leaderboard.md), so these rows compare directly with it. Boss gain is ON minus OFF distinction and damage is Jev's visible-damage severity, both out of 100. Weights are the raw current weights. No EMA is involved.

| Rank | Approach | Updates | Constant α | Eight ranking characters | Distinct Boss / 100 | Boss gain | Damage | Repeat | Integrity | Training |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| — | Matched baseline | — | OFF | ![OFF](samples/off.webp) | — | — | — | — | — | — |
| 1 | Fork · five least-damaged teachers, gain not protected · +12 | 524 | 1.5 | ![Fork · five least-damaged teachers, gain not protected · +12](samples/cleanest-12.webp) | 30.83 | +80.2 | 27.3 | 55.0 | 3 pass, 1 fail, 4 unresolved | completed |
| 2 | Fork, second seed · five least-damaged teachers · +12 | 524 | 1.88867 | ![Fork, second seed · five least-damaged teachers · +12](samples/cleanest-seed2-12.webp) | 29.49 | +75.6 | 22.0 | 42.3 | 7 pass, 1 unresolved | completed |
| 3 | Exact continuation · parent critic and optimizer, same targets · +12 | 524 | 2.04688 | ![Exact continuation · parent critic and optimizer, same targets · +12](samples/continued-12.webp) | 29.41 | +79.1 | 28.3 | 51.9 | 6 pass, 1 fail, 1 unresolved | completed |
| 4 | Exact continuation · parent critic and optimizer, same targets · +36 | 548 | 1.625 | ![Exact continuation · parent critic and optimizer, same targets · +36](samples/continued-36.webp) | 28.87 | +77.9 | 29.0 | 48.1 | 6 pass, 2 fail | completed |
| 5 | Parent · no further training | 512 | 1.5625 | ![Parent · no further training](samples/parent-0.webp) | 27.39 | +74.6 | 25.2 | 42.9 | 6 pass, 2 unresolved | parent |
| 6 | Fork at the lower generator rate · five least-damaged teachers · +48 | 560 | 1.95312 | ![Fork at the lower generator rate · five least-damaged teachers · +48](samples/cleanest-low-g-48.webp) | 27.34 | +76.5 | 29.5 | 46.5 | 1 pass, 1 fail, 6 unresolved | completed |
| 7 | Fork control at the lower generator rate · same targets · +48 | 560 | 1.44141 | ![Fork control at the lower generator rate · same targets · +48](samples/control-low-g-48.webp) | 27.03 | +75.2 | 27.2 | 41.9 | 5 pass, 3 unresolved | completed |
| 8 | Fork control, second seed · same targets · +12 | 524 | 2.1875 | ![Fork control, second seed · same targets · +12](samples/control-seed2-12.webp) | 25.23 | +70.0 | 23.3 | 54.7 | 6 pass, 1 fail, 1 unresolved | completed |
| 9 | Fork control · fresh critic and optimizer, same targets · +36 | 548 | 1.5 | ![Fork control · fresh critic and optimizer, same targets · +36](samples/control-36.webp) | 24.55 | +76.1 | 33.4 | 54.9 | 1 pass, 7 unresolved | completed |
| 10 | Fork · five least-damaged teachers, gain not protected · +36 | 548 | 2.1875 | ![Fork · five least-damaged teachers, gain not protected · +36](samples/cleanest-36.webp) | 24.21 | +77.1 | 33.9 | 58.5 | 0 pass, 2 fail, 6 unresolved | completed |
| 11 | Fork · five least-damaged teachers, gain not protected · +48 | 560 | 1.38281 | ![Fork · five least-damaged teachers, gain not protected · +48](samples/cleanest-48.webp) | 24.06 | +73.8 | 29.5 | 45.2 | 3 pass, 5 unresolved | completed |
| 12 | Fork · one cleaner teacher that keeps its Boss gain · +36 | 548 | 1.27344 | ![Fork · one cleaner teacher that keeps its Boss gain · +36](samples/cleaner-36.webp) | 23.96 | +71.1 | 30.9 | 36.0 | 3 pass, 1 fail, 4 unresolved | stopped at +36 · structural failure |
| 13 | Exact continuation · parent critic and optimizer, same targets · +48 | 560 | 2 | ![Exact continuation · parent critic and optimizer, same targets · +48](samples/continued-48.webp) | 23.64 | +73.2 | 28.6 | 47.9 | 1 pass, 7 unresolved | completed |
| 14 | Fork control · fresh critic and optimizer, same targets · +12 | 524 | 2 | ![Fork control · fresh critic and optimizer, same targets · +12](samples/control-12.webp) | 20.98 | +71.8 | 27.3 | 69.7 | 1 pass, 2 fail, 5 unresolved | completed |
| 15 | Fork control · fresh critic and optimizer, same targets · +48 | 560 | 1.25 | ![Fork control · fresh critic and optimizer, same targets · +48](samples/control-48.webp) | 20.27 | +69.1 | 29.3 | 43.3 | 4 pass, 2 fail, 2 unresolved | completed |

The rows include every measured checkpoint, with the saved +12 and +36 quarters. A structural failure on any ranking image keeps a RAW out of release whatever its score. The one-teacher fork's +12 ranking is unmeasured because one of its 56 repeated-design judgments returned no answer. It selected α 1.8359375 with +73.8 Boss gain.

## What each arm changed

| Arm | Entry | Critic and optimizers | Targets | Generator rate |
| --- | --- | --- | --- | ---: |
| Exact continuation | resume the parent FULL512 | the parent's own | unchanged | 0.00075 |
| Fork control | fork the parent RAW512 | fresh, seed 1 | unchanged | 0.00075 |
| One cleaner teacher | fork | fresh, seed 1 | 1 of 12 scenes replaced | 0.00075 |
| Five least-damaged teachers | fork | fresh, seed 1 | 5 of 12 scenes replaced | 0.00075 |
| Second-seed pair | fork | fresh, seed 2 | unchanged and 5 replaced | 0.00075 |
| Lower-rate pair | fork | fresh, seed 1 | unchanged and 5 replaced | 0.0002109375 |

The proposal names a fork control and a cleaner-target fork. The exact continuation, the second teacher rule, the second seed and the lower rate were added during the run to explain what the first two showed.

## Findings

**Short helped and long hurt.** Across the four arms at the published rate, the three measured +12 checkpoints average 27.1 points, the +36 checkpoints 25.4 and the +48 checkpoints 22.7. In the paired comparison on the eight ranking characters, damage rose over the parent in twelve of fourteen post-trained RAWs, by 2.0 to 8.7 points. The two exceptions are the second-seed forks at +12, at −1.9 for unchanged targets and −3.2 for the five cleaner teachers.

**Cleaner teachers beat their matched fork control at +12 in both seeds.** The five-teacher fork ranked 30.83 against the control's 20.98 with seed 1, and 29.49 against 25.23 with seed 2. Its second seed is a Pareto improvement over the parent: +1.0 gain and −3.2 damage in the paired comparison, with seven structural checks passing and one unresolved. The parent has six and two. The proposal's bar is five damage points, so this is a lead and not a win. The exact continuation at +12 ranked 29.41 on unchanged targets with a structural failure, which leaves "twelve more updates" as an equally good explanation for the score. It did not lower damage.

**Forked children collapse toward one design.** A fork inherits the adapter and particle bank. The critic, both optimizers and the prior's moment statistics start again, so an untrained critic first pushes the trained generator in one common direction. Repeated design over the 28 character pairs is 42.9 for the parent. At +12 it is 69.7 and 54.7 for the two fork controls and 55.0 for the first five-teacher fork. The second five-teacher fork measures 42.3, yet its images share one crimson palette, its smith shows two figures and its violinist no longer shows a person. The exact continuation keeps the trained critic and measures 51.9, 48.1 and 47.9. A fork that also carries the parent's critic is the next arm to try.

**Checkpoint variation is as large as these differences.** The exact continuation keeps the parent's critic, optimizer state, random streams and targets. It ranked 29.41 at +12, 28.87 at +36 and 23.64 at +48. The fork control ranked 20.98, 24.55 and 20.27. The five-teacher fork ranked 30.83, 24.21 and 24.06. Two seeds per arm at one checkpoint is the minimum that could show a pattern, and it is all this run has.

**The published generator rate is too high for a short second stage.** At α 1 on the four training-selection characters the parent scores 2.5. The continuation then read 1.4, 4.8, 13.3 and 3.1 over four quarters, and the fork control 13.7, 4.4, 6.3 and 2.9. At G 0.0002109375 the fork control read 2.9, 2.9, 3.1 and 1.5 and passed every structural check. After 48 updates at that rate the fork control ranked 27.03 and the five-teacher fork 27.34, both level with the parent. The cleaner teachers did not lower damage there: 29.5 against the control's 27.2. The lower rate holds the parent's quality and moves too little in 48 updates to show a target effect.

**48 updates is a small dose.** The batch holds two fields, so 48 updates show the generator 96 of the 240 fields. A scene's twenty fields receive about eight visits. The parent had 1,024.

**One arm stopped.** The one-teacher fork produced a structural failure at +36 on the watchmaker screening scene, where hair, beard and cloak merged into one mass (broken structure 0.80). The trainer stopped it before another update. Its +36 RAW ranks 23.96 as a diagnostic.

**At matched effect, the best gap is under the proposal's bar.** On the four search characters the parent's chosen strength gives +74.0 gain at 27.7 damage. The least damaged searched strength within two gain points was 23.5 and 23.6 for the two five-teacher forks at +12, and 24.2 and 24.3 for the continuation at +36 and +12. The proposal asks for five points. The cleaner-target forks and the unchanged-target continuation are within one point of each other.

## The teachers

All 24 archived teacher images replayed byte for byte from their prompts and seeds. The twelve current Boss teachers measure 18.2 to 30.0 damage and +69 to +81 gain. The tea-shop teacher fails the existing separate-figure check: it keeps the seated woman and adds an armored figure behind her.

The six most damaged scenes received three proposals each, written for solid shapes and plain silhouettes, rendered from the same neutral prompt and seed. Fourteen of the eighteen proposals measured less damage than the teacher they would replace, down to 18.0. Only one also matched its gain. That is the proposal's rule, and it replaced the tea-shop teacher alone. The elaborate designs that score highest for Boss distinction are the same designs the damage question marks down, so the rule asks for something these proposals rarely have.

The second rule keeps the least damaged single-figure proposal without protecting gain. It replaced five scenes. The librarian kept its teacher because no proposal measured lower.

Replacement rebuilds the positive trajectory and the late-cues transformation for the changed scenes and keeps the neutral trajectory, seed, matte and row order. Every unchanged scene was regenerated through the same code and reproduced the parent's 20 fields bit for bit.

## The fork

`open_training(mode="fork", fork=<parent RAW pin>)` validates the RAW hash, its scoped sidecar and native export record, including model identity, raw selection and disabled EMA. It loads the parent tensors strictly before the optimizers exist. The critic is the one a fresh cohort with that seed starts with. The child counts its own updates from zero and records the inherited 512 in its receipt and in each RAW sidecar. A child FULL resumes through the existing exact path.

On this checkpoint each child began with all 4,033 parent tensors. Before its first update each child drew the parent's eight ranking images at α 1.5625 and all eight OFF images byte for byte. The software tests cover tensor equality, the fresh critic and optimizer state, refusal of averaged or mismatched parents before any component is built, and reproduction of the next update after reopening a child FULL.

The parent FULL binds the trainer source that wrote it. The exact continuation therefore ran from the parent study's own checkout, and the forks ran from this branch.

## What to change in the proposal

1. **Follow the +12 lead first.** Run the five-teacher fork, the fork control and the exact continuation to +12 on at least three more seeds each, and measure them on a fresh confirmation panel. That is the cheapest test of whether cleaner teachers add anything over a few more updates.
2. **Choose the rate and budget together.** At 0.00075 a checkpoint is not a stable estimate of its arm beyond a few updates. At 0.0002109375 quality holds, and 48 updates show the generator only 96 of the 240 fields. A lower-rate second stage needs at least 120 updates, one pass over every field.
3. **Use the exact continuation as the equal-update control whenever the parent FULL exists.** Keep the fork control beside it to price the restart of critic and optimizer state.
4. **Run several seeds per arm and fix the checkpoint rule first.** The first seed's +12 jump at α 1 did not recur with the second seed, and +48 was the worst checkpoint in three of four arms.
5. **Give the teacher rule the same tolerance as the outcome rule.** Requiring gain at least equal to the current teacher rejected a proposal that measured 0.01 lower in gain and 0.10 lower in damage.
6. **Search one constant alpha.** The proposal lists constant, early and late schedules. The Anima leaderboard uses one constant strength for all ten steps.
7. **Build the fresh confirmation panel before claiming a win.** These rows use the board's four search and eight ranking characters, which are development cases.

The narrower arm that replaces teacher velocities at the final three positions was not run. It needs a cache layout whose native admission recomputes two teachers per scene.

## Reproduce

[Scripts, policy and the eighteen proposals](../../research/post-training/README.md). The campaign directory is `/ml2/particlesliders-working/anima-final-boss-post-training-20261126`. It holds every arm's FULL and RAW checkpoints, update receipts, renders, Jev receipts and alpha probes. The parent study was read only.
