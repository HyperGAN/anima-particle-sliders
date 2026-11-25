# Selecting alpha for distinct Final Boss designs

The Final Boss search now uses the **distinct Boss score** to move the alpha bracket, pick finalists and save the winning replay. This is the same score shown on the design leaderboard. Each character earns credit for its Boss gain and finished-image quality. If two characters repeat the same design, they share credit in proportion to Jev’s repetition rating. [Formula and matched samples](../distinct-designs.md)

## Completed design searches in this release

| Recipe | Schedule | Alpha | Distinct Boss score | Saved setting |
| --- | --- | ---: | ---: | --- |
| late-cues | early | 1.15625 | 28.84 | [Tuning](late-cues/tuning.json) |
| shared-best-middle | early | 1.4375 | 27.74 | [Tuning](shared-best-middle/tuning.json) |
| shared-best-opening | early | 1.25 | 27.63 | [Tuning](shared-best-opening/tuning.json) |
| soft45-background-orthogonal | early | 1.484375 | 27.24 | [Tuning](soft45-background-orthogonal/tuning.json) |
| shared-best-divergence | early | 1.25 | 26.34 | [Tuning](shared-best-divergence/tuning.json) |
| soft-contexts-45-blend50 | early | 1.15625 | 26.11 | [Tuning](soft-contexts-45-blend50/tuning.json) |
| early-cues | early | 1.3125 | 25.55 | [Tuning](early-cues/tuning.json) |
| incumbent-refit | early | 1.25 | 24.98 | [Tuning](incumbent-refit/tuning.json) |
| strong-boss-45-mix35 | early | 1.0625 | 24.50 | [Tuning](strong-boss-45-mix35/tuning.json) |
| two-cue-average | early | 1.125 | 21.63 | [Tuning](two-cue-average/tuning.json) |
| soft-projection-65-blend50 | early | 1.625 | 21.58 | [Tuning](soft-projection-65-blend50/tuning.json) |
| strong-boss-45-dose100 | early | 1.4375 | 21.26 | [Tuning](strong-boss-45-dose100/tuning.json) |
| shared-best-seed-split | early | 1.25 | 21.18 | [Tuning](shared-best-seed-split/tuning.json) |
| shared-best-uniform-steps | early | 1.40625 | 19.25 | [Tuning](shared-best-uniform-steps/tuning.json) |
| subject-apg | early | 1.25 | 13.58 | [Tuning](subject-apg/tuning.json) |

## Opening four at RAW192

The search selected **early α 1.25**, with an eight-character score of **27.63**. The four-character screen preferred early α 1.4375 within the early schedule. On the eight development characters it scored 25.38. This is a measured selection from a finite search.

| Schedule | Alpha | Eight-character distinct Boss score |
| --- | ---: | ---: |
| Early | 1.25 | 27.63 |
| Early | 1.15625 | 26.19 |
| Constant | 1.625 | 25.93 |
| Constant | 1.75 | 25.79 |
| Early | 1.4375 | 25.38 |

All 27 search probes started from saved renders. Measurement took 191 seconds on this run, including about 140 seconds across the four-character probes and 51 seconds across the eight-character comparisons. Fresh native renders add latency. The downloaded tuning includes the RAW checkpoint hash, native model and sampler, winning profile, scalar alpha and the ten resolved interval gains. [Opening-four saved tuning](shared-best-opening/tuning.json) · [Alpha path and exact case ratings](shared-best-opening/search.json) · [Jev responses](shared-best-opening/receipts.json.gz)

## Late cues at RAW192

The second completed search selected **early α 1.15625** at **28.84**, followed by early α 1.25 at 28.27. The 27 probes and five full comparisons took 178 seconds using saved renders. Both completed recipe searches independently kept their previous operating point. [Late-cue saved tuning](late-cues/tuning.json) · [Measured path and case ratings](late-cues/search.json)

## Search and confirmation

1. Start each of the constant, early and late schedules at α 1.00. Probe both directions at 0.25 and adapt the bracket for up to nine distinct probes per schedule.
2. Evaluate each probe on four search characters. Six pairs in two image orders give 12 Jev design judgments before cache reuse. The two highest-scoring alphas per schedule advance.
3. Select the final alpha on eight matched development characters. Each full comparison uses 28 pairs in both orders. Include earlier measured strengths of the same RAW adapter and the two shared controls. A strength can be skipped once its per-image score is at or below the current best distinct Boss score, because design repetition can only reduce its score. Eight historical strengths were ruled out this way for opening four.
4. Save the winning settings and their measurement receipt. The live design leaderboard switches each recipe to its tuned setting on completion. It labels recipes still using a strength inherited from the previous search.

On the equipped research host, the standard queue entry point is `python strength-search/adaptive_queue.py`, run from the campaign directory. Per-recipe replay settings are stored under `strength-search/leaderboard-retune-design/<recipe>/tuning.json`. The historical per-image queue can be run explicitly with `--objective visible`.

Verify the public selection and reconstruct its search path from the repository root.

```sh
python -m release_tools.final_boss_design_alpha docs/final-boss-raw192/vibe12/alpha-tuning/shared-best-opening
```
