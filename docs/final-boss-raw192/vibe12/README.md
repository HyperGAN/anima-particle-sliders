# Twelve-scene Final Boss development leaderboard

2026-11-25 05:35 UTC

113 settings from 13 trained recipes have all eight measurements. 2 recipes have finished the new search. Each row shows that recipe’s highest measured damage-adjusted Final Boss score across available strengths. Recipes with additional strengths still queued may move.

The sample strips show these cases from left to right — ranger, smith, stationer, ferryman, watchmaker, guide, guard, violinist.

![Slider off](samples/9961bd79a0a70a231c4f299dcf4693cf71677a021d05c7c195b69bddde024a55.webp)

| Rank | Training recipe | Schedule and α | Final Boss score ↑ | Boss gain ↑ | Damage ↓ | Matched samples |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | Background-only scene and lighting projection, 0.45 floor | early 1 | 12.50 | 71.50 | 38.79 | [![Eight edits](samples/0c8812aabf2c586f229d9eec16a75b5983440f45493e5a31ba1e491fccb50159.webp)](samples/0c8812aabf2c586f229d9eec16a75b5983440f45493e5a31ba1e491fccb50159.webp) |
| 2 | Stronger cues at the final three teacher positions | constant 1.34375 | 9.09 | 70.75 | 47.58 | [![Eight edits](samples/859ba1a154002a7781fffeaf7202b8aaa9120e200d4a16d60850d765095a8e95.webp)](samples/859ba1a154002a7781fffeaf7202b8aaa9120e200d4a16d60850d765095a8e95.webp) |
| 3 | Stronger cues at the first three teacher positions | constant 1 | 8.24 | 51.75 | 34.83 | [![Eight edits](samples/1879bd0b8daf65a22f61bdf4e7ab1ea43564560258f75a9a58aed3637ac9774a.webp)](samples/1879bd0b8daf65a22f61bdf4e7ab1ea43564560258f75a9a58aed3637ac9774a.webp) |
| 4 | Prompt-only control · all 240 fields | early 0.85 | 7.86 | 66.25 | 44.62 | [![Eight edits](samples/bcfaa77067057059456601919aee2c1774ec61c798a49052f1e275064a3d7005.webp)](samples/bcfaa77067057059456601919aee2c1774ec61c798a49052f1e275064a3d7005.webp) |
| 5 | Extra training exposure at the middle four denoising positions | early 1 | 6.91 | 58.75 | 45.00 | [![Eight edits](samples/ef31f7ef6d18ab7b86e6d73ad8af18f902cec0786b098acba415e913a68b977f.webp)](samples/ef31f7ef6d18ab7b86e6d73ad8af18f902cec0786b098acba415e913a68b977f.webp) |
| 6 | Six highest-divergence timestep positions | early 0.666667 | 6.88 | 52.88 | 40.79 | [![Eight edits](samples/08ca4675a83ad5d89bed799f17afb8da2c656d5fb271de02fc972384e84eeb50.webp)](samples/08ca4675a83ad5d89bed799f17afb8da2c656d5fb271de02fc972384e84eeb50.webp) |
| 7 | Extra training exposure at the opening four denoising positions | early 0.84375 | 3.75 | 43.38 | 43.33 | [![Eight edits](samples/6899d8093ec6ee58b10dc7535e5e5bdae911efd3580b79d40cc59cdc478dc5ac.webp)](samples/6899d8093ec6ee58b10dc7535e5e5bdae911efd3580b79d40cc59cdc478dc5ac.webp) |
| 8 | Second qualified seed on the shared teacher scenes | early 0.6375 | 3.70 | 45.62 | 44.88 | [![Eight edits](samples/1db9befc8424be5b5419e0b082f165e2a19fcbf6cc43eeaf0d92d86580b4e992.webp)](samples/1db9befc8424be5b5419e0b082f165e2a19fcbf6cc43eeaf0d92d86580b4e992.webp) |
| 9 | Percentile-filtered blend at the first three teacher positions | constant 0.6375 | 1.22 | 21.62 | 30.33 | [![Eight edits](samples/58b4ec1ea49ddb3870f02f399c6b35e152cb070035c7a0cc1056a2d096940da4.webp)](samples/58b4ec1ea49ddb3870f02f399c6b35e152cb070035c7a0cc1056a2d096940da4.webp) |
| 10 | Paired scene and lighting contexts, 50% blend, 0.45 floor | constant 0.6375 | 0.93 | 23.00 | 30.83 | [![Eight edits](samples/6b84d95f8ac27e2e7b1c8b3154c226e894baa74f8a940532887b47064bb3d008.webp)](samples/6b84d95f8ac27e2e7b1c8b3154c226e894baa74f8a940532887b47064bb3d008.webp) |
| 11 | Soft foreground weighting, 0.45 background floor | constant 0.6375 | 0.91 | 17.12 | 30.79 | [![Eight edits](samples/bf2a042511f47771f81af3b4c7a33535b4fd53fee57ae197e31c75e30cac46d9.webp)](samples/bf2a042511f47771f81af3b4c7a33535b4fd53fee57ae197e31c75e30cac46d9.webp) |
| 12 | Soft foreground weighting with full-strength Boss cues | constant 0.6375 | 0.86 | 24.00 | 39.50 | [![Eight edits](samples/f700da3fe457a0500b3146a6f84dd367c554f6d1cb4a318f3eab2448e0eda71d.webp)](samples/f700da3fe457a0500b3146a6f84dd367c554f6d1cb4a318f3eab2448e0eda71d.webp) |
| 13 | Soft foreground weighting plus a 35% blend of two averaged cues | constant 0.6375 | 0.80 | 19.50 | 35.00 | [![Eight edits](samples/b9abc62c596392d0751c8f560fe714517702c9c5da016a6b4af0d9883d713dee.webp)](samples/b9abc62c596392d0751c8f560fe714517702c9c5da016a6b4af0d9883d713dee.webp) |

The JSON includes every measured setting, original and edited image hashes, per-case Jev ratings, sample strips, native gains, RAW checkpoint hashes and completed search paths. [All measurements](standings.json) · [Post-training handoff](../vibe12-post-training.md)

Reproduce the ranking from the repository root.

```sh
python -m release_tools.final_boss_damage_search docs/final-boss-raw192/vibe12/standings.json
```
