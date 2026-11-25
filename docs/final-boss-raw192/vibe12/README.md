# Twelve-scene Final Boss development leaderboard

2026-11-25 06:21 UTC

**Opening-four exposure at early α 1.15625** is the visual comparison reference. The table below ranks all recipes using the same formula.

![Opening-four reference](samples/cafa6a533065473db410ac42e58a11dfb2df8e994f6bccc9c9e5d63c9585a1ed.webp)

[Blind finalist comparisons and case sensitivity](finalist-review.md)

204 settings from 16 trained recipes have all eight measurements. 8 recipes have finished the new search. Each row shows that recipe’s highest measured damage-adjusted Final Boss score across available strengths. Recipes with additional strengths still queued may move.

The sample strips show these cases from left to right — ranger, smith, stationer, ferryman, watchmaker, guide, guard, violinist.

![Slider off](samples/9961bd79a0a70a231c4f299dcf4693cf71677a021d05c7c195b69bddde024a55.webp)

| Rank | Training recipe | Schedule and α | Final Boss score ↑ | Boss gain ↑ | Damage ↓ | Matched samples |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | Stronger cues at the final three teacher positions | early 1.15625 | 37.09 | 77.50 | 26.56 | [![Eight edits](samples/d442fca93d0c41a5bcdb4543bdc95db318c5a289648a2786e4ce65de47e9b101.webp)](samples/d442fca93d0c41a5bcdb4543bdc95db318c5a289648a2786e4ce65de47e9b101.webp) |
| 2 | Background-only scene and lighting projection, 0.45 floor | early 1.48438 | 35.42 | 77.62 | 29.91 | [![Eight edits](samples/74fbddfc796d039e19342e3f9cf392a5164faaf305520968f93ca3239400cd37.webp)](samples/74fbddfc796d039e19342e3f9cf392a5164faaf305520968f93ca3239400cd37.webp) |
| 3 | Extra training exposure at the middle four denoising positions | early 1.4375 | 34.39 | 76.00 | 27.09 | [![Eight edits](samples/d9d29020386e7258cfeae4414ab10470ff353c9e6b59ca3a012756dfa4d3e91f.webp)](samples/d9d29020386e7258cfeae4414ab10470ff353c9e6b59ca3a012756dfa4d3e91f.webp) |
| 4 | Stronger cues at the first three teacher positions | early 1.3125 | 33.50 | 77.25 | 33.03 | [![Eight edits](samples/6a7211acae36d8895de03a0ba4756b507525594bb471a2c824895891518c653c.webp)](samples/6a7211acae36d8895de03a0ba4756b507525594bb471a2c824895891518c653c.webp) |
| 5 | Six highest-divergence timestep positions | early 1.25 | 33.48 | 75.88 | 28.59 | [![Eight edits](samples/ef3e62ea19eec9890e5d467c1614ed7c028e19bf3d69d3e1e677a91d13230905.webp)](samples/ef3e62ea19eec9890e5d467c1614ed7c028e19bf3d69d3e1e677a91d13230905.webp) |
| 6 | Extra training exposure at the opening four denoising positions | early 1.25 | 33.02 | 72.88 | 26.06 | [![Eight edits](samples/d70d807e7512117a3278aa046b1b8fe92e58888b354c6aa1c98e71265a3bc3e5.webp)](samples/d70d807e7512117a3278aa046b1b8fe92e58888b354c6aa1c98e71265a3bc3e5.webp) |
| 7 | Prompt-only control · all 240 fields | early 1.25 | 31.79 | 75.38 | 30.09 | [![Eight edits](samples/175065fdf76950edf16b15a66b1b9441d9e2e1496dee9c06358bbd3bbabce983.webp)](samples/175065fdf76950edf16b15a66b1b9441d9e2e1496dee9c06358bbd3bbabce983.webp) |
| 8 | Soft foreground weighting, 0.45 background floor | early 1.3125 | 28.63 | 72.75 | 31.37 | [![Eight edits](samples/8dfd53b332720dc08c4299ccbf003fd98e99ac114eb1dfd2e164241f224b8b09.webp)](samples/8dfd53b332720dc08c4299ccbf003fd98e99ac114eb1dfd2e164241f224b8b09.webp) |
| 9 | Second qualified seed on the shared teacher scenes | early 1.25 | 27.20 | 73.62 | 34.75 | [![Eight edits](samples/ad74d8cd9fc2630325d431716e14289117fd6587faf26abeb4350213542dd3c3.webp)](samples/ad74d8cd9fc2630325d431716e14289117fd6587faf26abeb4350213542dd3c3.webp) |
| 10 | Soft foreground weighting with full-strength Boss cues | early 1.125 | 25.01 | 71.25 | 33.97 | [![Eight edits](samples/44fc525f021bb8498889726da65ace13768b20a435014ea82d625612923de740.webp)](samples/44fc525f021bb8498889726da65ace13768b20a435014ea82d625612923de740.webp) |
| 11 | Six evenly spaced timestep positions | early 1.40625 | 23.19 | 69.38 | 32.12 | [![Eight edits](samples/b4810c815c9689711fbaf8e1ff304ebf296341e843b93fdfdb0eb6be8528c6ef.webp)](samples/b4810c815c9689711fbaf8e1ff304ebf296341e843b93fdfdb0eb6be8528c6ef.webp) |
| 12 | Neutral trajectory only | early 1.625 | 12.87 | 57.12 | 33.09 | [![Eight edits](samples/a23ef775dd61debdf18c6d07dddf0b9b823f451df4b6c916d2fa46cd3ace6492.webp)](samples/a23ef775dd61debdf18c6d07dddf0b9b823f451df4b6c916d2fa46cd3ace6492.webp) |
| 13 | Soft foreground weighting · prior incumbent recipe | early 0.666667 | 11.41 | 46.75 | 25.56 | [![Eight edits](samples/e62859f259300a06105346d6de8a261543c2726f888790bbda3136d84567f2e7.webp)](samples/e62859f259300a06105346d6de8a261543c2726f888790bbda3136d84567f2e7.webp) |
| 14 | Percentile-filtered blend at the first three teacher positions | constant 0.6375 | 2.04 | 21.62 | 20.19 | [![Eight edits](samples/58b4ec1ea49ddb3870f02f399c6b35e152cb070035c7a0cc1056a2d096940da4.webp)](samples/58b4ec1ea49ddb3870f02f399c6b35e152cb070035c7a0cc1056a2d096940da4.webp) |
| 15 | Paired scene and lighting contexts, 50% blend, 0.45 floor | constant 0.6375 | 1.66 | 23.00 | 24.31 | [![Eight edits](samples/6b84d95f8ac27e2e7b1c8b3154c226e894baa74f8a940532887b47064bb3d008.webp)](samples/6b84d95f8ac27e2e7b1c8b3154c226e894baa74f8a940532887b47064bb3d008.webp) |
| 16 | Soft foreground weighting plus a 35% blend of two averaged cues | constant 0.6375 | 1.35 | 19.50 | 22.59 | [![Eight edits](samples/b9abc62c596392d0751c8f560fe714517702c9c5da016a6b4af0d9883d713dee.webp)](samples/b9abc62c596392d0751c8f560fe714517702c9c5da016a6b4af0d9883d713dee.webp) |

The JSON includes every measured setting, original and edited image hashes, per-case Jev ratings, sample strips, native gains, RAW checkpoint hashes and completed search paths. [All measurements](standings.json) · [Post-training handoff](../vibe12-post-training.md)

Reproduce the ranking from the repository root.

```sh
python -m release_tools.final_boss_damage_search docs/final-boss-raw192/vibe12/standings.json
```
