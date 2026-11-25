# Distinct Final Boss designs across characters

2026-11-25 07:05 UTC

Opening-four exposure places **3 of 22** across matched characters, up from 8 on the per-image score. Its average design repetition is 35.7/100, compared with 49.0/100 for late cues. 2 of these recipes have completed the distinct-design alpha search. Remaining recipes enter at their best measured per-image strength.

Jev compares the character designs in each pair of finished images. It looks for repeated silhouettes, body construction, masks, costume shapes and visual motifs. A palette change or a different background alone carries little weight. On each pair, distinct bosses earn both per-image scores. If the same design repeats, the pair earns the stronger score. Jev’s continuous repetition rating interpolates between those outcomes. The per-image scores already account for matched Final Boss gain and visible damage. Zero-effect edits contribute zero.

Every row uses the same eight cases and the same RAW192 budget. Jev rates all 28 pairs in both image orders. Its repetition rating is an ordinal score on a 0–100 scale. The current alpha search uses this cross-character score on four search characters and on eight development characters for the final choice. [Saved alpha and search evidence](alpha-tuning/README.md)

| Rank | Recipe | Schedule and α | Distinct Boss score ↑ | Per-image score ↑ | Design repetition ↓ | Matched samples |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | Stronger cues at the final three teacher positions | Design-tuned · early 1.15625 | 28.84 | 37.09 | 49.0 | [![Eight edits](samples/d442fca93d0c41a5bcdb4543bdc95db318c5a289648a2786e4ce65de47e9b101.webp)](samples/d442fca93d0c41a5bcdb4543bdc95db318c5a289648a2786e4ce65de47e9b101.webp) |
| 2 | Extra training exposure at the middle four denoising positions | early 1.4375 | 27.74 | 34.39 | 43.3 | [![Eight edits](samples/d9d29020386e7258cfeae4414ab10470ff353c9e6b59ca3a012756dfa4d3e91f.webp)](samples/d9d29020386e7258cfeae4414ab10470ff353c9e6b59ca3a012756dfa4d3e91f.webp) |
| 3 | Extra training exposure at the opening four denoising positions | Design-tuned · early 1.25 | 27.63 | 33.02 | 35.7 | [![Eight edits](samples/d70d807e7512117a3278aa046b1b8fe92e58888b354c6aa1c98e71265a3bc3e5.webp)](samples/d70d807e7512117a3278aa046b1b8fe92e58888b354c6aa1c98e71265a3bc3e5.webp) |
| 4 | Background-only scene and lighting projection, 0.45 floor | early 1.48438 | 27.24 | 35.42 | 49.3 | [![Eight edits](samples/74fbddfc796d039e19342e3f9cf392a5164faaf305520968f93ca3239400cd37.webp)](samples/74fbddfc796d039e19342e3f9cf392a5164faaf305520968f93ca3239400cd37.webp) |
| 5 | Six highest-divergence timestep positions | early 1.25 | 26.34 | 33.48 | 48.3 | [![Eight edits](samples/ef3e62ea19eec9890e5d467c1614ed7c028e19bf3d69d3e1e677a91d13230905.webp)](samples/ef3e62ea19eec9890e5d467c1614ed7c028e19bf3d69d3e1e677a91d13230905.webp) |
| 6 | Paired scene and lighting contexts, 50% blend, 0.45 floor | constant 1.625 | 25.98 | 34.54 | 53.1 | [![Eight edits](samples/9e17143cf1967653b56d4dda12dda9a3b309df6adac07e8f47900ef991c56860.webp)](samples/9e17143cf1967653b56d4dda12dda9a3b309df6adac07e8f47900ef991c56860.webp) |
| 7 | Stronger cues at the first three teacher positions | early 1.3125 | 25.55 | 33.50 | 52.4 | [![Eight edits](samples/6a7211acae36d8895de03a0ba4756b507525594bb471a2c824895891518c653c.webp)](samples/6a7211acae36d8895de03a0ba4756b507525594bb471a2c824895891518c653c.webp) |
| 8 | Soft foreground weighting · prior incumbent recipe | early 1.25 | 24.98 | 31.12 | 49.2 | [![Eight edits](samples/a178d84adbde5a95f8d8ce8d1647812a7ea25c137ec2ed9721e2c958276c6af2.webp)](samples/a178d84adbde5a95f8d8ce8d1647812a7ea25c137ec2ed9721e2c958276c6af2.webp) |
| 9 | Soft foreground weighting plus 35% stronger Boss cues | early 1.0625 | 24.50 | 30.77 | 48.0 | [![Eight edits](samples/b0fc280276b939d08925646fc31d4f69e3be55b01ffe9d8909ed8daa32729b5b.webp)](samples/b0fc280276b939d08925646fc31d4f69e3be55b01ffe9d8909ed8daa32729b5b.webp) |
| 10 | Prompt-only control · all 240 fields | early 1.25 | 24.44 | 31.79 | 49.4 | [![Eight edits](samples/175065fdf76950edf16b15a66b1b9441d9e2e1496dee9c06358bbd3bbabce983.webp)](samples/175065fdf76950edf16b15a66b1b9441d9e2e1496dee9c06358bbd3bbabce983.webp) |
| 11 | Stronger Boss cues contrasted with matched ordinary attire | early 1.25 | 24.27 | 33.73 | 60.3 | [![Eight edits](samples/0f1f201e72abdd2456ddaf489ccbe068b48a561427fb4fc408faade3175e7c89.webp)](samples/0f1f201e72abdd2456ddaf489ccbe068b48a561427fb4fc408faade3175e7c89.webp) |
| 12 | Soft foreground weighting, 0.45 background floor | early 1.3125 | 23.48 | 28.63 | 39.7 | [![Eight edits](samples/8dfd53b332720dc08c4299ccbf003fd98e99ac114eb1dfd2e164241f224b8b09.webp)](samples/8dfd53b332720dc08c4299ccbf003fd98e99ac114eb1dfd2e164241f224b8b09.webp) |
| 13 | Paired scene and lighting contexts, 50% blend, 0.65 floor | early 1.4375 | 22.21 | 28.87 | 53.0 | [![Eight edits](samples/010f266bec310a8a5781a9917e8680ac891f48b5fef7a2875a336ace8e8f458a.webp)](samples/010f266bec310a8a5781a9917e8680ac891f48b5fef7a2875a336ace8e8f458a.webp) |
| 14 | Whole-image scene and lighting projection, 0.65 floor | early 1.625 | 21.58 | 28.04 | 53.7 | [![Eight edits](samples/dd80f8f951dc5353115008eb63514ecd3710b40558bfc4810d27fe917889a299.webp)](samples/dd80f8f951dc5353115008eb63514ecd3710b40558bfc4810d27fe917889a299.webp) |
| 15 | Soft foreground weighting with full-strength Boss cues | early 1.4375 | 21.26 | 27.28 | 51.0 | [![Eight edits](samples/252fc8bfac38b187eb98ff1af9b6d739c5da890f05a32161a40591928020cdee.webp)](samples/252fc8bfac38b187eb98ff1af9b6d739c5da890f05a32161a40591928020cdee.webp) |
| 16 | Soft foreground weighting plus a 35% blend of two averaged cues | constant 1.625 | 21.21 | 26.46 | 47.3 | [![Eight edits](samples/bc4bd531b2e431a621ee7476613ea4a3a4f9fd7b40a8bf8fdb9f504938460943.webp)](samples/bc4bd531b2e431a621ee7476613ea4a3a4f9fd7b40a8bf8fdb9f504938460943.webp) |
| 17 | Second qualified seed on the shared teacher scenes | early 1.25 | 21.18 | 27.20 | 47.7 | [![Eight edits](samples/ad74d8cd9fc2630325d431716e14289117fd6587faf26abeb4350213542dd3c3.webp)](samples/ad74d8cd9fc2630325d431716e14289117fd6587faf26abeb4350213542dd3c3.webp) |
| 18 | Percentile-filtered blend at the first three teacher positions | constant 1.48438 | 19.51 | 23.08 | 41.2 | [![Eight edits](samples/7b7a79c753eb1e10077d0144dba13fc38a2bc3dad97c8c3e4f66a69f33f11b8d.webp)](samples/7b7a79c753eb1e10077d0144dba13fc38a2bc3dad97c8c3e4f66a69f33f11b8d.webp) |
| 19 | Six evenly spaced timestep positions | early 1.40625 | 19.25 | 23.19 | 41.1 | [![Eight edits](samples/b4810c815c9689711fbaf8e1ff304ebf296341e843b93fdfdb0eb6be8528c6ef.webp)](samples/b4810c815c9689711fbaf8e1ff304ebf296341e843b93fdfdb0eb6be8528c6ef.webp) |
| 20 | Neutral trajectory only | early 1.625 | 10.79 | 12.87 | 51.7 | [![Eight edits](samples/a23ef775dd61debdf18c6d07dddf0b9b823f451df4b6c916d2fa46cd3ace6492.webp)](samples/a23ef775dd61debdf18c6d07dddf0b9b823f451df4b6c916d2fa46cd3ace6492.webp) |
| 21 | Train on adapter rollout states, 0.45 floor | early 0.666667 | 9.61 | 10.99 | 41.7 | [![Eight edits](samples/896b2ea3b205c59be9fecc0ebddaef58307740b4376b8dfbc4288dcb36a65996.webp)](samples/896b2ea3b205c59be9fecc0ebddaef58307740b4376b8dfbc4288dcb36a65996.webp) |
| 22 | In-scene costume cue blend, 35% | early 0.666667 | 6.54 | 7.27 | 42.1 | [![Eight edits](samples/6a354e7241d0c9d01cd3057d29939ae743a739896ce1249f94338555d0eab4d5.webp)](samples/6a354e7241d0c9d01cd3057d29939ae743a739896ce1249f94338555d0eab4d5.webp) |

## How the result depends on the characters

In a separate blind check Jev compared two four-character lineups as whole sets. It rated which lineup produced more individually recognizable Final Boss designs. Positive margins favor opening four over late cues. The two development partitions reuse the same eight images. The four search scenes are separate from those eight and were used during alpha selection.

| Matched characters | Opening-four preference margin |
| --- | ---: |
| Ranger, Smith, Stationer, Ferryman | +38.5 |
| Watchmaker, Guide, Guard, Violinist | -52.2 |
| Ranger, Stationer, Watchmaker, Guard | +5.5 |
| Smith, Ferryman, Guide, Violinist | -38.0 |
| Archivist, Bookkeeper, Botanist, Seamstress | -23.8 |

On the four search scenes, mean design repetition is 43.9/100 for opening four and 43.3/100 for late cues. The development advantage in variety depends on the characters.

Three preselected same-design examples from a separate historical confirmation set used different native renders of the same characters. Three different-design examples used the same art style. All six landed on the expected side of the scale in both image orders. These controls check whether the judge recognizes a repeat. Agreement with human rankings on new lineups remains unmeasured.

[Exact rubric, case pairs and lineups](design-range.json) · [Jev decision receipts](design-receipts.json.gz) · [Per-image rankings](README.md)

```sh
python -m release_tools.verify_final_boss_designs docs/final-boss-raw192/vibe12
```
