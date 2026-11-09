# Current finished-output learning-rate guidance

Use the existing **Candlelit C2 G=0.00075 RAW960** checkpoint for this tested
setup. Its mean intended-effect gain is **0.0038009807467460632**, the best of
three scored saved arms on four examined DEV cases. No further C3 comparison,
new LR search or fresh confirmation is required. Package defaults and the six
published showcase controls keep their original identities.

The selector is one number: intended-effect intensity ON minus matched OFF in
finished outputs, averaged equally across the declared cases. Higher wins.
Teacher-response SSE, preservation drift, generic beauty and completion are
not selectors or tiebreaks. Use RAW current weights for training, scoring,
export and inference; common session options explicitly set `ema_decay=None`.
Do not enable weight EMA or reinterpret historical EMA snapshots as RAW.

## Exact choice and scope

| Saved control | G | Mean intended-effect ON−OFF gain |
| --- | ---: | ---: |
| C0 | 0.0000029296875 | 0.0018655452877283096 |
| C1 | 0.000046875 | 0.0030342315634091697 |
| **C2** | **0.00075** | **0.0038009807467460632** |

These three RAW960 endpoints share batch8, declared horizon1600, fixed actual
D=0.018000000000000002 and bank LR=0.03. Finished images used native512,
10 steps, CFG1 and the original cached CPU-BF16 conditioning. All four examined
DEV cases retain equal weight. RAW960 is an endpoint within the original
1600-update schedule, not a shortened schedule or an instruction to train again.

This is best tested among three scored arms, not a completed four-arm
comparison, independently confirmed recipe, unseen holdout, universal optimum
or new product default. C3's separately retained damaged preview remains
separate, with no C3 effect score invented. Old seed42, Wild Hair and pilot
results also remain separate. Frozen-model scores are not human judgments or
statistical validation. The [portable metadata](finished-output-rate.json)
retains the exact options, per-case gains, candidate means and evidence hashes.

## Actual API option placement

The exact common option values are:

```json
{"lr": 0.00075, "d_lr_mult": 24.000000000000004, "prior_lr_mult": 40.0, "ema_decay": null}
```

D is `lr * d_lr_mult`; the bank rate is `lr * prior_lr_mult`. Preserve these
original multiplier literals rather than computing replacements from rounded
printed rates. Both multipliers belong to the shared recipe/checkpoint identity.

The [registered lifecycle factory](../lumen_studio/lifecycle.py) accepts these
keys in `SessionConfig.options`. For example, the configuration shape is:

```python
from particle_sliders.lifecycle import SessionConfig

options = dict(caller_owned_options)  # Verified model/cache/cohort settings.
options.update(lr=0.00075, d_lr_mult=24.000000000000004,
               prior_lr_mult=40.0, ema_decay=None, steps=1600, batch=8)
config = SessionConfig('anima', caller_declared_cohort, **options)
```

This only shows accepted option placement. The registered `anima` factory is
**CPU-only with rendering disabled**; it does not expose a generic native GPU
reproduction command. The separately admitted native
[`PG09Trainer`](../lumen_studio/pg09_training.py) accepts `lr`, `d_lr_mult` and
`prior_lr_mult` as constructor keywords, with `total_steps=1600` and
`batch_size=8` for this schedule. It does **not** accept an `ema_decay` keyword;
retain its RAW-only shared game without adding EMA. Runtime, verified cache,
numerical profile and native admission remain explicit caller dependencies;
see [the trainer API](../PG09_TRAINING.md) and [lifecycle limits](../LIFECYCLE.md).
This scalar mapping alone cannot reproduce the captured native cohort.

The original native source/model, conditioning/cache, RAW checkpoint and media
are not bundled here. The JSON evidence records are basename/byte-count/SHA256
references, not machine-private paths or downloadable artifacts. Use the saved
cohort with its original identities; do not relabel old packets or replay its
completed updates. A changed future recipe needs its own declared identity and
must not overwrite an existing run.

## Historical diagnostic screen

The earlier [PG0.9.5 native512 SSE study](../NATIVE-LR-PG095.md) used batch2,
horizon128 and a teacher-response objective. Its G=1e-5 options preset, source
pins, held-out score and reports stay unchanged as historical diagnostics.
They do not supersede the finished-output choice above or require another
search or confirmation.
