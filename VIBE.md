# Learn a vibe from successful image contexts

The opt-in `anima-vibe --vibe` path collects training data without editing the
catalog config files. A Nanoodle text workflow proposes matched neutral/positive
prompts. Frozen Anima renders N matched outputs per pair. An image-capable NanoGPT
decision workflow judges the finished images against your fixed criterion. Higher
mean positive-minus-neutral scores keep a prompt in the pool; weaker prompts are
replaced by new proposals informed by the saved results.

For a final-boss creator, start with a criterion such as:
`fantasy RPG final boss, imposing antagonist silhouette, epic battle scene`.
The workflow supplies concrete subjects and cues rather than using this same
sentence for every teacher. Each positive prompt keeps its neutral subject and
scene. Curation scores do not by themselves establish a trained adapter's effect;
a completed scoped training result is recorded below.

## Source setup

This source feature's optional extra pins the new `particle-sliders-vibe` add-on
to a reviewed companion source commit, without requiring a PyPI release:

```sh
# Use the existing compatible Anima native environment; no base-model repin.
python -m pip install '/path/to/anima-particle-sliders[vibe]'
```

The add-on installs `particle_sliders_vibe` outside the original core's source
closure. It does not depend on a newer core or modify the pinned PG0.9.5 training
cohort. Latest Particle Sliders can also dispatch generic product workflows using
`particle-slider collect`; the standalone `particle-slider-vibe` command works
alongside older pinned cores.

For local development, install the companion
`/path/to/particle-sliders/packages/particle-sliders-vibe[nanoodle]` source first
and this product with `--no-deps -e`; the native environment's existing exact
core pin remains the source of truth.

Keep `NANOGPT_API_KEY` in the environment. Choose a current NanoGPT
text-generation model and an image-capable decision model. The default
`--judge-provider nanogpt` runs Nanoodle's Decide node through NanoGPT's
`/api/v1/decisions` endpoint. This path does not use a TypeSafe key or make
direct TypeSafe requests. No model ID is guessed or substituted.

NanoGPT lists decision models separately at `/api/v1/decision-models`; its
ordinary text-model list omits them. On November 17, 2026, that catalog listed
Liquid D1 as `liquid/d1`, with `image_input: false` and text-only state. D1 can
score text but cannot serve as this finished-image judge on the advertised
NanoGPT route. It also returns decisions rather than generating prompt text.
The current image-capable Jev lookup is `typesafe/jev-1.14`, through NanoGPT.
The `typesafe/jev-latest` alias currently advertises text-only input. Check the
live catalog's capabilities before changing the explicit image judge.

```sh
anima-vibe \
  --vibe 'fantasy RPG final boss, imposing antagonist silhouette, epic battle scene' \
  --model-dir /path/to/pinned/anima-turbo \
  --prompt-model "$VIBE_PROMPT_MODEL" --judge-model typesafe/jev-1.14 \
  --out runs/final-boss-vibe --device cuda:0 \
  --pool-size 8 --replace 4 --best-of 3 --rounds 4
```

Rendering and target preparation use CUDA only, with no model offload. Native
resource admission/leases remain the caller's responsibility, as for Anima's
existing native APIs. No GPU work is started by `--help` or manifest validation.
`--resolution` must be a multiple of 64 within Anima's native 512–1536 range;
unsupported sizes are rejected before model identity reads or allocation.

`--prompt-workflow FILE` and `--score-workflow FILE` replace the supplied
model-specific Nanoodle graphs. The proposal graph accepts `Context` and returns
`Pairs`, a JSON array of `{neutral, positive}` objects. The scoring graph accepts
`Criterion` and `Image` and exposes a Decide score node named `judge`. Its full
graph, model, rubric, native sampler and source identities bind the collection.
The default rubric measures visible intended-effect intensity, not beauty or
preservation. Nanoodle's one-based score is normalized to 0–1; its full typed
distribution is retained. Missing or invalid measurements fail explicitly.
NanoGPT independently rounds its reported score and probabilities to two
decimals; the adapter allows only the resulting bounded rounding discrepancy
and keeps the reported score, original response and validation diagnostics.

Increase `--rounds` with the same arguments to extend collection. Completed
proposals, images and judgments are reused, including partial-round receipts.
Unresolved calls refuse automatic retry; reconcile the recorded request ID
before choosing `--retry-pending`. Nanoodle itself does not currently guarantee
server-side idempotency for these IDs. All prior outputs and losing prompts are
retained, and no existing training run is reset.

## Prepare targets and train

For each prompt, the manifest keeps the best context whose finished positive
image scores at least **0.75** and strictly above its matched neutral image.
`--min-score` explicitly changes the qualification threshold; no qualifying
contexts is a reported incomplete collection, not a synthetic target. Prompt
fitness remains the equal-case mean across all N matched cases. Qualification
does not rewrite that ranking. Held-out evaluation needs separate prompts/seeds;
curation scores do not prove generalization.

The command prints the immutable snapshot path:
`runs/final-boss-vibe/training/<manifest-sha256>/manifest.json`.
Append `--prepare-targets` to collect the corresponding frozen responses in the
same native process, or prepare them later in the pinned Anima environment:

```sh
anima-vibe-targets --manifest /path/to/manifest.json \
  --model-dir /path/to/pinned/anima-turbo \
  --device cuda:0 --out /path/to/new-vibe-target-cache
```

The manifest records the exact selected neutral/positive prompts and seed, the
full judgments, media hashes, criterion and source collection. Its independent
validator does not relax the original catalog's split or balance requirements.
Target preparation uses the existing `prepare_targets` implementation: both
frozen trajectories, both responses at every shared state, all ten positions,
and native model/sampler identity checks. Each new snapshot gets its own cache
identity. Existing target caches and all completed LR work stay unchanged.

Use the resulting `TargetCache` and manifest `variation` with Anima's admitted
PG0.9.5 trainer API, following [PG09_TRAINING.md](PG09_TRAINING.md) and its native
admission/initialization requirements. Collection is deliberately before that
training step. The CLI does not silently launch training or change the existing
trainer's parameters. Fresh training must use raw weights and `ema_decay=None`.

## Verification scope

Offline tests cover adaptive replacement, best-of-N selection, matched seeds,
strict score parsing, interruption/resume, media integrity, real Nanoodle graph
execution with injected HTTP, and admission of the resulting manifest by the
existing cache entrypoint. They use explicit render/judge fixtures. Actual live
requests now authenticate through NanoGPT after loading a fresh environment
key. `typesafe/jev-latest` returned a typed text decision, and
`typesafe/jev-1.14` returned a measured judgment for the native positive image.
The earlier HTTP401 responses came from a stale inherited environment key;
they do not establish a model failure. Direct TypeSafe is not used.

A bounded native probe rendered one manual neutral/final-boss pair at 512×512,
seed 741901, using the unchanged pinned Anima base with no attached adapter.
Both PNG hashes and dimensions were verified after exact process reap; the
owned PID and its NVML entry were absent. Measured peaks were 7044 MiB NVML and
4,552,081,408 bytes host RSS, within GPU 20 GiB/allocator 19 GiB/RSS 24 GiB caps and live
GPU 5 GiB/host 47 GiB floors. The pair remains explicitly unscored, and produced no
training updates. This original probe verifies the native render route.

A subsequent live run completed four adaptive rounds with a pool of two prompt
pairs, one replacement per round and two matched seeds per pair. NanoGPT
`openai/gpt-5-mini` proposed five distinct pairs over time; `typesafe/jev-1.14`
judged all 32 native finished images. All 36 cloud requests returned HTTP200.
The strongest prior prompt survived each replacement, and a completed resume
replayed zero calls. Eight contexts qualified at the unchanged 0.75 threshold,
and native target preparation wrote and verified eight shards / 160 records,
covering both frozen trajectories and all ten positions. Every PNG and target
shard hash was checked after exact exit0/reap and owned PID/NVML absence.

The live run preserved the exact core0.13/PG0.9.5 closure and frozen base, used
no adapters or EMA, and performed zero training updates. It ran under the same
20GiB GPU / 19GiB allocator / 24GiB RSS caps and live 5GiB GPU / 47GiB host
floors, with measured peaks of 7066MiB NVML and 3,163,910,144 bytes RSS.
This demonstrates working collection, adaptive retention, resume and qualified
target preparation; it does not establish trained-slider quality or held-out
generalization.

## Completed native training: final-boss vibe

A separately identified, captured **ParticleGAN0.10 EMPF** run reused that
completed collection and its160 native fields, then completed **256 actual
updates**, batch2/seed1, with RAW weights and `ema_decay=None`. The original
public PG0.9.5 trainer, defaults and saved cohorts retain their source identities.
This CLI still collects/prepares data; it does not silently launch that separate
training profile.

| RAW checkpoint | Mean intended-effect ON1 minus matched OFF0 |
| --- | ---: |
| 64 | 0.4675 |
| 128 | 0.48041666666666666 |
| 192 | **0.5033333333333334** |
| 256 | 0.45875 |

**RAW192 is the unique maximum** across the four declared quarters. Each value
equally averages the same eight matched training cases. All16 OFF/ON
catastrophic-only checks pass at every quarter. Beauty, preservation, teacher
SSE and confidence are not selectors. This is not unseen confirmation or a new
learning-rate optimum. The provisional reference rates were G0.00075,
D0.009000000000000001 and bank0.03; completed LR work was reused.

An independent archive-byte audit verifies all256 FULL hashes, embedded update
counts, journals, original initialization receipts and actual regularizer calls.
All4,033 RAW payloads match FULL at each quarter. Exact continuation through
resource-floor stops repeated no completed updates. A fresh original CUDA
backbone then loaded selected RAW192 through `ParticleAdapter.load_export` and
reproduced case0's native512 PNG exactly, with no training, new effect judgment
or official fresh initialization. The public reader file is byte-identical to
the captured reader used in this proof.

[Compact hashes, values and verification scope](docs/vibe-native256-verification.json).
Weights remain local; this source draft makes no public weight-release claim.
