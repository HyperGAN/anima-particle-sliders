# Learn a vibe from successful image contexts

To train a slider end to end on ParticleGAN 0.12.0, run `anima-vibe-train`. [Unattended native training](#unattended-native-training).

The opt-in `anima-vibe --vibe` path collects training data using frozen
Anima and an image-capable Jev judge. Each round retains successful scenes
and explores new ones with a Nanoodle prompt workflow.

## Worked example for other products

The [shared `--vibe` instructions](https://github.com/HyperGAN/particle-sliders/blob/main/docs/vibe.md) give the template for every `*-particle-sliders` product. This historical example records the released timed Final Boss. Fresh runs use the [constant-strength selection protocol](#rate-and-checkpoint-selection).

1. **Select the teachers.** Run `anima-vibe --vibe 'final boss'` with the [native collection arguments](#source-setup). Render each ordinary character and its Boss treatments in the same scene at a matched seed. Jev requires an ordinary Boss rating ≤ 0.35 and a positive rating ≥ 0.75. Three collections supplied 14 qualified pairs. Comparing their designs removed two lookalikes, leaving 12 teaching scenes. [Selection protocol](#named-portfolios)
2. **Build the late-cues targets.** Capture both frozen trajectories for each scene. At each of ten positions, ask how the Boss prompt changes the predicted denoising direction at that same state. This gives 240 fields. At positions 1–7 use the Boss-minus-neutral edit. At positions 8–10 mix 65% of that edit with 35% from the more explicit Boss prompt. Weight the subject at 1.0 and the background at 0.45 with a soft mask, then restore each edit’s original norm. [Exact cue prompts and parameters](docs/final-boss-raw192/vibe12/training/late-cues.json) · [Field construction](research/final-boss-vibe12/anima-final-boss-vibe12-20261124/adapters/spatial_stock.py)
3. **Train the adapter.** Keep Anima frozen and train the native adapter with stock ParticleGAN for 192 updates, batch size 2, learning rate 0.00075, seed 1 and `ema_decay=None`. Compare recipes at RAW192 using the shared prompts, seeds and native Euler sampler. To reproduce this run, restore its hash-pinned target cache. For a new collection, save its own manifest and cache. [Restore the winning targets](docs/final-boss-raw192/README.md#use-the-winning-training-targets) · [Native training and campaign entry points](research/final-boss-vibe12/README.md)
4. **Select alpha on the saved RAW.** Use the working Boss score to steer the search and rank the final settings. It asks whether the slider, switched on, gives a final boss that is not broken: the effect stops counting once the character reads as a final boss, a broken drawing or a missing figure scores nothing, and repeated designs across characters are discounted. [What ON should mean](docs/post-training/README.md#what-on-should-mean) The recorded run below used the earlier distinct Boss score, which rewarded Boss strength and gain over OFF and penalized accidental damage. The search tried constant, early and late schedules on four characters, starting at 1.0 with a 0.25 step. Alpha has no upper limit: while the best probe is also the strongest one tried, the search measures the next stronger step. The recorded run below predates that rule and stopped at α 1.75, so its constant and late schedules were not measured above it. After 27 probes, it scored five settings on eight other characters. Early α 1.0 scored 26.39, early α 1.25 scored 28.27, and **early α 1.15625 won at 28.84/100**. The late teacher cue and the early inference schedule are two independently chosen settings. [Measured probes and case ratings](docs/final-boss-raw192/vibe12/alpha-tuning/late-cues/search.json) · [Run or verify the alpha search](docs/final-boss-raw192/vibe12/alpha-tuning/README.md)
5. **Release the first-ranked recipe.** Bind its RAW hash, target cache, alpha and exact interval gains in the default manifest. Show matched OFF and ON images beside each entry. The normal replay command reproduced the scored Ranger image byte for byte. [Leaderboard with samples](docs/final-boss-raw192/vibe12/distinct-designs.md) · [Current default](docs/final-boss-raw192/default.json) · [Replay the default](docs/final-boss-raw192/README.md#replay-the-default)

## Named portfolios

Use `--vibe 'final boss'` or `--vibe '1990s anime'` with a new output directory
to run the shared finish-aware portfolio search. The prompt model proposes
scenes. Anima renders a neutral and several prompt treatments at matched
seeds. The Boss profile varies costume and silhouette. The anime profile
varies whole-frame cel treatment and proposes environments, objects,
animals and people in separate slots. It uses the calibrated cel-animation
rubric and asks which distinctive cel treatment dominates the visible frame.

Jev measures intended effect and multiplies it by one minus the larger of the
visible overbake and broken-layout scores before matched neutral subtraction.
The selector ranks by distinct finish gain. It compares each proposed portfolio under a
second Jev effect reading and keeps a replacement only when that reading
holds up. It caps each confident family and the unclassified bucket at two.
It checks lookalikes in both image orders and counts pairs averaging 0.65
or higher as near-duplicates.

Distinct finish gain sums the strongest mutually distinct set, includes
negative gains from the selected portfolio, and divides by the original
number of scenes. Missing slots remain in that denominator. Catastrophic
integrity is checked separately.

`portfolio.json` preserves measurements and proposal history. The
`teacher-selection` directory records the improved portfolios, second reads,
lookalike audits and the recipe that won each comparison. The CLI exports the
highest-distinct-finish measured round that contains a qualified teacher.
When that round still has gaps, one text-model call proposes scene-preserving
rewrites. Anima tests the rewrite, its named-vibe form and a cue-first
prompt at the existing matched seed, then reselects from the expanded pool.

The default is one adaptive round. Pass `--rounds 2` to feed prior scores into
a second proposal. Each round keeps stronger scenes and replaces up to
`--replace` weaker scenes. The qualified manifest keeps the same positive-gain
and raw effect threshold. Both profiles render native prompt variants so the
existing cache builder can replay the exact selected prompts and seeds.

Final Boss proposals start from ordinary adults in everyday clothing. The positive
prompt keeps the same person, setting, pose, framing, and lighting, and changes the
character's costume and silhouette. Jev's raw Boss rating for the neutral image
must be **0.35 or lower**. At the default **0.75** target threshold, this gives a
raw Boss gain of at least **0.40**. Subsequent rounds replace scenes whose neutral
renders all miss that ceiling. The manifest records the ceiling for target-cache
validation. Existing manifests retain their recorded qualification rules.

On two fresh panels of eight automatically proposed scenes each, the new
selector and rewrite pass improved distinct finish and the second effect
read. Scores below are on a 100-point scale.

| Native CLI panel | Previous distinct finish | Current distinct finish | Qualified teachers |
| --- | ---: | ---: | ---: |
| Final Boss | +11.5 | +16.3 | 3 → 5 |
| 1990s anime | +12.5 | +22.7 | 1 → 1 |

The four-round anime run exported round three, with distinct finish improving
from +21.9 to +32.3 and qualified teachers increasing from three to four.
These comparisons measure teacher selection on finished images.

Free-text vibes, explicit `--score-workflow` overrides, and saved
`collection.json` runs use the adaptive prompt-pair workflow. Each proposed
positive retains its neutral subject and scene. A collection remains bound to
the source and judge graph that started it.

## Source setup

This source feature's optional extra pins the new `particle-sliders-vibe` add-on
to a reviewed companion source commit, without requiring a PyPI release:

```sh
# Use the existing compatible Anima native environment; no base-model repin.
python -m pip install '/path/to/anima-particle-sliders[vibe]'
```

This checkout uses the runtime pins described in [LIFECYCLE.md](LIFECYCLE.md).
The `particle_sliders_vibe` add-on
has its own source identity and also works with the archived core. This
checkout exposes collection through `anima-vibe`. The shared core's generic
`particle-slider collect` command requires a core checkout containing the
[companion change](https://github.com/HyperGAN/particle-sliders/pull/208).

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
  --vibe 'final boss' \
  --model-dir /path/to/pinned/anima-turbo \
  --prompt-model "$VIBE_PROMPT_MODEL" --judge-model typesafe/jev-1.14 \
  --out runs/final-boss-vibe --device cuda:0 \
  --pool-size 8 --replace 4 --best-of 3 --rounds 4
```

Use `python -m lumen_studio.vibe_cli` to run the same entrypoint from a source
checkout. Rendering and target preparation use CUDA only, with no model offload. Native
resource admission/leases remain the caller's responsibility, as for Anima's
existing native APIs. No GPU work is started by `--help` or manifest validation.
`--resolution` must be a multiple of 64 within Anima's native 512–1536 range;
unsupported sizes are rejected before model identity reads or allocation.

`--prompt-workflow FILE` and `--score-workflow FILE` replace the supplied
model-specific Nanoodle graphs. The proposal graph accepts `Context` and returns
`Pairs`, a JSON array of `{neutral, positive}` objects. The scoring graph accepts
`Criterion` and `Image` and exposes a Decide score node named `judge`. Its full
graph, model, rubric, native sampler and source identities bind the collection.
The default free-text question is “How well does the candidate convey ‘{concept}’ without
breaking the image?” The `--vibe` value supplies the concept. Four score levels
range from absent effect or a broken image to an unmistakable effect in an
intact image. Nanoodle's one-based score is normalized to 0–1 and its full typed
distribution is retained. Missing or invalid measurements fail explicitly.
NanoGPT independently rounds its reported score and probabilities to two
decimals; the adapter allows only the resulting bounded rounding discrepancy
and keeps the reported score, original response and validation diagnostics.

Increase `--rounds` with the same arguments to extend collection. Completed
proposals, images and judgments are reused, including partial-round receipts.
All prior outputs and losing prompts are retained, and no existing training
run is reset.

### Resuming an interrupted named collection

Every render, proposal and judgment is one saved call under `calls/`. A call
is `pending` while it runs, then `done` or `failed`. Rerunning the same
command does the following.

| Saved call | What the next run does |
| --- | --- |
| `done` | Reuses the result after checking the image hash. |
| `failed` | Repeats it. The provider answered with a refusal, a server error or an unusable reply, so no result exists. An unusable reply is kept in the record. |
| `pending` render | Repeats it. Rendering is local and deterministic. |
| `pending` proposal or judgment | Stops with `unresolved call <request ID>`. The request may have been processed. Reconcile that request ID with the provider, then rerun with `--retry-pending`. Nanoodle does not currently guarantee server-side idempotency for these IDs. |

The CLI checks `NANOGPT_API_KEY` (and `TYPESAFE_API_KEY` for
`--judge-provider typesafe`) and the pool arguments before it reads the model,
so a missing key leaves no open calls.

Teacher selection reports its own state in the printed summary. When the
second effect read or the rewrite pass cannot be measured, the round keeps its
audited collection-time portfolio and the summary says
`"teacher_selection": "incomplete"` with the cause, any unresolved request
IDs and the next step. An incomplete result is saved as
`round-N.temporary.json`, is recomputed on the next run, and never replaces a
round that finished its checks. A manifest exported from an incomplete
selection is a valid audited portfolio. Rerun until the summary says
`"complete"` before comparing teachers across selectors.

`--prepare-targets` leaves a finished, intact target cache untouched, so its
`index.json` hash stays the one a release pins.

Free-text collection uses the shared add-on's ledger, where every interrupted
call, including a local render, waits for `--retry-pending`.

## Prepare frozen targets

For free-text collection, the manifest keeps the best context per prompt whose finished positive
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

Collection and target preparation finish at the cache. For fresh training in
the current checkout, [see the native admission and training API](LIFECYCLE.md).
Validate target provenance and use RAW weights with `ema_decay=None`. The
historical CUDA proof below applies to the original runtime and cache identity.

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
PG0.9.5 dependencies and saved checkpoints belong to the
[original source checkout](https://github.com/HyperGAN/anima-particle-sliders/tree/cf5a4d7679ab98fcb6896a08a76fd2baae332190).

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

The three rates belong to the optimizer. Compare rate candidates on the same frozen
teacher fields, with the same training updates, seed, matched scenes and constant
unit slider strength. For Final Boss, use the working Boss score, which counts a broken
drawing as no effect and discounts repeated designs. Record structural failures separately. Once training
has selected the rates and RAW checkpoint, tune the single rendering alpha.
The native gain limit of 5 applies only when rendering an image.

An independent archive-byte audit verifies all256 FULL hashes, embedded update
counts, journals, original initialization receipts and actual regularizer calls.
All4,033 RAW payloads match FULL at each quarter. Exact continuation through
resource-floor stops repeated no completed updates. A fresh original CUDA
backbone then loaded selected RAW192 through `ParticleAdapter.load_export` and
reproduced case0's native512 PNG exactly, with no training, new effect judgment
or official fresh initialization. The public reader file is byte-identical to
the captured reader used in this proof.

[Compact hashes, values and verification scope](docs/vibe-native256-verification.json).
The audited weights remain local.

## Unattended native training

The `anima-vibe-train` command collects matched teacher images, prepares frozen
targets, trains a stock EMPF adapter and confirms a saved RAW on held-out
prompts. The bundled scene file contains Final Boss, 1990s anime and Visually
appealing character. Subjects are specified as adults in the prompts. Use the
pinned native Anima environment and set `NANOGPT_API_KEY` for Jev. On a shared
GPU, `--gpu-cap-gib` limits this process’s PyTorch allocations (19 GiB by default).

```sh
anima-vibe-train --vibe 'Final Boss' --model-dir /models/anima-turbo-v11 \
  --out runs/boss --device cuda:0
```

At startup three pinned PNGs check the image judge on a coherent portrait, a
broken drawing and a split frame. The script checks both images in each teacher
pair. It tries another matched seed for rejected scenes, with eight attempts per
scene by default. After five unsuccessful seeds it also reframes the neutral
and its treatments as a continuous camera shot. Increase `--collection-attempts`
to extend an incomplete collection using its saved images and decisions.
Set `--min-teachers` to choose the minimum number of distinct
qualifying scenes. A missing teacher score makes that image ineligible. A missing
validation score leaves the matched mean unmeasured, and the script still acts
on confirmed structural defects. Decisions and rejected images are saved with
the run.

For a bundled concept with a confirmed recipe, the ordinary command uses its
measured G, D and particle rates and its selected update count first. Fresh Final Boss fits start at the released RAW192 rates and 192-update schedule. The failed RAW128 recipe remains available in the registry for replay. Other shorter
runs keep the 256-update optimizer schedule used in qualification. Set
`--steps 256` to compare the four original checkpoints again. If the default
fit fails during training selection,
the ordinary command searches alternative rates automatically. Explicit rates
request a fixed fit unless `--search-rates` is supplied. A resumed run keeps the rates recorded with
its existing checkpoints. `--g-lr`, `--d-lr` and `--particle-lr` override
the corresponding rate. [Measured native recipes](configs/anima/study-recipes.json).

### Original measured settings on ParticleGAN 0.12.0

The default Final Boss leaderboard calibrates one constant alpha for each trained recipe and ranks the reserved selections on eight other characters. Both stages use the working Boss score. The board linked here was measured with the earlier distinct Boss score, including visible damage and repeated character designs. [Rankings and matched samples](docs/leaderboard.md)

Each gain is the equally weighted mean change in the intended effect with the slider
on versus off across four held-out prompts and seeds. All twelve pairs improved.
These settings apply to Anima Turbo v1.1 at 512 × 512 with ten Euler steps and CFG 1.

| Slider | G | D | Particle | Updates | Strength | Held-out gain |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Final Boss · failed visual trial | 0.00028125 | 0.009 | 0.03 | 128 | constant, α 1 | +0.1042 |
| 1990s anime | 0.000375 | 0.018 | 0.03 | 192 | constant, α 1.5 | +0.2050 |
| Visually appealing character | 0.000375 | 0.018 | 0.03 | 256 | constant, α 1 | +0.1792 |

The 1990s anime and character-appeal runs compared 64, 128, 192 and 256 updates.
The released Final Boss is Vibe12 late-cues RAW192 with early α 1.15625. In a replay on the four RAW128 confirmation prompts with byte-identical OFF PNGs, that checkpoint gained +0.9442 versus +0.1042 for the constant-strength RAW128 study. The integrity observer passed two reference images, left one unmeasured and flagged one. Both effect scores and structural judgments are recorded separately. The trainer writes the per-case comparison to `reference-comparison.json` when the rubric, prompts, seeds, OFF images and scores match the pinned benchmark. It reports other panels as unmatched. For all three bundled sliders a full run requires an intact confirmation and a gain over its own pinned reference before returning exit 0. A research-only caller can use `--no-require-reference-gain` to accept a completed measurement regardless of its release decision. Exit 3 keeps a measured candidate and its gallery on disk when it falls short or lacks a matched reference. An incomplete or structurally unresolved confirmation exits 2. [Released checkpoint and settings](docs/final-boss-raw192/default.json) · [Pinned comparison](configs/anima/final-boss-reference.json)

Fresh anime teachers must strengthen the cel treatment of the frame. Fresh character appeal teachers must strengthen the central character’s design. The paired judge receives the actual neutral image first and the proposed target second. A failed or uncertain pair returns the collector to the other candidates or next seed. The saved pair decision binds both PNG hashes. The script keeps subject age in the generation prompts.

The 1990s anime reference is RAW192, constant α 1.5. For visually appealing character it is RAW256, constant α 1. Both started with the same three-strength grid as the historical Final Boss trial. Fresh runs freeze the recipe, rates and RAW after the training comparison. Alpha selection measures a single scalar applied at every native denoising step and reserves it before confirmation. The run compares that setting with the slider’s pinned reference on matched confirmation images. [Anime reference](configs/anima/1990s-anime-reference.json) · [Character appeal reference](configs/anima/character-appeal-reference.json)

The original RAW128 search tried α 0.5, 1 and 1.5 on each schedule. Early α 1.0 had a selection gain of +0.342 but two screening images had uncertain integrity. Constant α 1.5 gained +0.363 with one uncertain selection image. A recorded HTTP 502 left one early α 1.5 effect missing. Its saved PNG later scored 0.89, giving that setting a +0.8117 selection gain with three uncertain structural checks. The study judge now retries explicit transient HTTP responses twice and keeps a separate receipt for each attempt. An interrupted request with an unknown outcome stays unmeasured. Constant α 1.0 was the strongest fully measured result on the original grid. The earlier mixed-schedule search and its selected strengths remain available for replay.

The original study ranked teacher targets one image at a time. It accepted a tea-shop image in which an armored figure stands beside the original subject. New Final Boss collections compare the two images before freezing the field. A separate boss or an uncertain pair triggers the next eligible candidate or seed. The original nine-scene experiment used unweighted whole-image fields, G 0.00028125 and 128 updates on a 256-update optimizer schedule. Vibe12 used G 0.00075 and 192 updates with a 192-update schedule. On the four matched prompts, swapping the failed RAW128 from constant 1.0 to the released early α 1.15625 raised its mean gain from +0.1042 to +0.5442. Running the released RAW192 at the same two strengths yielded +0.6900 and +0.9442. Those comparisons isolate the effect of the inference schedule on each checkpoint. The released twelve-scene training recipe also applied a foreground weight and additional late teacher cues. [Field construction](#worked-example-for-other-products)

To test the late-cues method on the eight corrected bundled teaching pairs, pass `--boss-targets late-cues` when starting a fresh Final Boss run. The script verifies each mask against the exact neutral and target PNGs, weights the teacher edit by 1.0 on the subject and 0.45 on the background, and blends 35% of a stronger cue at the last three positions. It stores the transformed cache separately and checks native predictions against a saved teaching field for each scene. A custom recipe file can supply different matched images, masks and explicit prompts. The original experiment and restored reference each keep their own frozen fields.

The two 128-update studies on the same eight corrected pairs each used G 0.00028125. Whole-image fields selected RAW96 at constant α 1.6875. Its comparison gain was +0.2275 with one unresolved structure check. Foreground-weighted late-cue fields selected RAW128 at constant α 1.4375. One structure check failed and one effect request remained unmeasured. A third study kept the eight weighted targets and tried G 0.00075 with the released 192-update horizon. A portrait collapsed at update 48 and the script stopped before update 49. These specific fits remain failed or unmeasured. The published training rates are a starting point for fresh data.

On the 0.12 runtime, the original Vibe12 RAW192 reproduced the archived Ranger and four matched comparison PNGs byte for byte using its ten published gains. A separate 48-update training control matched all 240 of the original teacher fields, all 4,033 initial adapter tensors and the minibatch order at every update. At early α 1.15625 the 0.11.1 and 0.12 RAW48 checkpoints gained +0.255 and +0.313 on the same four scenes. Both had structural failures. [Verified inference replay](docs/final-boss-raw192/vibe12/pg012-replay.json)

The original Final Boss rate search found six settings that were intact at 64
and later failed a structural check. The failed RAW128 fit used nine fresh
matched teacher scenes and compared 32, 64, 96 and 128 updates. Its screening
panel included the exact watch-shop scene that had exposed duplicated characters.
The upper endpoints are the best tested choices for those runs.

### Rate and checkpoint selection

Use `--search-rates` to compare generator, critic and particle learning rates at the
same training budget and constant α 1. The first screen tests the reference plus a
halving and doubling on each axis. The search repeats around a strictly better
measured setting, then refines in factors of √2. If an entire initial screen
fails, it can try a smaller generator rate. The default budget is 31 distinct
rate settings. Set `--rate-arm-budget` to change it. The coverage record lists any
untested or unmeasured neighbors, including an improving edge left open when
the budget runs out. Exact ties keep their observation order. The trainer
continues from the strongest qualified saved FULL. It rechecks the original rate scenes at every later quarter.
A confirmed split or other catastrophic defect stops the candidate before the
next update. A checkpoint with an unresolved recheck is ineligible for selection.
The full-budget check also applies when resuming a screened run from an older version.
Fresh Final Boss fits include the exact watch-repair-shop prompt and seed that
revealed a duplicated figure in training. Both fixed-rate and searched fits
recheck those screening scenes. Previously frozen panels keep their original cases. It also checks
the proposed replay strength on those scenes before using the held-out panel. If the bracket
yields no usable fit, it tests lower G values
with the reference critic and particle rates held fixed. For a run
using measured rates, pass `--g-lr`, `--d-lr` and `--particle-lr`. Final Boss ranks by the working Boss score on matched finished images. The other sliders rank by mean intended-effect gain.
`--steps 256` compares updates 64, 128, 192 and 256. The training comparison renders each eligible RAW at constant α 1 on the same prompts and seeds and ranks by the registered slider objective. The script writes the selected rates and one RAW hash to `training-selection.json` before alpha calibration starts. Calibration starts at α 1, tests both sides, includes probes at 0.5, 1.5 and 2, and extends a rising edge up to 5. The strongest measured setting is refined to neighboring probes within 1/32. Missing objective measurements are ineligible. The same α is used for all ten steps. A failed calibration ends the run with its evidence. A retry uses the frozen training selection. Before rendering the first held-out image, the script verifies the metric, matched panel, RAW hash, ten equal gains and full scalar trace against `strengths-reservation.json`, then saves the selected RAW, rates and strength in `confirmation-selection.json`. If that
measurement is interrupted, a rerun checks the same selection. To reserve
the selection while keeping held-out ON images unseen, use `--select-only`.
Run the same command without that flag to confirm it.
`outcome.json` records the resolved interval gains, cases and
original RAW path. `status.json` records the result and an incomplete run's
exit reason. Rerun with the same directory and arguments to reuse recorded work.

Open `leaderboard.md` in the run directory to compare matched images. Render a new prompt with the recorded RAW and confirmed strength.

```sh
anima-vibe-render --outcome runs/boss/outcome.json --model-dir /models/anima-turbo-v11 \
  --prompt 'An adult cartographer in a sunlit observatory' --seed 13419 --out cartographer.png
```


To move a confirmed slider, copy its RAW safetensors file, its adjacent
`.scoped.json` and `.engine.json` sidecars, and `outcome.json` into one directory.
The render command verifies the local RAW hash against the recorded selection.
