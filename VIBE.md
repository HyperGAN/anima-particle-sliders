# Learn a vibe from successful image contexts

The opt-in `anima-vibe --vibe` path collects training data without editing the
catalog config files. A Nanoodle text workflow proposes matched neutral/positive
prompts. Frozen Anima renders N matched outputs per pair. An image-capable Jev
score workflow judges the finished images against your fixed criterion. Higher
mean positive-minus-neutral scores keep a prompt in the pool; weaker prompts are
replaced by new proposals informed by the saved results.

For a final-boss creator, start with a criterion such as:
`fantasy RPG final boss, imposing antagonist silhouette, epic battle scene`.
The workflow supplies concrete subjects and cues rather than using this same
sentence for every teacher. Each positive prompt keeps its neutral subject and
scene. This is adaptive curation, not a measured quality improvement yet.

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

Keep `NANOGPT_API_KEY` and `TYPESAFE_API_KEY` in the environment. Choose a current
NanoGPT text-generation model and a TypeSafe **Jev** model. The default
`--judge-provider typesafe` runs Nanoodle's Decide node through a transport to
TypeSafe's official `/v1/systemone` API, converting inline images into documented
asset placeholders. Each credential is sent only to its own provider. An
authenticated model-list request verified `jev-latest` and `jev-preview` during
implementation. `--judge-provider nanogpt` is available for an explicitly
verified NanoGPT decision-model route; no model ID is guessed or substituted.

```sh
anima-vibe \
  --vibe 'fantasy RPG final boss, imposing antagonist silhouette, epic battle scene' \
  --model-dir /path/to/pinned/anima-turbo \
  --prompt-model "$VIBE_PROMPT_MODEL" --judge-model jev-latest \
  --out runs/final-boss-vibe --device cuda:0 \
  --pool-size 8 --replace 4 --best-of 3 --rounds 4
```

Rendering and target preparation use CUDA only, with no model offload. Native
resource admission/leases remain the caller's responsibility, as for Anima's
existing native APIs. No GPU work is started by `--help` or manifest validation.

`--prompt-workflow FILE` and `--score-workflow FILE` replace the supplied
model-specific Nanoodle graphs. The proposal graph accepts `Context` and returns
`Pairs`, a JSON array of `{neutral, positive}` objects. The scoring graph accepts
`Criterion` and `Image` and exposes a Decide score node named `judge`. Its full
graph, model, rubric, native sampler and source identities bind the collection.
The default rubric measures visible intended-effect intensity, not beauty or
preservation. Nanoodle's one-based score is normalized to 0–1; its full typed
distribution is retained. Missing or invalid measurements fail explicitly.

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
requests reached both providers: NanoGPT rejected prompt generation with HTTP401,
and TypeSafe rejected the image judgment with HTTP402. These are credential/account
blockers, not successful model measurements. A live Anima + Nanoodle + Jev
collection and native target/training run remain unverified.
