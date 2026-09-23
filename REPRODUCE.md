# Reproduce the Anima release

The source includes the original training engine and native runtime integration, original prompt catalogs, complete train/development manifests, conversion checks, ordinary-LoRA fitting, and release assembly scripts. Weights, full-resolution PNGs, sample sidecars, training traces, normalization tensors and measurement reports are on [Hugging Face](https://huggingface.co/ntc-ai/anima-particle-sliders).

The routed-particle implementation and fitting algorithm come from the pinned
[particle-sliders-core](https://github.com/HyperGAN/particle-sliders/tree/a119ca1ecd3d5d6c437065839d22739b04f2f4d8/packages/particle-sliders-core).
`requirements.txt` installs that commit. `core.lock.json` records the pin and
does not hash the retired `concept-slider-core` tree. Training calls
`winning_formulation().require(...)` after overlaying the Anima generator
learning rate `2e-5`. Architecture is `gmix`; the overlay is provisional
`particle-gmix-1600-v2`.

The published weights used an absolute noise hold of 1.0. The shared stamp's
`noise_hold_ratio` is 1.3, so the hold is edit RMS times 1.3. New training
follows the stamp. Inference still loads the published Candlelit, Moonlit, and
other released adapters: their embedded runtime hash is the historical identity
in `configs/anima/released-identity.json`, accepted when the converted Anima
model matches. The original CUDA environment lock stays intact; install the
core separately as shown below.

## Environment and model

The release was run on an RTX A6000 using Python 3.12.13 and the complete pinned CUDA 12.6 environment below. Use a separate environment:

```bash
git clone https://github.com/HyperGAN/anima-particle-sliders.git
cd anima-particle-sliders
uv venv .venv-anima --python 3.12.13
uv pip sync --python .venv-anima/bin/python configs/anima/requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-anima/bin/python --no-deps -r requirements.txt
.venv-anima/bin/python release_tools/prepare_model.py --output artifacts/model
.venv-anima/bin/hf download ntc-ai/anima-particle-sliders --local-dir artifacts/release
```

Model conversion downloads the exact official checkpoint revision, tokenizers and Diffusers converter recorded in `configs/anima/model.lock.json`. It verifies the converted tensor hashes and restores the original conversion provenance before checking the complete portable identity. An incompatible conversion fails rather than silently loading the adapters into a different model.

## Render the particles

```bash
CUDA_VISIBLE_DEVICES=0 .venv-anima/bin/python release_tools/infer.py \
  --model artifacts/model \
  --adapter artifacts/release/weights/candlelit-unit-alpha.safetensors \
  --prompt 'an adult woman wearing a blue coat, standing in a stone corridor, anime illustration' \
  --seed 29001 --strength 1 --output artifacts/candlelit.png
```

For an ordinary LoRA, pass `--format lora --adapter artifacts/release/distilled/native/candlelit-unit-alpha.safetensors`. The sample sidecars record the exact gallery prompt, seed, dimensions, steps and checkpoint hashes. Strengths are direct scalars; zero exactly bypasses the native branches.

To replay the release's verified A6000 kernel policy, use its launcher instead of a plain Python launch:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-anima/bin/python scripts/anima_cuda_host.py \
  --gpu 0 --sm-count 82 --kernel-policy --script release_tools/infer.py -- \
  --model artifacts/model --adapter artifacts/release/weights/moonlit-unit-alpha.safetensors \
  --prompt 'an adult man wearing a gray coat, standing beside a plain wall, anime illustration' \
  --seed 29001 --strength 1 --output artifacts/moonlit.png
```

This policy changes kernel-selection heuristics. Exact pixel replay is only claimed for the recorded software and hardware path; other GPUs can differ. The source does not weaken checkpoint identity checks.

## Rebuild training targets and train

The checked-in `data/train.json` and `data/dev.json` reproduce the archived lighting manifests. The historical catalog also contains Theatrical definitions to retain those manifest hashes; Theatrical is not released. Bad Intent has separate `data/bad-intent-train.json` and `data/bad-intent-dev.json` manifests. Do not use the final-test characters for fitting or checkpoint selection.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-anima/bin/python scripts/anima_cuda_host.py \
  --gpu 0 --sm-count 82 --kernel-policy --script release_tools/train.py -- \
  candlelit --model artifacts/model --root artifacts/reproduction \
  --prepare-targets --until 1600
```

Repeat with `moonlit`. Target generation evaluates both frozen prompts at every timestep on both trajectories at 512 × 512. Training uses the frozen seed-7 recipe, microbatch one, effective batch eight, and no gradient checkpointing. It saves resumable state and immutable EMA exports. It does not select a perceptual winner or certify a newly trained checkpoint's image quality. Generation, training and preparation should run on a GPU you have assigned to this process.

For Bad Intent use `bad-intent` in the same training command. Files remain under
`targets/uncanny` and `runs/uncanny`, its original training identity. Its target
contract deliberately edits expression, pose, framing and atmosphere, without
an identity-preservation requirement. The archived manifests are hash-pinned and
their rows recompile exactly from the original psychological-horror compiler.
`lumen_studio/bad_intent.py` retains the original full-catalog provenance hash:
Studio's unrelated Theatrical definitions had changed by this run, while the
shared rows used for Bad Intent are identical to this repository's catalog.

The public `weights/bad-intent.safetensors` contains rank 8, alpha 8 and unchanged
step-1600 learned weights. The native inference command accepts both the original
particle format and this embedded-alpha format; use `--strength 1` as the starting
point. The ordinary LoRA uses `--format lora` and its `distilled/native` file.

Training source/recipe hashes and original target-cache indexes accompany the published checkpoints. The original runs reused targets from an earlier runtime with a verified single-image-equivalence certificate; fresh targets use the current equivalent runtime. Rebuilt cache/provenance fingerprints can therefore differ from the original archival run. Numerical determinism depends on the pinned device and kernel policy as well as the seed.

## Distill and verify

The distiller reads prepared `model/`, `targets/`, and `runs/` directories under one root. Place or link the verified converted model at `artifacts/reproduction/model`; use each trained `ema-001600.safetensors`, or copy the published particle file into that location to distill the released teacher exactly.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-anima/bin/python scripts/anima_cuda_host.py \
  --gpu 0 --sm-count 82 --kernel-policy --script release_tools/distill.py -- \
  --root artifacts/reproduction --output artifacts/distilled \
  --samples data/release-samples.json
CUDA_VISIBLE_DEVICES='' .venv-anima/bin/pytest tests/lumen_studio tests/test_shared_core.py tests/test_release_lora.py -q
node --check lumen_studio/static/studio.js
```

`data/release-samples.json` fixes the development sample requests before fitting. The distiller uses training activations for regression and development activations for reporting, then renders matched ordinary-LoRA comparisons. See [DISTILLATION.md](DISTILLATION.md) for the exact ridge objective and measurements.

After preparing Bad Intent's own caches, reproduce its distill using the exact
public teacher and the separate two-case development sample manifest:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-anima/bin/python scripts/anima_cuda_host.py \
  --gpu 0 --sm-count 82 --kernel-policy --script release_tools/distill.py -- \
  --root artifacts/reproduction --output artifacts/bad-intent/distilled \
  --variation uncanny --export-name bad-intent \
  --teacher artifacts/release/weights/bad-intent.safetensors \
  --samples data/bad-intent-samples.json --strengths 0.5 1 \
  --particle-samples-output artifacts/bad-intent/samples
```

ComfyUI integration can be checked in a separate environment with that checkout's dependencies:

```bash
CUDA_VISIBLE_DEVICES='' python release_tools/verify_comfy.py /path/to/ComfyUI artifacts/release
```

`validation/` records the release checks. `release-manifest.json` on Hugging Face gives each released file's size and SHA256. The release does not include optimizer-state binaries or the large regenerated target tensor caches; the source, catalogs, seeds, model pins and training recipe are provided to rebuild them.

## Rebuild the publication

Final Form, Afterimage and Dusk use the same original training recipe. Their
exact train/development manifests are `data/{final-form,afterimage,dusk}-{train,dev}.json`;
the training command accepts those public names and verifies the archived
manifest hashes. Dusk is a 200-update pilot; the expressive campaigns ran to
1600. `data/expansion-selection.json` records the checkpoint and alpha chosen
for each release. No untouched final-test characters enter this selection.

The additional calibration and distillation commands, through the same pinned
host launcher and exclusive GPU lease, are:

```bash
python release_tools/prepare_expansion.py particles --studio /path/to/studio/artifacts/anima/remote --output artifacts/expansion-v1
python release_tools/prepare_expansion.py fit --studio /path/to/studio/artifacts/anima/remote --output artifacts/expansion-v1
python release_tools/prepare_expansion.py loras --studio /path/to/studio/artifacts/anima/remote --output artifacts/expansion-v1
python release_tools/prepare_expansion.py final --studio /path/to/studio/artifacts/anima/remote --output artifacts/expansion-v1
python release_tools/stage_expansion.py --studio /path/to/studio/artifacts/anima/remote --results artifacts/expansion-v1 --folder artifacts/release
```

The first sweep compares the checked-in checkpoint/alpha candidates; `fit`
uses only training activations for the fixed-ridge solve. The LoRA sweep
measures complete-denoiser edits on 40 shared development states, plus matched
renders. Final rendering reads the exact calibrated files. Staging verifies
unchanged learned tensors and pixel-exact replay against the selected sweep
images. The published `evidence/{final-form,afterimage,dusk}/` directories
include sweeps, fits, selected values, training traces, and original compiler
and definition sources. Use the gallery build/verify commands below afterward.

The current release embeds the selected Studio particle alphas and uses
strength 1.0 for all new examples. To stage the calibrated derivatives, render
all current matched images, build the gallery and verify its provenance:

```bash
python release_tools/calibrated_release.py stage --folder artifacts/release --studio /path/to/studio/artifacts/anima/remote
# Run render through the pinned host launcher with the exclusive GPU lease.
ANIMA_GPU_LEASE="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" \
python scripts/anima_cuda_host.py --gpu 0 --sm-count 82 --kernel-policy \
  --script release_tools/calibrated_release.py -- render \
  --folder artifacts/release --studio /path/to/studio/artifacts/anima/remote
python release_tools/calibrated_release.py build --folder artifacts/release --studio /path/to/studio/artifacts/anima/remote
python release_tools/calibrated_release.py verify --folder artifacts/release --studio /path/to/studio/artifacts/anima/remote
```

Use an available GPU; the exclusive lease rejects another Anima coordinator.
The staging step reads `exports/alpha-unit-v1/exports.json`, verifies the exact
selected particle hashes and unchanged learned tensors, and creates new named
LoRAs by changing alpha. It also applies the checked-in
`data/bad-intent-balance.json`: derive the current Bad Intent particle at alpha
12 from its original step-1600 EMA, retain LoRA alpha 24, and use separate
`balanced-alpha` sample paths. The earlier `data/featured-refresh.json` records
the archived balcony scene under `featured-v2` paths. The subsequent
`data/example-refreshes.json` records the Bad Intent portrait under `featured-v3`
and the current Afterimage exposure studies under `featured-v4`. Staging applies those
replacements; rendering uses each recorded prompt and seed with the exact
public adapters at Particle 1 / Distill 1 / Off 0. Bad Intent uses a neutral
development-manifest prompt. Afterimage uses an authored exposure-study prompt with the
same development characters, so the records explicitly distinguish those
prompts from the archived neutral training-evaluation prompts. No final-test
characters are used. The Afterimage prompt asks for soft long-exposure ghosting
in all three formats; the two selected Off seeds do not show the offset face
and hand repetitions visible with the adapters. `data/afterimage-echo-audit.json`
records the selection and rejected alternatives. Previous comparisons remain in the catalog's
`example_archive`, and all earlier files remain at their original paths.
No training or regression is repeated. The original release
assembly instructions below describe the preceding, uncalibrated publication.

For the additive Bad Intent release, start with a downloaded release and the
distillation results above. Assemble the new card, comparisons, evidence and
plugin without replacing previous weights, samples or featured portraits:

```bash
.venv-anima/bin/python release_tools/add_bad_intent.py \
  --folder artifacts/release --results artifacts/bad-intent \
  --root artifacts/reproduction \
  --teacher artifacts/release/weights/bad-intent.safetensors
.venv-anima/bin/python release_tools/publish.py --folder artifacts/release
```

The publisher defaults to validation only and requires committed source. Its explicit `--publish` option creates the intended model repository, uploads a single release revision, and verifies all remote Git/LFS hashes plus three download readbacks. For an existing release, first inspect its current revision and pass `--expected-parent REVISION`; every existing weight, sample, preview and evidence artifact must remain byte-identical. Source, docs and the plugin package can then be updated atomically. Run `release_tools/verify_live.py --output artifacts/live-check` with Playwright and Chrome to check the deployed card's images, equations and mobile layout.
