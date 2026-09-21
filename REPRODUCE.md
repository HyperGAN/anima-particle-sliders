# Reproduce the Anima release

The source includes the original training engine and native runtime integration, original prompt catalogs, complete train/development manifests, conversion checks, ordinary-LoRA fitting, and release assembly scripts. Weights, full-resolution PNGs, sample sidecars, training traces, normalization tensors and measurement reports are on [Hugging Face](https://huggingface.co/ntc-ai/anima-concept-sliders).

The routed-particle implementation and fitting algorithm come from the pinned
[shared core in sliders-conceptmod](https://github.com/mikkel/sliders-conceptmod/tree/main/packages/concept-slider-core).
`core.lock.json` records the exact revision and implementation hashes. The runtime
hashes the actual shared reference implementation, which is byte-identical to the
original release, so existing checkpoint identities remain valid. The original
CUDA environment lock stays intact; install the core separately as shown below.

## Environment and model

The release was run on an RTX A6000 using Python 3.12.13 and the complete pinned CUDA 12.6 environment below. Use a separate environment:

```bash
git clone https://github.com/mikkel/anima-concept-sliders.git
cd anima-concept-sliders
uv venv .venv-anima --python 3.12.13
uv pip sync --python .venv-anima/bin/python configs/anima/requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-anima/bin/python --no-deps -r requirements.txt
.venv-anima/bin/python release_tools/prepare_model.py --output artifacts/model
.venv-anima/bin/hf download ntc-ai/anima-concept-sliders --local-dir artifacts/release
```

Model conversion downloads the exact official checkpoint revision, tokenizers and Diffusers converter recorded in `configs/anima/model.lock.json`. It verifies the converted tensor hashes and restores the original conversion provenance before checking the complete portable identity. An incompatible conversion fails rather than silently loading the adapters into a different model.

## Render the particles

```bash
CUDA_VISIBLE_DEVICES=0 .venv-anima/bin/python release_tools/infer.py \
  --model artifacts/model \
  --adapter artifacts/release/weights/candlelit.safetensors \
  --prompt 'an adult woman wearing a blue coat, standing in a stone corridor, anime illustration' \
  --seed 29001 --strength 3 --output artifacts/candlelit.png
```

For an ordinary LoRA, pass `--format lora --adapter artifacts/release/distilled/native/candlelit.safetensors`. The sample sidecars record the exact gallery prompt, seed, dimensions, steps and checkpoint hashes. Strengths are direct scalars; zero exactly bypasses the native branches.

To replay the release's verified A6000 kernel policy, use its launcher instead of a plain Python launch:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-anima/bin/python scripts/anima_cuda_host.py \
  --gpu 0 --sm-count 82 --kernel-policy --script release_tools/infer.py -- \
  --model artifacts/model --adapter artifacts/release/weights/moonlit.safetensors \
  --prompt 'an adult man wearing a gray coat, standing beside a plain wall, anime illustration' \
  --seed 29001 --strength 5 --output artifacts/moonlit.png
```

This policy changes kernel-selection heuristics. Exact pixel replay is only claimed for the recorded software and hardware path; other GPUs can differ. The source does not weaken checkpoint identity checks.

## Rebuild training targets and train

The checked-in `data/train.json` and `data/dev.json` reproduce the archived manifests used by the released runs. The historical catalog also contains Theatrical definitions to retain those manifest hashes; this release publishes only Candlelit and Moonlit. Do not use the final-test characters for fitting or checkpoint selection.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-anima/bin/python scripts/anima_cuda_host.py \
  --gpu 0 --sm-count 82 --kernel-policy --script release_tools/train.py -- \
  candlelit --model artifacts/model --root artifacts/reproduction \
  --prepare-targets --until 1600
```

Repeat with `moonlit`. Target generation evaluates both frozen prompts at every timestep on both trajectories at 512 × 512. Training uses the frozen seed-7 recipe, microbatch one, effective batch eight, and no gradient checkpointing. It saves resumable state and immutable EMA exports. It does not select a perceptual winner or certify a newly trained checkpoint's image quality. Generation, training and preparation should run on a GPU you have assigned to this process.

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

ComfyUI integration can be checked in a separate environment with that checkout's dependencies:

```bash
CUDA_VISIBLE_DEVICES='' python release_tools/verify_comfy.py /path/to/ComfyUI artifacts/release
```

`validation/` records the release checks. `release-manifest.json` on Hugging Face gives each released file's size and SHA256. The release does not include optimizer-state binaries or the large regenerated target tensor caches; the source, catalogs, seeds, model pins and training recipe are provided to rebuild them.

## Rebuild the publication

With a downloaded release in `artifacts/release`, regenerate the card and preview grids without generating new images:

```bash
.venv-anima/bin/python release_tools/build.py --output artifacts/release
.venv-anima/bin/python release_tools/publish.py --folder artifacts/release
```

The publisher defaults to validation only and requires committed source. Its explicit `--publish` option creates the intended model repository, uploads a single release revision, and verifies all remote Git/LFS hashes plus three download readbacks. For an existing release, first inspect its current revision and pass `--expected-parent REVISION`; every existing weight, sample, preview and evidence artifact must remain byte-identical. Source, docs and the plugin package can then be updated atomically. Run `release_tools/verify_live.py --output artifacts/live-check` with Playwright and Chrome to check the deployed card's images, equations and mobile layout.
