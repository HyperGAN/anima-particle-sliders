# ComfyUI: Anima particles and ordinary LoRAs

Use **Anima Turbo v1.1** with its Qwen text encoder and VAE. These adapters are for the diffusion MODEL, not the text encoder.

## Native particle sliders

Use a current ComfyUI installation with Python 3.10 or newer. Install this repository as a custom node and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HyperGAN/anima-particle-sliders.git
python -m pip install -r anima-particle-sliders/requirements.txt
```

Use the Python executable belonging to your ComfyUI installation for the install command (including its embedded Python on portable installations). The plugin ZIP needs the same requirements install. This installs the pinned shared algorithm package from [sliders-conceptmod](https://github.com/mikkel/sliders-conceptmod/tree/main/packages/concept-slider-core).

Download [Bad Intent](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/weights/bad-intent-balanced-alpha.safetensors?download=true), [Candlelit](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/weights/candlelit-unit-alpha.safetensors?download=true) or [Moonlit](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/weights/moonlit-unit-alpha.safetensors?download=true) into `ComfyUI/models/loras/`. Update an existing plugin installation for the embedded-alpha format. Insert **Anima Particle Slider (ntc-ai)** between the diffusion model loader and your sampler:

```text
Load Diffusion Model (anima-turbo-v1.1)
  → Anima Particle Slider (adapter_name: bad-intent-balanced-alpha.safetensors, strength: 1)
  → existing Anima Turbo sampling workflow
```

Keep the official Turbo model-sampling settings. Start with 10 steps, CFG 1, Euler, 768 × 768. Strength 0 is an exact bypass; **1.0 is the calibrated default**. Higher strengths extrapolate the effect. Calibration is embedded in alpha (Bad Intent 12, Candlelit 16, Moonlit 23.08072421821097, all rank 8); do not add an external gain. No trigger word is needed. The sample prompt stays fixed while strength changes.

The old `bad-intent.safetensors` particle remains at alpha 8. Its strength 1.5
is equivalent to the current alpha-12 particle at strength 1. The Bad Intent
LoRA retains alpha 24; no LoRA replacement is needed for this balance update.

Final Form, Afterimage and Dusk use the same plugin. Download their current
`*-unit-alpha.safetensors` files from [the particle weights folder](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/weights)
and start at strength 1.0. Their separate ordinary LoRAs are in
[distilled/comfyui](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/distilled/comfyui).

Chain two particle nodes to mix Candlelit and Moonlit. The nodes use independent direct strengths. To match a Studio mix, normalize each slider to `energy × proportion / sum(proportions)` before entering its node strength.

The custom node retains all particle tensors and maps all 224 native Q/K/V/output branches to ComfyUI's separate projections. It registers temporary hooks inside the clone's model-call wrapper and removes them in `finally`, including after failures. It does not modify the original model's persistent weights. Use the particle files with this node.

## Distilled ordinary LoRAs

Download the files from [distilled/comfyui](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/distilled/comfyui), give them distinct local names such as `candlelit-distilled.safetensors`, and place them in `ComfyUI/models/loras/`.

Select the current **`*-unit-alpha.safetensors`** files. Bad Intent uses alpha 24;
Candlelit uses 16 and Moonlit 23.08072421821097. The older filenames retain the
original alpha-8 exports for reproduction. For the old Bad Intent LoRA, strength
3 is equivalent to the current file at strength 1.

Use standard **Load LoRA** with **MODEL strength 1**, **CLIP strength 0**, or **Load LoRA (Model Only)**. No custom particle node is required. They approximate the particle adapters; use the model card's matched comparisons to judge the differences.

## Verification scope

[CPU integration evidence](validation/comfyui.json) records loading all three real particle files, all 224 projection mappings, exact native branch math, zero bypass, clone isolation, exception cleanup, and ordinary-LoRA loader acceptance. Public images were rendered in the pinned native Anima runtime. Full ComfyUI GPU image generation has not been validated in this release. ComfyUI and native sampling/text-conditioning details may produce different images from the same integer seed.
