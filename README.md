# Anima Concept Sliders

**Bad Intent — psychological horror for Anima Turbo v1.1.** An eager smile, a looming gaze, something wrong. Plus Moonlit and Candlelit lighting sliders. Each comes as the original Particle adapter and a standard LoRA Distill.

## Samples

**Start with Particle — the original slider.** Each comparison reads **Particle → Distill → Off**, left to right. Distill is the ordinary-LoRA approximation; Off is the base model.

### Bad Intent

Intense psychological horror, without gore. Start at **strength 1**. This deliberately changes expression, pose, framing and sometimes appearance or medium; identity preservation is outside its brief. Formerly named Uncanny.

**Particle · strength 1** → **Distill · strength 1** → **Off · strength 0**

![Bad Intent: Particle first, Distill second, Off third](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/bad-intent-case0-particle-distill-off.jpg)

Full-resolution samples: [Particle](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/samples/bad-intent/case0/str1.png) · [Distill](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/distilled/samples/bad-intent-case0-str1.png) · [Off](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/samples/bad-intent/case0/str0.png)

Same prompt and seed **29001**. 768 × 768, 10 Euler steps, CFG 1. **The Distill is much milder at the same strength:** it does not reproduce the original's looming pose or pronounced expression in these examples. Choose Particle for the featured effect.

<details><summary>Prompt</summary>

masterpiece, best quality, safe, solo, 1girl, mature female, adult woman age 35, tan skin, medium brown skin, long straight slate-blue hair, hazel eyes, neutral expression, closed mouth, looking at viewer, wearing a tobacco-brown coat and black skirt, long sleeves, plain fabric, standing upright, facing viewer, arms at sides, upper body, waist-up view, eye-level camera, in an unadorned stone corridor with bare stone walls, anime illustration

</details>

<details><summary>Bad Intent: bare prompt comparison and strength 0.5 samples</summary>

![Bad Intent bare prompt: particle versus distilled LoRA and Off at strength 1](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/bad-intent-case1-particle-distill-off.jpg)

Particle → Distill → Off, at strengths 1 → 1 → 0. Same seed and settings as above. The particle version also shifts this case toward a rendered/realistic medium.

1boy, male focus, mature male, adult man age 35, light skin, short curly red hair, blue eyes, neutral expression, closed mouth, looking at viewer, wearing a cream cable-knit sweater and olive trousers, long sleeves, plain fabric, standing upright, facing viewer, both hands on hips, cowboy shot, thighs in frame, eye-level camera, in an unadorned stone corridor with bare stone walls

[All Bad Intent particle samples, strengths 0 / 0.5 / 1](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/samples/bad-intent) · [Distill samples](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/distilled/samples)

</details>

### Moonlit

**Particle · strength 3** → **Distill · strength 3** → **Off · strength 0**

![Moonlit: Particle first, Distill second, Off third](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/moonlit-portrait-particle-distill-off.jpg)

Full-resolution samples: [Particle](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/samples/featured-portrait/moonlit-particle.png) · [Distill](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/samples/featured-portrait/moonlit-distill.png) · [Off](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/samples/featured-portrait/off.png)

Same close-up portrait prompt and seed **29001**. 768 × 768, 10 Euler steps, CFG 1. Strength 3 extrapolates beyond the trained endpoint at 1.

<details><summary>Prompt</summary>

1boy, male focus, solo, mature male, adult man age 35, light skin, short curly red hair, blue eyes, neutral expression, closed mouth, looking at viewer, wearing a cream cable-knit sweater, crew neck, facing viewer, head upright, close-up portrait, head and shoulders, face filling the frame, eye-level camera, in an unadorned stone corridor with bare stone walls, anime illustration, neutral diffuse illumination, moderate shadow contrast

</details>

### Candlelit

**Particle · strength 3** → **Distill · strength 3** → **Off · strength 0**

![Candlelit: Particle first, Distill second, Off third](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/candlelit-portrait-particle-distill-off.jpg)

Full-resolution samples: [Particle](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/samples/featured-portrait/candlelit-particle.png) · [Distill](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/samples/featured-portrait/candlelit-distill.png) · [Off](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/samples/featured-portrait/off.png)

Same close-up portrait prompt and seed **29001**. 768 × 768, 10 Euler steps, CFG 1. Strength 3 extrapolates beyond the trained endpoint at 1.

<details><summary>Prompt</summary>

1boy, male focus, solo, mature male, adult man age 35, light skin, short curly red hair, blue eyes, neutral expression, closed mouth, looking at viewer, wearing a cream cable-knit sweater, crew neck, facing viewer, head upright, close-up portrait, head and shoulders, face filling the frame, eye-level camera, in an unadorned stone corridor with bare stone walls, anime illustration, neutral diffuse illumination, moderate shadow contrast

</details>

<details>
<summary>More lighting samples: strengths 0–5, four matched prompts per slider</summary>

Each grid reads **0, 1, 2** across the first row, then **3, 4, 5**. Original step-1600 EMA particles.

### Moonlit

**Comparison 1** · [PNG images and prompt](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/samples/moonlit/case0)

![Moonlit: matched samples at strengths 0 through 5](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/moonlit-case0.jpg)

<details><summary>Prompt and seed</summary>

masterpiece, best quality, safe, solo, 1girl, mature female, adult woman age 35, tan skin, medium brown skin, long straight slate-blue hair, hazel eyes, neutral expression, closed mouth, looking at viewer, wearing a tobacco-brown coat and black skirt, long sleeves, plain fabric, standing upright, facing viewer, arms at sides, upper body, waist-up view, eye-level camera, in an unadorned stone corridor with bare stone walls, anime illustration, neutral diffuse illumination, moderate shadow contrast

Seed: `29001`.

</details>

**Bare prompt** · [PNG images and prompt](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/samples/moonlit/case3)

![Moonlit: matched samples at strengths 0 through 5](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/moonlit-case3.jpg)

<details><summary>Prompt and seed</summary>

1boy, male focus, mature male, adult man age 35, light skin, short curly red hair, blue eyes, neutral expression, closed mouth, looking at viewer, wearing a cream cable-knit sweater and olive trousers, long sleeves, plain fabric, standing upright, facing viewer, both hands on hips, cowboy shot, thighs in frame, eye-level camera, in an unadorned stone corridor with bare stone walls

Seed: `29001`.

</details>

<details>
<summary>Two more matched prompts</summary>

**Comparison 2** · [PNG images and prompt](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/samples/moonlit/case1)

![Moonlit: matched samples at strengths 0 through 5](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/moonlit-case1.jpg)

<details><summary>Prompt and seed</summary>

masterpiece, best quality, safe, solo, 1girl, mature female, adult woman age 35, dark skin, deep brown skin, short silver coils, brown eyes, neutral expression, closed mouth, looking at viewer, wearing a jade high-neck blouse and charcoal trousers, long sleeves, plain fabric, facing viewer, head upright, portrait, head and shoulders, eye-level camera, in a quiet courtyard, anime illustration, neutral diffuse illumination, moderate shadow contrast

Seed: `29001`.

</details>

**Comparison 3** · [PNG images and prompt](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/samples/moonlit/case2)

![Moonlit: matched samples at strengths 0 through 5](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/moonlit-case2.jpg)

<details><summary>Prompt and seed</summary>

masterpiece, best quality, safe, solo, 1girl, mature female, adult woman age 35, tan skin, medium brown skin, long straight slate-blue hair, hazel eyes, neutral expression, closed mouth, looking at viewer, wearing a tobacco-brown coat and black skirt, long sleeves, plain fabric, sitting upright on a plain wooden chair, facing viewer, feet flat on floor, hands resting in lap, full body, entire figure in frame, eye-level camera, in a wood-paneled room, anime illustration, neutral diffuse illumination, moderate shadow contrast

Seed: `29001`.

</details>

</details>

### Candlelit

**Comparison 1** · [PNG images and prompt](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/samples/candlelit/case0)

![Candlelit: matched samples at strengths 0 through 5](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/candlelit-case0.jpg)

<details><summary>Prompt and seed</summary>

masterpiece, best quality, safe, solo, 1girl, mature female, adult woman age 35, tan skin, medium brown skin, long straight slate-blue hair, hazel eyes, neutral expression, closed mouth, looking at viewer, wearing a tobacco-brown coat and black skirt, long sleeves, plain fabric, standing upright, facing viewer, arms at sides, upper body, waist-up view, eye-level camera, in an unadorned stone corridor with bare stone walls, anime illustration, neutral diffuse illumination, moderate shadow contrast

Seed: `29001`.

</details>

**Bare prompt** · [PNG images and prompt](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/samples/candlelit/case3)

![Candlelit: matched samples at strengths 0 through 5](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/candlelit-case3.jpg)

<details><summary>Prompt and seed</summary>

1boy, male focus, mature male, adult man age 35, light skin, short curly red hair, blue eyes, neutral expression, closed mouth, looking at viewer, wearing a cream cable-knit sweater and olive trousers, long sleeves, plain fabric, standing upright, facing viewer, both hands on hips, cowboy shot, thighs in frame, eye-level camera, in an unadorned stone corridor with bare stone walls

Seed: `29001`.

</details>

<details>
<summary>Two more matched prompts</summary>

**Comparison 2** · [PNG images and prompt](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/samples/candlelit/case1)

![Candlelit: matched samples at strengths 0 through 5](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/candlelit-case1.jpg)

<details><summary>Prompt and seed</summary>

masterpiece, best quality, safe, solo, 1girl, mature female, adult woman age 35, dark skin, deep brown skin, short silver coils, brown eyes, neutral expression, closed mouth, looking at viewer, wearing a jade high-neck blouse and charcoal trousers, long sleeves, plain fabric, facing viewer, head upright, portrait, head and shoulders, eye-level camera, in a quiet courtyard, anime illustration, neutral diffuse illumination, moderate shadow contrast

Seed: `29001`.

</details>

**Comparison 3** · [PNG images and prompt](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/samples/candlelit/case2)

![Candlelit: matched samples at strengths 0 through 5](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/candlelit-case2.jpg)

<details><summary>Prompt and seed</summary>

masterpiece, best quality, safe, solo, 1girl, mature female, adult woman age 35, tan skin, medium brown skin, long straight slate-blue hair, hazel eyes, neutral expression, closed mouth, looking at viewer, wearing a tobacco-brown coat and black skirt, long sleeves, plain fabric, sitting upright on a plain wooden chair, facing viewer, feet flat on floor, hands resting in lap, full body, entire figure in frame, eye-level camera, in a wood-paneled room, anime illustration, neutral diffuse illumination, moderate shadow contrast

Seed: `29001`.

</details>

</details>

</details>

Strength **1** is the trained endpoint. Higher strengths extrapolate and can change clothing, framing and scene details; occasional camera-frame marks are visible in the unfiltered examples. These development samples illustrate behavior, not a final-test quality benchmark.

<a id="comfyui"></a>

## Get the adapters

| Slider | Original particles | Distilled LoRA for ComfyUI | Distilled LoRA for native runtime |
|---|---|---|---|
| Bad Intent | [Download](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/weights/bad-intent.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/distilled/comfyui/bad-intent.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/distilled/native/bad-intent.safetensors?download=true) |
| Moonlit | [Download](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/weights/moonlit.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/distilled/comfyui/moonlit.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/distilled/native/moonlit.safetensors?download=true) |
| Candlelit | [Download](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/weights/candlelit.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/distilled/comfyui/candlelit.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/distilled/native/candlelit.safetensors?download=true) |

**Particles:** use the [ComfyUI particle plugin](https://github.com/mikkel/anima-concept-sliders#comfyui), or the [native loader](https://github.com/mikkel/anima-concept-sliders/blob/main/REPRODUCE.md). [Plugin ZIP](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/comfyui/anima-concept-sliders.zip?download=true) · [ComfyUI setup](https://github.com/mikkel/anima-concept-sliders/blob/main/COMFYUI.md).

**Distilled LoRAs:** use standard **Load LoRA**, MODEL strength 1 and CLIP strength 0. These rank-8 linear approximations need no particle plugin. Compare their images below before choosing a format.

[Public source and training reproduction](https://github.com/mikkel/anima-concept-sliders) · [Every sample and checkpoint](https://huggingface.co/ntc-ai/anima-concept-sliders/blob/main/catalog.json) · [File hashes](https://huggingface.co/ntc-ai/anima-concept-sliders/blob/main/release-manifest.json) · [Related YuE2 release](https://huggingface.co/ntc-ai/yue2-concept-sliders).

The reusable algorithms come from the [shared core in sliders-conceptmod](https://github.com/mikkel/sliders-conceptmod/tree/main/packages/concept-slider-core). This Anima repository pins an exact core revision and owns the model integration, recipe and release evidence.

## Distilled comparisons

<details>
<summary>Compare lighting particles and distills at strengths 1, 3 and 5</summary>

Top row: original particles at strengths **1, 3, 5**. Bottom row: the distilled ordinary LoRA at the same strengths. Prompt, seed, dimensions and sampler are identical. [All four prompt comparisons](https://huggingface.co/ntc-ai/anima-concept-sliders/tree/main/distilled/samples).

### Moonlit — particles and distilled

![Moonlit particle versus distilled LoRA at strengths 1, 3 and 5](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/moonlit-distilled.jpg)

### Candlelit — particles and distilled

![Candlelit particle versus distilled LoRA at strengths 1, 3 and 5](https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/assets/candlelit-distilled.jpg)

</details>

The featured comparisons show a milder lighting change from the distills at the same strength. They are experimental approximations, not interchangeable replacements.

The distills were fitted on training activations. Development projection-relative MSE (lower is closer to the particle delta; **not** an image-quality score):

| Slider | Relative MSE |
|---|---:|
| Bad Intent | 0.005434 |
| Moonlit | 0.007559 |
| Candlelit | 0.020457 |

[Distillation formulation and reproduction](DISTILLATION.md).

## How the sliders learn

These are nonlinear attention adapters for **Anima Turbo v1.1**. The base diffusion transformer, text encoders, text conditioner, and VAE stay frozen. Bad Intent, Candlelit and Moonlit each have their own adapter, critic, and learned particle cloud. Bad Intent uses the same game with a broader psychological-horror target: expression, pose and framing may change as well as atmosphere. Its internal training identifier remains `uncanny`.

### The particle branch

Every self-attention and cross-attention Q, K, V, and output projection receives an additive branch: 224 projections across 28 transformer blocks. For a projection input \\(x\\), define

$$
u = W_{\mathrm{down}}x,\qquad q=R(u),\qquad a=\operatorname{softmax}\left(\frac{qP^\top}{\sqrt{4}}\right),\qquad z=aP.
$$

Here \\(W_{\mathrm{down}}\\) maps into rank 8, \\(P\in\mathbb R^{128\times4}\\) is the shared particle cloud, and \\(R\\) is a per-projection MLP of width 16. The branch MLP \\(F\\) has width 48. All hidden activations are LeakyReLU with slope 0.2; both MLPs have three hidden linear layers and a final output layer. The projection becomes

$$
y = W_0x + s\,W_{\mathrm{up}}F([u,z]).
$$

The up matrix maps rank 8 back to the projection output width and starts at zero. The same soft routing runs during training and inference. Strength \\(s=0\\) bypasses the branch exactly. Training uses \\(s=1\\); the gallery includes extrapolation through \\(s=5\\). For multiple sliders the deltas sum on the same input. Studio normalizes nonnegative mix proportions as \\(s_j=E m_j/\sum_k m_k\\).

An ordinary LoRA has a linear branch \\(BAx\\). The routed particle branch is nonlinear, so its full behavior cannot be merged into a fixed weight matrix. The [distilled LoRAs](DISTILLATION.md) are fitted approximations.

### Paired frozen targets

For each matched neutral/positive prompt pair and seed, generate both frozen trajectories from the same initial noise. At every one of the ten sampling positions on **both** paths, evaluate the neutral and positive frozen velocities at that same latent state. No spatial pooling is used.

Let \\(v_0(z,t,c_n)\\) and \\(v_0(z,t,c_p)\\) be those frozen velocities. Let \\(v_\theta(z,t,c_n)\\) be the adapted neutral prediction. The desired and student edits are

$$
\Delta_T=v_0(z,t,c_p)-v_0(z,t,c_n),\qquad
\Delta_S=v_\theta(z,t,c_n)-v_0(z,t,c_n).
$$

The normalized residual is

$$
e_t=\frac{\Delta_S-\Delta_T}{S_t}
=\frac{v_\theta(z,t,c_n)-v_0(z,t,c_p)}{S_t}.
$$

Each \\(S_t\\) is fitted only on training targets. First compute the per-coordinate sample standard deviation of \\(\Delta_T\\), floored at \\(10^{-4}\\). Multiply that scale by the median per-row RMS of the normalized edits, so their median RMS becomes one. Development data never fits normalization.

### The paired-error game

Use the same Gaussian noise vector for real and fake within a pair:

$$
x_r=\sigma_t(k)\epsilon,\qquad x_f=\sigma_t(k)\epsilon+e_t,\qquad \epsilon\sim\mathcal N(0,I).
$$

The critic is the pinned global-mix network: a projection of the full velocity residual into eight width-48 tokens, one attention layer with four heads, and output score \\(8\tanh(\cdot/8)\\). Separate saved streams choose the D/G rows and noise.

$$
L_D=\mathbb E\,\operatorname{softplus}(D(x_f)-D(x_r))+L_{\mathrm{cap}},\qquad
L_G=\mathbb E\,\operatorname{softplus}(D(x_r)-D(x_f))+L_{\mathrm{VIC}}.
$$

Every fourth update the critic receives the exact autograd gradient cap, with lazy multiplier four:

$$
L_{\mathrm{cap}}=\frac{4}{2}\mathbb E\left[
\max(0,\|\nabla D(x_r)\|_2-1)^2+
\max(0,\|\nabla D(x_f)\|_2-1)^2\right].
$$

Other updates have zero cap loss. VIC acts on a random 64-particle subset: mean \\(\max(0,1-\sqrt{\operatorname{Var}(P_j)+10^{-4}})\\) plus the squared off-diagonal sample covariance sum divided by particle dimension four. Both regularizer coefficients are one. There is no reconstruction or perceptual auxiliary loss in the particle training game.

Noise starts at the training edit RMS divided by 0.28. It decays geometrically toward 0.03 over 1,600 updates, with an absolute hold at 1.0 when the start exceeds that hold. Update \\(k\\) uses schedule index \\(k-1\\). The pilot keeps the same horizon.

### Released recipe and evidence

| Setting | Value |
|---|---|
| Checkpoint | EMA after 1,600 updates, seed 7 |
| Adam learning rates | Generator \\(2\times10^{-5}\\); critic \\(9\times10^{-4}\\); particles \\(6\times10^{-3}\\) |
| Adam betas / decay | \\((0,0.999)\\) / zero |
| Effective batch / microbatch | 8 / 1 |
| EMA decay | 0.995 |
| Training resolution | 512 × 512 |
| Public samples | 768 × 768, 10 Euler steps, CFG 1, scheduler shift 3 |
| Per slider training data | 24 paired prompt rows × 16 seeds; both trajectories × 10 positions |

The original adult character catalog separates 12 training characters, four development characters, and eight final-test characters. Six definitions train each slider; two extra paraphrases are held out. Candlelit and Moonlit edit lighting; Bad Intent uses expression, pose, framing and atmosphere definitions. The public gallery uses development cases, including a bare prompt, and is not a final-test benchmark.

These are final-budget experimental checkpoints. They were not selected as the best checkpoint by perceptual quality. Strengths above one extrapolate beyond training and can change clothing, pose, setting, or composition. The measured 1,600-update runs did not establish convergence. Training traces, normalization provenance, checkpoint hashes, and development measurements accompany the weights.

### Shared algorithm implementation

The reusable routing, global-mix critic, paired losses, particle regularizer, noise function and ordinary-LoRA solver live in [sliders-conceptmod](https://github.com/mikkel/sliders-conceptmod/tree/beaffeb3640c4554a7315998c04a5909f384b972/packages/concept-slider-core). This release pins that revision in `requirements.txt` and [core.lock.json](https://huggingface.co/ntc-ai/anima-concept-sliders/blob/main/core.lock.json). Anima owns the model integration, target collection, training recipe and sample evidence. The extracted reference implementation is byte-identical to the original; runtime identity hashes the implementation rather than its compatibility import. [Architecture and research](https://github.com/mikkel/sliders-conceptmod/blob/main/docs/shared-core.md).

## License

The adapters are derivatives of [CircleStone Anima](https://huggingface.co/circlestone-labs/Anima) and are distributed under the [CircleStone Labs Non-Commercial License](ANIMA-LICENSE.md). Rights are granted directly by CircleStone Labs LLC. See [NOTICE](NOTICE). Independently authored source code is MIT; vendored reference code retains its included license. This is an independent ntc-ai release.
