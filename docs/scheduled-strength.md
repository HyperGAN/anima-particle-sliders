# Scheduled strength for Anima

Use the **early** schedule for the measured Anima Turbo Candlelit RAW256
profile. It uses the same RAW checkpoint and ten native denoising steps with
more strength in the first half of the traversed noise range. Across six
matched 512 × 512 cases, Candlelit gain was +0.00609 with early steering,
+0.00379 at constant strength and +0.00186 with late steering.

## Render with the measured schedule

Use the Candlelit RAW256 adapter with SHA-256
`90f831220f630c5a23923da3a7b571b5cc5ab5d99c4150dd054e98b4ddc24ac1`.
The inference command multiplies `--strength` by the supplied gains for each
denoiser call and saves the native sigmas and effective strengths next to the
PNG. Pass the neutral prompt of the case you want to render.

```sh
python -m release_tools.infer \
  --model /path/to/anima-turbo-v1.1 --adapter /path/to/RAW-0256.safetensors \
  --format particles --prompt "$PROMPT" --seed 29001 --width 512 --height 512 \
  --steps 10 --strength 1 \
  --step-gains '[1.5,1.5,1.5,1.5,1.5,1.5,1.5,0.9666667853461673,0.5,0.5]' \
  --output candlelit-early.png
```

For an OFF comparison, replay at `--strength 0`. For the constant ON comparison,
omit `--step-gains` and use `--strength 1`.

## Final Boss RAW192

Use early strength at nominal 2/3 for the tested Final Boss RAW192 checkpoint
on Anima Turbo v1.1 at 512 × 512 with ten native Euler intervals and CFG 1.
The four held-out cases averaged +0.34333 Final Boss intensity over matched
OFF, compared with +0.18833 at equal-dose constant strength and +0.34083
at constant strength 1. Early improved three of four cases over strength 1.
All 16 images passed the integrity check. An independent CUDA render
reproduced all four early PNGs byte for byte.

With the exact RAW192 export, pass `--strength 0.6666666666666666` and the
ten relative gains recorded in the evidence. Match the native sigma grid in
the render sidecar before using the result as a replay.
[Checkpoint, gains and per-case evidence](final-boss-raw192-scheduled-strength.json)

## Apply it to another Anima slider

Use the saved RAW adapter. Resolve the gains from that render's native noise
intervals, then pass the gains in denoiser order. If you change the step count
or scheduler, resolve the gains on the new grid. Compare early, late and
constant renders on matched cases and use the profile with the highest mean
intended-effect gain. Check catastrophic output separately.
[Particle Sliders schedule guide](https://github.com/HyperGAN/particle-sliders/blob/main/docs/strength-schedules.md)

## Calibrate activation-dependent strength

Anima can allocate the saved RAW particle adapter's nominal strength with CFP,
CLABS or the matched random control from Particle Sliders core. At each native
Euler interval, read the current latent as the denoiser will receive it and
average each BF16 input channel across the single image's spatial positions
using an FP32 reduction. The shared signed projection and gate act on that
channel vector before the one CFG 1 conditioned denoiser call. A firing uses
the nominal RAW adapter strength and an inactive interval uses exact OFF.
This adapts the papers' allocation equations to the Anima particle residual.
The fired Anima edit uses the routed particle adapter. The paper applies a rank-one
addition to a language-model residual.

After calibration, pass its JSON plan and the explicit nominal strength:

```sh
python -m release_tools.infer \
  --model "$ANIMA_MODEL" --adapter "$RAW_ADAPTER" --format particles \
  --prompt "$PROMPT" --seed "$CASE_SEED" --width "$WIDTH" --height "$HEIGHT" \
  --steps "$NATIVE_STEPS" --strength "$CALIBRATED_STRENGTH" \
  --gate-plan "$CALIBRATED_GATE_JSON" --output "$OUTPUT_PNG"
```

The plan requires `method` and `parameter`. Supply the unit `direction` fitted
in Anima latent channel space for `cfp` or `clabs`. In CFP `parameter` is the
calibrated alignment threshold. In CLABS it is the calibrated bias. CLABS also
requires `gate_seed` and a `clabs` object containing measured `mean` and `std`
and the declared `epsilon`, `setpoint`, `proportional_gain`, `integral_gain`
and `beta`. For `random`, set `parameter` to the matched firing rate and pass
its `gate_seed`. The gate owns one state per render. The sidecar records the
plan and its SHA-256, fired intervals, probabilities, effective strengths,
and realized dose `sum((sigma[i] - sigma[i+1]) * effective_strength[i])`.
An activation-plan render at strength zero replays OFF without gate steps.

Fit the direction on independent concept-positive and neutral trajectories for
this exact input signal. The saved Candlelit teacher cache supplies paired
positive and neutral prompt trajectories and stored latents for planning the
fitting split. Bind the SAE and direction to the chosen latent channel signal,
model source, conditioning, geometry and sigma grid. Use a separate active
calibration split to estimate the projection statistics, match the declared
budget with the shared bisection helper, and select the admissible strength.
Keep the archived Candlelit and Final Boss matched cases for scoring beginning
with the saved OFF and fixed-schedule results. Compare finished-output mean
intended-effect ON minus OFF at matched dose and check catastrophic output
separately.
[Paper equations and the shared calibration procedure](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/docs/strength-schedules.md#activation-dependent-steering)

### Candlelit RAW256 native gate study

A tied one-sparse autoencoder learned a 32-feature dictionary from 384 paired
Candlelit and neutral native trajectories. The ten most enriched positive
features determined the direction using the shared core. A separate 12-case
live calibration targeted the paper’s 50% intervention rate. The fitted CFP
threshold fired on 60 of 120 intervals. The fitted CLABS bias fired on 61 of
120, within one interval of the target. CLABS used the paper’s controller gains
and normalization measured on the calibration OFF trajectories.

These results cover six held-out Turbo v1.1 renders at 512 × 512, CFG 1 and
ten native FlowMatch Euler intervals. Each gate fired at the archived RAW256
adapter’s nominal strength of 1. Each constant control matched that gate’s
realized sigma-integrated strength on the same case. All six archived OFF
images reproduced byte for byte. Each reported gain is the equal-case mean
Candlelit intensity of the finished image minus its matched OFF score.

| Allocation | Gate gain | Matched constant gain | Cases ahead | Held-out firing |
| --- | ---: | ---: | ---: | ---: |
| CFP | +0.003817 | +0.000802 | 5/6 | 22/60 |
| CLABS | +0.003469 | +0.000948 | 5/6 | 23/60 |
| Random | +0.002596 | +0.002189 | 3/6 | 32/60 |

All 36 gated and constant outputs and the six matched OFFs passed the
independent categorical integrity check. Realized dose varied across the
gates, so compare each row with its own constant. These are scoped research
results from the captured ParticleGAN 0.10.0 runtime. The measured early
profile above remains the recommended Candlelit profile. Refit and calibrate
for a different adapter or native grid.
[Fitted values, native grid and per-case evidence](candlelit-raw256-activation-gates.json) · [Right Token, Right Time](https://arxiv.org/html/2611.18274v1#A2)

### Render the measured Candlelit CFP setting

Use Anima Turbo v1.1 and the exact Candlelit RAW256 adapter identified in the
evidence. The following opt-in command reads the fitted direction, calibrated
threshold, nominal strength and render dimensions directly from the study.
CFP uses the native activation projection and needs only the image render
seed. Set `ANIMA_MODEL`, `RAW_ADAPTER`, `PROMPT` and `CASE_SEED` for your render.

```sh
evidence=docs/candlelit-raw256-activation-gates.json
test "$(sha256sum "$RAW_ADAPTER" | cut -d' ' -f1)" = "$(jq -er .checkpoint_sha256 "$evidence")" &&
jq -e '{method:"cfp", parameter:.verified_calibration.cfp.parameter,
        direction:.fitted_native_probe.direction}' "$evidence" > candlelit-cfp-plan.json &&
python -m release_tools.infer \
  --model "$ANIMA_MODEL" --adapter "$RAW_ADAPTER" --format particles \
  --prompt "$PROMPT" --seed "$CASE_SEED" \
  --width "$(jq -er .profile.width "$evidence")" \
  --height "$(jq -er .profile.height "$evidence")" \
  --steps "$(jq -er '.profile.sigmas | length - 1' "$evidence")" \
  --strength "$(jq -er .profile.nominal_on_strength "$evidence")" \
  --gate-plan candlelit-cfp-plan.json --output candlelit-cfp.png
```

The resulting JSON sidecar records the plan, the gate decisions and the
effective strength for every step. Check the adapter, base model and native
sigma grid against the studied profile before comparing the rendered output
with its archived counterpart.

```sh
jq -e --slurpfile study "$evidence" \
  '.adapter_sha256 == $study[0].checkpoint_sha256 and
   .model_identity.sha256 == $study[0].profile.base_sha256 and
   .sigmas == $study[0].profile.sigmas' candlelit-cfp.json
```

### Render the measured Candlelit CLABS setting

CLABS uses the fitted direction, live-calibrated bias and controller settings
from the same study. On the six held-out cases it gained +0.003469 over OFF
and +0.002520 over its equal-dose constant, with five of six cases ahead of
that constant. It fired on 61 of 120 calibration intervals and 23 of 60
held-out intervals.

Use the same model and RAW256 export as the CFP example. For an archived
case, supply its neutral `PROMPT` and native `CASE_SEED`. Set `CACHED_CASE`
to its original frozen conditioning shard. The study seeded the CLABS
random number generator with the first 64 bits of that shard's SHA-256.
The image noise seed is a separate input.

```sh
evidence=docs/candlelit-raw256-activation-gates.json
test "$(sha256sum "$RAW_ADAPTER" | cut -d' ' -f1)" = "$(jq -er .checkpoint_sha256 "$evidence")" &&
python - "$evidence" "$CACHED_CASE" > candlelit-clabs-plan.json <<'PY'
from hashlib import file_digest
import json
from pathlib import Path
import sys

study = json.loads(Path(sys.argv[1]).read_bytes())
with Path(sys.argv[2]).open("rb") as case:
    gate_seed = int(file_digest(case, "sha256").hexdigest()[:16], 16)
print(json.dumps({
    "method": "clabs",
    "parameter": study["verified_calibration"]["clabs"]["parameter"],
    "direction": study["fitted_native_probe"]["direction"],
    "gate_seed": gate_seed,
    "clabs": study["clabs_parameters"],
}))
PY
```

Run the inference command from the CFP example with `--gate-plan
candlelit-clabs-plan.json --output candlelit-clabs.png`, then run the
same identity and sigma checks on `candlelit-clabs.json`. To render that
case's constant, omit the gate plan and use strength equal to the sidecar's
`sigma_weighted_strength` divided by the difference between the first and
last native sigmas.

For the original held-out cases, check the sidecar's `effective_strengths`
against these measured masks in denoising order. Each 1 applies the
profile's nominal strength.

| Frozen case | Image noise seed | CLABS mask | Matched constant strength |
| --- | ---: | --- | ---: |
| `candlelit-dev-00` | 29001 | `1111000000` | 0.1818181872367859 |
| `candlelit-dev-01` | 29001 | `1110000000` | 0.12500005960464478 |
| `candlelit-dev-02` | 29001 | `1111000000` | 0.1818181872367859 |
| `candlelit-dev-03` | 29027 | `1111100000` | 0.25 |
| `candlelit-dev-04` | 29027 | `1111000000` | 0.1818181872367859 |
| `candlelit-dev-05` | 29001 | `1101000000` | 0.13374114036560059 |

The archived results used frozen conditioning in the captured ParticleGAN
0.10.0 runtime. Use that runtime and those cases for a native comparison.
## Fitted prompt steering on native velocities

`TurboRuntime.render_spsr` applies a fitted shared `PromptSteering` module
once per native Euler interval. It pools the current 16-channel velocity
field in FP32, evaluates the fitted coefficient, and adds the fitted
direction times that coefficient before the original Euler step. The
standalone intervention runs with the particle mixer at zero. It returns
the issued coefficients, sigmas, and integrated coefficient for the render.
[Steer Like the LLM](https://arxiv.org/abs/2605.03907)

The Candlelit study used the captured Turbo v1.1 CUDA runtime, 512×512 images,
and 10 Euler intervals. Its archived `probe-raw.pt` has SHA-256
`374483b8da7444c763bcdf40d557f421cae8db80e34c5e4dd52050ef1a46be82`.
The raw 16-channel fit used 384 prompt-paired native velocity trajectories
and 5,760 completed probe updates. Use the originally archived prompt
conditioning and seed when reproducing that study.

```python
from hashlib import file_digest
from pathlib import Path

import torch
from particle_sliders.prompt_steering import PromptSteering

with Path(probe_checkpoint).open("rb") as source:
    expected = "374483b8da7444c763bcdf40d557f421cae8db80e34c5e4dd52050ef1a46be82"
    if file_digest(source, "sha256").hexdigest() != expected:
        raise ValueError("Use the fitted Candlelit S-PSR checkpoint")
    source.seek(0)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
steering = PromptSteering(runtime.channels).eval().requires_grad_(False)
steering.load_state_dict(checkpoint["probe"], strict=True)
steering.to(runtime.device)
image, trace = runtime.render_spsr(
    prompt, seed, steering=steering, width=512, height=512, steps=10,
)
```

The archived study set its constant coefficient to 0.0784376747981228 from
12 disjoint live calibration renders. Across six held-out images, S-PSR
averaged +0.00068 over OFF and the calibrated constant averaged +0.00132.
S-PSR led on two of six cases, and both arms passed independent integrity
checks. The constant was calibrated across separate prompts, and the two
arms used different realized doses on some held-out cases. Exact PNG
reproduction uses the archived conditioning and numerical profile.
