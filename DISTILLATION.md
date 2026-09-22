# Ordinary rank-8 LoRA distillation

The native particle adapters remain the reference. The ordinary LoRAs replace each nonlinear branch with two linear matrices so they can use standard ComfyUI **Load LoRA**: MODEL strength controls the edit; CLIP strength is **0**.

For each particle branch, keep the learned up matrix $U$ fixed and fit a new down matrix $A$ to the routed bottleneck output $h(x)=F([W_{down}x,z(x)])$ on actual Anima projection inputs. For input and target row matrices $X,H$:

$$
A^\top=X^\top(XX^\top+\lambda I)^{-1}H,\qquad
\lambda=0.01\,\operatorname{mean}\operatorname{diag}(XX^\top).
$$

The distilled delta is $(\alpha/8)UAx$, rank eight. The original distills used alpha eight; the current calibrated releases use Bad Intent 24, Candlelit 16 and Moonlit 23.08072421821097. The exporter preserves alpha in both native and ComfyUI files. It has no particle cloud, router, activation, or learned bias. This is local activation regression, not diffusion-step acceleration: the base is already Turbo and still uses the same 10-step sample schedule.

Calibration uses all 24 training rows, the first seed per row, five positions (neutral-path states at 0, 2, 4, 6 and positive-path states at 9), and eight evenly spaced token activations per projection. The native particle teacher runs at strength one on those cached states. That gives 960 inputs per projection. The ridge coefficient is fixed before development evaluation.

Held-out measurements use development characters on positive-path states at positions 1 and 5 and neutral-path states at position 8. Relative projection MSE is the sum of squared output-delta errors divided by the sum of squared particle output deltas across all projections. It is an internal approximation diagnostic, not an image-quality score. The public image comparisons show the original particles and distills on identical prompts and seeds at strengths 1, 3, and 5. Drift compounds through the denoiser, especially above the trained strength.

`release_tools/distill.py` calls `concept_slider_core.fit_routed_down` from the [pinned shared core](https://github.com/mikkel/sliders-conceptmod/tree/beaffeb3640c4554a7315998c04a5909f384b972/packages/concept-slider-core), exports both native and ComfyUI tensor names, and renders the comparisons. The exporter does not fuse Q/K/V because ComfyUI Anima exposes separate projections. `release_tools/verify_comfy.py` checks all 224 mappings against actual ComfyUI modules and verifies native/ComfyUI matrix equality.

The original renders showed a milder lighting change from the distills at equal strength. Small aggregate projection errors do not guarantee matched images, and individual projection errors can be much larger. The original per-projection measurements remain included with each student; they describe the pre-calibration fit, not the calibrated full-denoiser edit.

Bad Intent (internal training ID `uncanny`) uses exactly this fitting recipe with
its own training/development caches and step-1600 EMA teacher. Its held-out
projection-relative MSE was **0.005434** before calibration. On the two original public development cases,
the alpha-8 strength-1 distill was much milder and did not reproduce the particle
teacher's looming crop or pronounced expression. This is an approximation
failure despite the small projection error. Its matched samples use 0.5 and 1;
Candlelit/Moonlit retain their original 1, 3 and 5 comparisons. No final-test
characters were used for fitting, tuning or these demonstrations.

## Bad Intent alpha calibration, 2026-09-21

A subsequent development sweep found that **alpha 24 at rank 8** recovers the
looming composition and intense expression in both featured cases. The original
alpha-8 distill at **MODEL strength 3** performs the same operation. This changes
only the ordinary LoRA's alpha; that first sweep kept the particle teacher at
alpha 8, strength 1. The subsequent visual-strength balance below raises the
current particle alpha to 12.
The alpha-24 native and ComfyUI derivatives are the current downloads under
`distilled/{native,comfyui}/bad-intent-unit-alpha.safetensors`. Original alpha-8
files and historical examples remain at their original paths. Use nominal MODEL
strength 1 and CLIP strength 0 with the current ComfyUI file.

The audit compares the complete denoiser edit `v_adapter - v_base` on 40 neutral
cached states: all ten positions for four development characters, seed 29001,
512px. The teacher uses the exact public particle file. Recomputed base outputs
match the cache exactly. These errors measure the full edit, rather than the
individual projection deltas used in the original fitting report:

| Distill alpha | Equivalent old strength | Relative full-edit MSE | Edit cosine | Edit RMS / teacher RMS |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 0.8255 | 0.4235 | 0.3534 |
| 16 | 2 | 0.5374 | 0.6877 | 0.7899 |
| **24** | **3** | **0.4301** | **0.8205** | **1.1418** |
| 40 | 5 | 1.0923 | 0.6596 | 1.3859 |
| 64 | 8 | 1.6605 | 0.5160 | 1.4787 |

Alpha 24 is the closest of these tested settings by full-edit error. Matched
768px/10-step renders also recover the forward lean and strong smile; alpha 16
remains mild and alpha 40/64 exaggerate eyes, darkness and facial geometry. The
male case remains illustrated rather than reproducing the teacher's rendered
medium. This is a useful strength calibration, not an exact distillation or a
final-test quality result. No learned matrix or training formulation changed.

`release_tools/audit_distill.py` reproduces the measurement and image sweep under
the pinned host launcher with an exclusive `ANIMA_GPU_LEASE`. To derive either
ordinary-LoRA format from its original alpha-bearing file:

```bash
python release_tools/rescale_lora_alpha.py \
  artifacts/release/distilled/comfyui/bad-intent.safetensors \
  artifacts/bad-intent-alpha24/comfyui/bad-intent-alpha24.safetensors --gain 3
```

The exporter preserves learned matrices and records the source hash. Verification
found unchanged learned tensors, matching native/ComfyUI matrices and alpha for
all 224 projections, and exact native-hook equality between old strength 3 and
new strength 1. Both original strength-1 images replayed pixel-for-pixel under
the verified GPU launcher. The initial candidate samples were equivalent copies
of the sweep renders; the current public gallery was regenerated directly from
the new files at nominal strength 1. See
[`validation/bad-intent-alpha24.json`](validation/bad-intent-alpha24.json) for the
hashes, measurements and limitations.

## Calibrated particles and lighting LoRAs

Current Candlelit particles embed alpha 16 and Moonlit particles embed alpha
23.08072421821097, both at rank 8. These are the exact previously selected Studio
exports: nominal strength 1 applies gains 2 and 2.8850905272763714 relative to
the original particles. Bad Intent's current particle alpha is 12. There is no extra
UI multiplier. Every learned tensor is unchanged from the original EMA export.

Lighting LoRAs carry the same alpha gain as the corresponding particle teacher.
This is a portable strength adjustment, not a new regression fit or a claim of
exact teacher matching. Bad Intent's distinct LoRA alpha 24 comes from the
complete-denoiser sweep above. All current files and examples use strength 1 as
the default. The older strength-3/5 examples remain historical artifacts.

The gallery contains 12 matched comparisons: both Bad Intent development cases,
four original development prompts plus the featured portrait for each lighting
slider. All 36 Particle/Distill/Off images were regenerated from the selected
exports. [Unit-alpha verification](validation/unit-alpha.json) records the exact
file hashes, embedded alphas, unchanged learned weights, sample settings and
pixel-exact Off replays. Full ComfyUI GPU generation remains outside this audit.

## Matching Bad Intent's visual intensity

The first publication's alpha-24 LoRA looked more intense than its alpha-8
particle reference. A follow-up sweep compared particle alphas 8, 10, 12 and 16
with LoRA alphas 20, 22 and 24 on the same two development prompts/seeds, plus
the same 40 shared neutral states used above. The selected defaults are now
**particle alpha 12, LoRA alpha 24**, both rank 8 at nominal strength **1.0**.

The particle's forward pose and grin become stronger in both examples. Alpha 16
pushes head tilt, grin and the male rendering-medium change further. Alpha 12
was chosen for visual intensity, preserving the already preferred LoRA effect.
The particle's 50% alpha increase does not imply a 50% whole-image or velocity
change, since the complete denoiser remains nonlinear.

| Particle alpha | LoRA alpha | LoRA / particle edit RMS | Edit cosine | Relative full-edit MSE |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 24 | 1.1418 | 0.8205 | 0.4301 |
| 10 | 24 | 1.0278 | 0.8706 | 0.2669 |
| **12** | **24** | **0.9333** | **0.8533** | **0.2783** |
| 16 | 24 | 0.8638 | 0.8122 | 0.3429 |

Alpha 10 is the closest numerical amplitude/error match in this subset; 12 is
the visual selection. The selected edit amplitudes differ by about 7% on these
states. These measurements do not prove equal perceived strength or that the
original must look better than its distill. Both images remain approximations
of an intended effect, and the male particle still differs in medium.

The current particle download is `weights/bad-intent-balanced-alpha.safetensors`.
It contains unchanged learned tensors from the original step-1600 EMA; only
alpha and provenance differ. The LoRA files are unchanged. The new examples
under `samples/balanced-alpha/bad-intent/` were rendered directly from these
files at nominal 1. Original alpha-8 and first calibrated samples remain
available at their original paths. See the reproducible
[`data/bad-intent-balance.json`](data/bad-intent-balance.json) selection and
[`validation/bad-intent-balance.json`](validation/bad-intent-balance.json) audit.
