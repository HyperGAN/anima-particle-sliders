# Ordinary rank-8 LoRA distillation

The native particle adapters remain the reference. The ordinary LoRAs replace each nonlinear branch with two linear matrices so they can use standard ComfyUI **Load LoRA**: MODEL strength controls the edit; CLIP strength is **0**.

For each particle branch, keep the learned up matrix $U$ fixed and fit a new down matrix $A$ to the routed bottleneck output $h(x)=F([W_{down}x,z(x)])$ on actual Anima projection inputs. For input and target row matrices $X,H$:

$$
A^\top=X^\top(XX^\top+\lambda I)^{-1}H,\qquad
\lambda=0.01\,\operatorname{mean}\operatorname{diag}(XX^\top).
$$

The distilled delta is $UAx$, rank eight, with alpha eight. It has no particle cloud, router, activation, or learned bias. This is local activation regression, not diffusion-step acceleration: the base is already Turbo and still uses the same 10-step sample schedule.

Calibration uses all 24 training rows, the first seed per row, five positions (neutral-path states at 0, 2, 4, 6 and positive-path states at 9), and eight evenly spaced token activations per projection. The native particle teacher runs at strength one on those cached states. That gives 960 inputs per projection. The ridge coefficient is fixed before development evaluation.

Held-out measurements use development characters on positive-path states at positions 1 and 5 and neutral-path states at position 8. Relative projection MSE is the sum of squared output-delta errors divided by the sum of squared particle output deltas across all projections. It is an internal approximation diagnostic, not an image-quality score. The public image comparisons show the original particles and distills on identical prompts and seeds at strengths 1, 3, and 5. Drift compounds through the denoiser, especially above the trained strength.

`release_tools/distill.py` calls `concept_slider_core.fit_routed_down` from the [pinned shared core](https://github.com/mikkel/sliders-conceptmod/tree/beaffeb3640c4554a7315998c04a5909f384b972/packages/concept-slider-core), exports both native and ComfyUI tensor names, and renders the comparisons. The exporter does not fuse Q/K/V because ComfyUI Anima exposes separate projections. `release_tools/verify_comfy.py` checks all 224 mappings against actual ComfyUI modules and verifies native/ComfyUI matrix equality.

The featured renders show a milder lighting change from the distills at equal strength. Small aggregate projection errors do not guarantee matched images, and individual projection errors can be much larger. The complete per-projection measurements are included with each student.
