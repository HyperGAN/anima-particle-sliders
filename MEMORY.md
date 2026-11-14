# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Train 768px default no checkpointing | 1 | 72 | 24 | 768×768; ≤512 tokens; B=2; BF16 frozen; FP32 adapter/critic/activation envelope; no checkpointing |
| Train 768px explicit checkpointing | 1 | 20 | 24 | 768×768; ≤512 tokens; B=2; BF16 frozen; FP32 adapter/critic/activation envelope; checkpointed |
| Train 768px dense fallback | 1 | 204 | 24 | 768×768; ≤512 tokens; B=2; BF16 frozen; FP32 adapter/critic/activation envelope; no checkpointing |
| Render 768px | 1 | 20 | 24 | 768×768; ≤512 tokens; B=1; BF16 frozen; FP32 adapter/critic/activation envelope; no checkpointing |
| Render 1536px | 1 | 44 | 24 | 1536×1536; ≤512 tokens; B=1; BF16 frozen; FP32 adapter/critic/activation envelope; no checkpointing |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. They describe the finite profiles above, one worker and one learned slider, with RAW weights and **no model-weight EMA**. A larger image, token sequence, image/reference count, logical batch, cache or number of simultaneous workers needs a new calculation; there is no finite universal maximum for unbounded inputs.

## Scope and residency

Base: **`circlestone-labs/Anima@f973fc41ec7545364ac9776c2440285f43ff2a30`**. TurboRuntime loads transformer, Qwen0.6B encoder, conditioner and VAE together on one GPU. The default constructor has checkpointing=False; checkpointing=True is an explicit supported runtime option. One learned adapter is included; simultaneous additional mixer adapters add their own state and graphs.

The 1536 render is the runtime maximum dimension, with one image and ≤512 encoded tokens. Larger request collections, arbitrary batch sizes or extra adapters are outside these rows. No quantized loader is selected by this package.

The table's efficient-SDPA rows require a flash/efficient implementation. An automatic backend can fall back to dense math; use the dense row where provided, or add the dense term below before planning capacity. A planning row does not change code, precision, optimizer, rates, initialization, checkpoints or qualification.

## Frozen and trainable ledger

- split_files/diffusion_models/anima-turbo-v1.1.safetensors: **4,182,230,656 bytes** (3.895 GiB).
- split_files/text_encoders/qwen_3_06b_base.safetensors: **1,192,135,096 bytes** (1.110 GiB).
- split_files/vae/qwen_image_vae.safetensors: **253,806,246 bytes** (0.236 GiB).

The turbo source DiT file also contains its conditioner. The three exact original file sizes bound all four loaded BF16 components; do not add the Base text_conditioner file a second time. The captured Base Diffusers Cosmos config supplies the matching 28-layer/2048-width topology, not a claim that Base and Turbo weights match.

Trainable allowance: **8,373,153 parameters**. 28 × [8×(4×4096+2×4096+2×3072) + 8×6476] +512 bank +40,737 critic = 8,373,153. FP32 trainable weights and gradients require `4A+4A` bytes. Frozen base parameters have no gradient or optimizer tensor allocation. Routed particle-bank parameters, where present, are counted once; a frozen response cache is not a trainable bank.

The current `bispf_optimized` PG0.9.5 recipe uses **DualNorm, momentum 0, direction_blend**; it is not assumed to be Adam. Persistent per-parameter moment tensors are0 in this profile; scalar counters are small. The peak ledger additionally reserves **`48A+32MiB` bytes** for protected normals, displacement/direction copies, FP64 promotion and thin-matrix SVD/Gram workspace. This is a conservative temporary-allocation allowance, not a claim that the optimizer uses no memory. Selecting historical `bispf`/Adam instead adds its actual state (typically another`8A` bytes for two FP32 moments); nonzero momentum or a different solver requires recalculation. Weight EMA adds0 bytes and must stay disabled.

## Numeric peak ledger (GiB before headroom)

| Profile | Frozen GPU weights | Activation / decode | Trainable + optimizer | Other GPU allowance | Host loading/cache subtotal |
|---|---:|---:|---:|---:|---:|
| Train 768px default no checkpointing | 5.242 | 48.125 | 0.468 | 0.125 | 15.378 |
| Train 768px explicit checkpointing | 5.242 | 8.250 | 0.468 | 0.125 | 15.378 |
| Train 768px dense fallback | 5.242 | 154.000 | 0.468 | 0.125 | 15.378 |
| Render 768px | 5.242 | 6.750 | 0.031 | 0.125 | 15.378 |
| Render 1536px | 5.242 | 27.000 | 0.031 | 0.125 | 15.378 |

CPU rows place their active tensors in host RAM; their frozen/activation entries are not GPU residency. Sequential phase sub-ledgers are in the source-facts report.

Cache bound for the training profile: **128 rows**, each with three FP32 native fields, plus **256 cached prompts**. Tensor-only cache allowance: `226,492,416+268,435,456` bytes; masks/indices and serialization overhead remain covered by the explicit host subtotal and reserve. These bounds are inputs to this calculation, not new runtime limits.

## Activation, cache and host calculation

The auditable inputs and per-profile subtotals are in [source-facts.json](docs/memory/source-facts.json). For a transformer envelope, let `N=image tokens+bounded text tokens`, `D=hidden width`, `I=FFN width`, `H=attention heads`, `L=covered blocks`, and `B=simultaneously retained native graphs`:

- One-block inference allowance: `4B(16ND+6NI)` bytes. The4-byte coefficient deliberately covers FP32 intermediates even for BF16 weights;16 and6 allow Q/K/V/output, normalization/residual, gated FFN and adapter intermediates.
- Uncheckpointed training allowance: `4LB(16ND+6NI)`; checkpointed allowance: `4(4LBND+2B(16ND+6NI))`. Checkpointing saves layer inputs and recomputes block internals; it does not remove the frozen model, adapter or critic.
- Dense math fallback adds `8BH N²` bytes for inference and `16LBH N²` for uncheckpointed training, or `32BH N²` for the checkpointed current-block envelope. These are conservative materialized-score/probability allowances; flash/efficient SDPA avoids that quadratic storage. GQA expansion uses the query-head count for this bound.
- Decode allowance is separately based on bounded output/tile area and channel width; profile notes record the bound. Peak is the **maximum of sequential phases**, with the actual overlapping model weights added inside each phase. Sampling steps are sequential and do not multiply resident peak.
- Frozen CPU rows are bounded by the profile (three FP32 native fields per cached row, plus explicitly bounded prompt embeddings). Loader staging/serialization copies and the largest published shard are included in host subtotals. Host cache/page residency and kernel workspaces vary, so these are planning allowances, not an allocator theorem.

Final GPU budget is `ceil_to_4GiB(1.25×GPU subtotal+2GiB)`; host is `ceil_to_4GiB(1.25×host subtotal+4GiB)`. The explicit25% plus runtime reserve covers allocator fragmentation, CUDA context/libraries, kernel workspaces, small FP32 islands and Python/serialization overhead. These chosen reserve multipliers are explicit planning assumptions; architecture metadata alone cannot prove every allocator or fused-kernel peak. It is not borrowed from scheduler admission caps, card size or global free-memory floors. Do not add every row's peak together for sequential phases; parallel workers do add their simultaneous requirements.

This profile's text-state allowance is **58,720,256 bytes** (0.055 GiB): conservative returned text KV. It is already included in the phase or host allowance, not added twice. Text-only encoding has no generated-token loop. If the actual encoder returns KV state, it is explicitly included as `2×layers×B×KV_heads×tokens×head_dim×dtype_bytes`; cache-disabled encoders allocate no persistent autoregressive KV. Recurrent/linear-attention state is accounted separately where applicable. Arbitrary output-token generation and image-conditioned encoders are outside these profiles.

## Sources and reproducibility

- [configs/anima/model.lock.json](configs/anima/model.lock.json)
- [lumen_studio/backends/anima.py](lumen_studio/backends/anima.py)
- [lumen_studio/particles.py](lumen_studio/particles.py)
- [lumen_studio/pg09_training.py](lumen_studio/pg09_training.py)
- [Captured architecture, published file/index sizes and arithmetic](docs/memory/source-facts.json) retains exact model revisions, metadata SHA256s and direct Hugging Face config URLs. No weights were downloaded or opened for this calculation.
- Optimizer facts: the pinned ParticleGAN0.9.5 `recipes.py` (`bcap`/`bispf_optimized`, lines1435–1451), `optim/dualnorm.py` momentum/state branch, and `optim/direction_blend.py` FP64 normals. These are part of the dependency selected by the repository's existing core lock; their source facts are recorded in the combined calculation report.

Documentation was calculated from source and published metadata on 2026-11-14. No model construction, import of Torch/Diffusers, forward pass, benchmark, sampling, training or new GPU work was performed.
