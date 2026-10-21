# Anima image lifecycle migration: source audit and bounded plan

Review base: open draft PR5, `4477716c8dbedd3a079e3299a1229adaaa2b407f`,
branch `audit/anima-shared-pg09-binding-20261017`. GitHub read on 2026-10-21
reported this as the only open PR. Inventory main
`8bcf7f2cd3d08933556f79b87cb1519e7cecbe8d` is not the migration base.
This isolated branch preserves the entire PR5 payload. No public ref changes.

Common interface: `fd89e1c780ab5210cd1be98ec3b70977e6349379`, core 0.7.0.
The exact local copy is `../core-milestone`; the shared owner's checkout is
read-only. Read `IMAGE-LIFECYCLE.md`, `derivative-integration-handoff.md` and
`hypergan-interface-seams.md` at that commit. This is a local stacked milestone,
not a main integration, publication, dependency-adoption or runtime approval.

## Entrypoints actually inspected

| Area | Source and behavior at PR5 | Migration disposition |
| --- | --- | --- |
| Loading | `runtime.create_runtime` dispatches to `backends.anima.TurboRuntime`; verifies `anima-lock.json`, staged file hashes and exact environment, loads local ModularPipeline components in BF16, freezes backbone/conditioners/VAE | Reuse local-only loading in the new owned CPU factory; no loading executed here |
| Teacher capture | `cache.prepare_targets`: CFG1, 10 native Euler steps, both frozen paths, paired velocities at each latent/time, frozen neutral embedding; `TargetCache` reads CPU shards | Consume an existing cache only; no target preparation, relabeling or migration |
| Actual PG training | `pg09_training.PG09Trainer`: immutable `CachedResponses`, actual `ParticleAdapter` bank and `AnimaRuntimeResponseAdapter`, `NativeFieldCritic`, installed `ConditionalPG09` with balanced `bispf_optimized` | Reuse these definitions in a complete lifecycle factory; keep existing explicit API |
| Resume | `PG09Trainer` automatically restores `pg09-resume.pt` from its directory with shared game state validation | New sessions use only explicit v1 resume; never auto-discover legacy state |
| Response evaluation | `trainer.game.evaluate` exposes native velocity responses | New `session.evaluate` retains `[1,C,1,H,W]`; does not claim images |
| Export | `PG09Trainer.export` writes immutable native safetensors plus `.pg09.json`, live weights, exact runtime/cache/manifest identities | Preserve native format; new `session.export` is the separate v1 raw/EMA inference packet |
| Inference | `TurboRuntime.render`, Studio `Coordinator.render_one`, `release_tools/infer.py`, Comfy hooks and `single_particle.adapter_class` load selected native/alpha/single-particle/LoRA formats | Preserve current dispatch and artifacts; new v1 native render fails closed pending parity qualification |
| Legacy train/API/worker | CLI train/benchmark, release training, campaign, benchmark/preflight and `/api/training/runs` refuse early; worker rejects old jobs under lease and preserves image work | Keep PR5 safeguards byte-for-byte; no automatic PG routing |
| Other fitting | release distillation/expansion uses closed-form ordinary-LoRA approximation | Separate workflow, not migrated to GAN sessions |

## Native laws and parameter ownership

Anima Turbo v1.1 is a 5D continuous velocity field with a singleton time plane.
`predict` divides native scheduler timesteps by `num_train_timesteps`. Training
conditions are mean neutral token embeddings plus normalized time, position/9,
frozen trajectory choice and scale +1. Row/display IDs and seeds are excluded.
`CachedResponses` verifies and owns cache index/order, shard bytes and native
runtime identity, including explicit historical compatibility evidence when
applicable. Do not fabricate that evidence.

The generator arm is `routed_particle_gan`: the actual 128x4 FP32
`ParticleAdapter.particles` is both the hooked bank and shared prior parameter.
Native attention branches retain rank 8 and existing target names. The frozen
BF16 backbone is excluded from the trainables. The native patch critic uses
rank-two Linear weights with raw logits; retain that geometry/optimizer law.
Training learns the +1 endpoint; bounded public response strengths are [0,1].
Reference images and native rendering are unsupported in this slice.

## Bounded implementation

1. Add independent installable metadata and `particle_sliders.image_backends`
   registration, product -> core at exact milestone -> ParticleGAN. Keep both
   historical requirements/locks untouched. Package the existing config bytes
   at their original relative paths for installed runtime provenance.
2. Validate finite options, CPU/BF16 policy and local paths before model
   allocation. Factory may receive an already staged CPU runtime, or load a
   local model itself; caller never constructs a game/provider/critic/optimizer.
3. Reuse existing provider, bank, critic and core backend/game. Use common EMA,
   stepping, observation, packet/restore/export and CLI. Capture checkpoint
   settings before binding and remove only owned hooks/adapters on close or
   partial failure. Keep borrowed runtime/mixes intact.
4. Do not introduce a run UI. HyperGAN consumes session `on_step`, manifest and
   packet receipts. Native sampling belongs in a product callback after parity.
   Inventory custom `NativeFieldCritic`, `Branch` and the routed bridge for the
   separate HNDL matched-parameter/PG-update parity migration.

## Core rough spots and gates

- Core `Binding` has no pre-update native admission hook. Existing Anima
  requires a coordinator decision over exact model/cache/teacher parity before
  the first non-CPU update. Reject non-CPU factory configuration and staged
  runtimes until the parent resolves this seam. Do not embed a custom game loop
  or silently bypass admission.
- Compact inference loading at this milestone rebuilds game/cache. Keep that
  limitation explicit; wait for the core owner's reviewed interface update.
- Native sampler/decoder parity and pretrained memory/quality remain unqualified.
  Rendering is gated before factory/model allocation.
- HNDL conversion is separately parity-tested work. Existing raw-torch network
  definitions are inventoried, not declared migrated.
- Core publication/main integration and exact installed package review remain
  unresolved. The new proposed dependency is not an active environment pin.

## Deferred verification (parent must schedule)

No tests, dependency installs, model loads or builds during this source-only
phase. Later run one bounded installed-wheel CPU job outside all source trees:
actual native tiny Cosmos/bank update, exact split resume with metrics/EMA/RNG/
coverage, invalid restore before mutation, raw/EMA export/load, exact zero and
fractional responses, signed/reference/render/device rejection before loading,
observation state/RNG preservation, setup/update failure cleanup, borrowed
runtime mix/hook/checkpoint restoration, installed source origins and dependency
direction guard. Use CUDA hidden, HF offline and one thread; no model downloads.
Archive hashes, wheels and command receipts only after that scheduled phase.
