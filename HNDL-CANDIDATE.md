# Anima HyperGAN/HNDL candidate

This separate source cohort uses the actual shared core0.12/PG0.9.4 game and
HyperGAN develop/HNDL0.10 builder, for both the native routed adapter and the
conditional patch critic. It retains Anima's genuine five-dimensional velocity
fields at the public ParticleGAN penalty boundary. There is no new loss,
optimizer, copied GAN loop or continuous-domain admission bypass.

Bounded installed CPU results and remaining limits: [HNDL-CPU.md](HNDL-CPU.md).

The root `pyproject.toml` remains the known CPU-tested core0.8.1/PG0.9.1 cohort.
`cohorts/hypergan-hndl/pyproject.toml` is a distinct candidate build manifest:
Anima0.3.0, exact shared source
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, official PG0.9.4, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10.0. That shared source is
local and unpublished at this writing; ordinary remote dependency resolution is
blocked until the core owner publishes the reviewed commit. Do not substitute
another SHA, install this over an active run, or combine the two environments.

Build the candidate locally in its own environment with patched setuptools:

```sh
python cohorts/hypergan-hndl/build.py --out /absolute/path/to/candidate-wheels
```

The helper stages a temporary source tree and uses the separate manifest; it
does not edit the root manifest. For local qualification only, install the exact
joint core wheel and its pinned framework dependencies, then install the built
product wheel with `--no-deps`. This is not a full dependency-resolution receipt.
Use backend `anima-hndl` with the same local cache/model options as `anima`.
Discovery registers only `anima-hndl` in this candidate wheel. Its native-only
inference backend has a distinct protocol and no teacher-cache requirement.

The shared routed graph is reused directly from `particle_sliders`; the product
critic is a declarative HNDL version of the existing local-patch architecture.
The original patch extraction stays inside the critic, while the game receives
unaltered `[1,C,1,H,W]` fields. The shared builder verifies exact HyperGAN build
provenance and the HNDL source closure before attachment. Profile/source/graph
identities enter the shared checkpoint manifest. Historical packets and native
exports are not converted or relabeled; use their original environments.

CPU fixtures use a tiny random Cosmos model without pretrained loaders. Native
pretrained/GPU execution, sampler/decoder parity, HNDL performance at production
resolution, and model-weight publication remain unqualified. CPU construction
and response tests do not open accelerator admission; GPU0 remains reserved.
