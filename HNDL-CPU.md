# Anima inventory and HyperGAN/HNDL CPU receipt — 2026-10-31

Public base: `c018f32759f49a05c2360a04b4a83924a6851913`.
The existing installer repair `f636916501a33aa0b322ede76d26cb8d972b5f41`
is retained as an ancestor, without recreating its changes. No repository
AGENTS.md or repository-local skill instructions were present.

The default manifest remains Anima0.2.0/core0.8.1/PG0.9.1. Historical
requirements, both source locks, model requirements, checkpoint identity
acceptance and the shared import shim remain unchanged. The historical Comfy
ZIP retains the original requirements bytes under both filenames. The source
distribution now includes the real installers, cohort receipts, release helpers
and fixtures; both default and candidate wheels rebuild from that archive.

The separate Anima0.3.0 candidate manifest selects exact shared core
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` and its `hypergan-hndl` extra.
Actual installed core0.12.0, official PG0.9.4, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL0.10.0 execute the tests.
The shared verifier checks the complete declared framework/builder closure and
clean build provenance. The product uses the shared routed HNDL graph, a parity
tested declarative version of its existing patch critic, and the shared game.
Its native fields remain `[1,C,1,H,W]`; width/domain admission is unchanged.

Validation used isolated Python3.12.15 environments with read-only dependency
views of existing CPU fixture libraries, Torch2.14.1+cpu / torchvision0.29.1+cpu,
and pinned diffusers source `d5baa4fb548294f47dbca49890abd4b291204c60`.
Product wheels and the new core wheel were installed into their respective
environments; explicit assertions verify imported product/core wheel origins.
This is installed software-contract evidence, not clean native dependency
resolution. In particular, the fixture library view has Hub1.33.0 while the
native model cohort declares Hub1.27.0; no pretrained loader was called.

Passed checks:

- Stable installed runtime: 87 lifecycle, shared-PG consumer/history, ownership,
  zero-strength, restore, dispatch and cleanup cases, plus one installed-origin
  assertion. Default core0.8.1 explicitly refuses the opt-in HNDL profile.
- Source/archive/release regressions: 123 cases, including a real sdist build,
  both wheel rebuilds, unchanged installer/lock bytes and historical ZIP imports.
- Opt-in installed candidate: 9 cases. Three actual Anima branch shapes match
  outputs, input gradients and particle-bank gradients. Two patch geometries
  match critic outputs and first/second penalty derivatives. A real tiny random
  Cosmos session verifies actual shared updates, generator optimizer leaf
  ownership, bank gradient flow, frozen base, exact split continuation, raw/EMA
  native-only reload and exact zero/fractional response parity after cache removal.
  Builder tampering and accelerator requests fail before attachment/allocation.
- An actual two-update checkpoint created by the installed core0.8.1/PG0.9.1
  product is rejected by core0.12/PG0.9.4 with `unsupported session packet version;
  no automatic migration`; complete live checkpoint state remains unchanged.
- The new HNDL lane still rejects a flat continuous vector of width170.

Reproduction commands (run from an isolated checkout parent; replace paths):

```sh
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  STABLE_ENV/bin/python -I -m pytest -q \
  -o pythonpath=/absolute/repo/tests/lumen_studio \
  /absolute/repo/tests/lumen_studio/test_image_lifecycle.py \
  /absolute/repo/tests/lumen_studio/test_image_lifecycle081.py \
  /absolute/repo/tests/lumen_studio/test_pg09_training.py \
  /absolute/repo/tests/lumen_studio/test_pg09_consumer_history.py \
  /absolute/repo/tests/lumen_studio/test_legacy_dispatch_ordering.py
```

For the candidate, copy `tests/hndl/test_anima_hndl.py` and the three imported
`test_image_lifecycle*.py` / `test_pg09_training.py` helpers outside the checkout;
run only `test_anima_hndl.py` with `--import-mode=importlib` and an absolute
`pythonpath` to that fixture directory. This avoids checkout egg-info shadowing
the candidate wheel's entrypoints. Initial harness failures (fixture path,
metadata formatting and manifest lookup) were corrected; no passing result is
inferred from those failed runs.

Outstanding: shared core commit publication and coordinated review; full native
dependency resolution; pretrained/native GPU admission; production-resolution
HNDL performance; native sampler/decoder parity. There are no authored repository
CI workflows; GitHub's dynamic Dependency Graph is separate from these tests.
Nothing here merges, deploys, trains a pretrained model, or publishes weights or
packages. Keep this profile opt-in until its distinct cohort is reviewed.
