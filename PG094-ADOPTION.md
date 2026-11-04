# Current Anima source: ParticleGAN 0.9.4

The default Anima **0.3.0** installer now selects shared core **0.12.0** at
`8ffca928550f97fdb9713151ca3023fc71e77e2c`, which requires official
**particlegan==0.9.4**. Both `pip install .` and
`pip install -r requirements.txt` resolve that exact public Git commit.
Python >=3.11 and setuptools >=84.2.1 remain required; the native extra now
bounds Pillow to >=13,<14, matching the current upstream image dependency.

The existing callable `lumen_studio.pg09_training.PG09Trainer` and registered
`anima` lifecycle use the same shared `ConditionalPG09` entrypoint. At this
core version it is the official 0.9.4 engine, also exported as
`particle_sliders.pg094.ConditionalPG094`. There is no product-local replacement
for the loss, penalty, prior or optimizer. Existing source wiring passes the
actual 128x4 trainable routed bank and native `[1,C,1,H,W]` fields. It selects
non-saturating BISPF, SBNP, zero-momentum DualNorm and constant effective learning
rates. The native projection and product critic remain unchanged.

## Verification, 2026-11-04

A fresh Python 3.12 venv, without system site-packages, installed the declared
native dependency set from public sources, with Torch 2.6.0+cpu and torchvision
0.21.0+cpu. Tests used one CPU thread and no pretrained weights or GPU work.

- **73 installed conformance checks passed**, run outside the checkout with
  isolated Python: actual tiny random Cosmos, routed-bank and branch updates,
  official recipe identity, exact split resume, raw/EMA native exports,
  zero/fractional-strength responses, optimizer-history rejection, frozen
  cache/source evidence, gradient ownership, cleanup and unsupported-device
  refusals.
- **111 packaging, release and compatibility checks passed**: wheel and sdist
  reconstruction, exact default dependency metadata, preserved HNDL candidate
  source closure, historical Comfy installer bytes, gmix identity and released
  checkpoint loading contracts.
- `uv pip check` passed. The installed core's PEP610 metadata was checked
  against the default manifest's exact Git commit; the shared engine verifies
  the installed ParticleGAN source bytes before construction.

Reproduce from a fresh environment after installation:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /absolute/new-env/bin/python -I -m pytest -c /dev/null \
  /absolute/checkout/tests/lumen_studio/test_pg09_training.py \
  /absolute/checkout/tests/lumen_studio/test_pg09_consumer_history.py \
  /absolute/checkout/tests/lumen_studio/test_image_lifecycle.py \
  /absolute/checkout/tests/lumen_studio/test_image_lifecycle081.py \
  /absolute/checkout/tests/test_current_installation.py -q
```

Run this from an empty directory outside the checkout so product imports come
from the installed wheel. The `081` test filename identifies the regression
suite's origin; its assertions now verify the current PG0.9.4 packets.

## Artifact and qualification boundaries

Start a new environment and a new run directory. Old source-bound training
packets are rejected by the shared identity contract; this change does not
convert their optimizer, recipe or source identity. The six published sliders
remain historical gmix weights. Their files, source locks, loading code and
release metadata are unchanged.

`requirements-legacy.txt` / `core.lock.json` preserve core0.3.0.
`requirements-pg09.txt` / `core.pg09.lock.json` preserve the original
core0.6.0/PG0.9.1 candidate. Their package-origin tests only run in that original
core0.6.0 environment; the current installation has its own active test.
The separate HyperGAN/HNDL 0.3.0a2 profile retains core a794 and its exact source
closure. It already uses PG0.9.4 and is not silently rebound to this default
package. Its native-extra Pillow bound is kept consistent with the default.

CPU source qualification does not establish pretrained training, GPU memory,
image rendering or useful model quality. Lifecycle accelerator/render refusals,
the explicit trainer's external native-admission callback, immutable cache
requirements and the original model/environment lock remain in force. The
full native environment lock still differs from the optional dependency set,
including its historical Pillow version. The older capability descriptions
retain their original qualification labels; this document and installed
package identities record the new source cohort without upgrading those
native qualification claims. No package/model publication or deployment is
part of this source update.
