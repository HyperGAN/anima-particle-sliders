# Active modern cohort: exact core a794

Only the opt-in Anima candidate changes: product 0.3.0a2, profile/protocol v3,
core 0.12.0 at `a794e3b4a8813b012f2c4faa3da82b78d4a90214`, official PG 0.9.4,
HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL 0.10.0.
The actual routed adapter, declarative patch critic and shared game remain in
use; architecture, objectives and native admission are unchanged. The default
0.2.0/core 0.8.1/PG 0.9.1 environment and historical locks remain unchanged.

`anima_hndl/cohort.json` recursively binds all 50 canonical core source/data files
from that exact Git tree and all 65 installed product source/data files, including
nested configuration. The canonical core closure digest is
`6e0ed435f259c104bed9d18b4370040a4e4b9d3a71953b7079116272c2524296`.
Actual PEP610 Git identity and imported origins are checked, not only 0.12.0.
The prior core 4abc wheel's digest is removed from admission: the current qualified
wheel list is empty. Do not regenerate the manifest to admit altered sources.
The changed product closure enters adapter/critic/native checkpoint identity.

## Reproduce artifact installation

Use separate build and runtime venvs. Both candidates share a distribution name
with stable, so pip can overwrite it; dedicated installation is an operator
requirement plus runtime admission, not an install-time security boundary.
From a trusted checkout, in a build venv with setuptools>=84.2.1:

```sh
/absolute/build-env/bin/python -I cohorts/hypergan-hndl/build.py \
  --format both --out /absolute/artifacts
```

This produces `anima_particle_sliders-0.3.0a2-py3-none-any.whl` and
`anima_particle_sliders-0.3.0a2.tar.gz` without editing the stable manifest.
Create a new runtime venv without shared/system site-packages, change to an empty
directory outside the checkout, and resolve the wheel normally:

```sh
/absolute/runtime-env/bin/python -I -m pip install --isolated \
  --index-url https://pypi.org/simple --no-cache-dir --report install-report.json \
  'torch @ https://download-r2.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-linux_x86_64.whl#sha256=59e78aa0c690f70734e42670036d6b541930b8eabbaa18d94e090abf14cc4d91' \
  'torchvision @ https://download-r2.pytorch.org/whl/cpu/torchvision-0.21.0%2Bcpu-cp312-cp312-linux_x86_64.whl#sha256=d6874431e678ba107b60a83f255c33f3755f06bad587b1b919aa514ec325dcd8' \
  '/absolute/artifacts/anima_particle_sliders-0.3.0a2-py3-none-any.whl[native]' \
  'pytest>=8' 'setuptools>=84.2.1'
/absolute/runtime-env/bin/python -I -m pip check
```

The candidate's declared dependency selects exact core a794 and its real
HyperGAN/HNDL integration. The fresh CPU install passed normal public-source
resolution without shared dependencies or `--no-deps`. Import origins, Git
metadata and all canonical bytes were independently checked. The CPU environment
does not contain pretrained weights and is not the full native model lock.

To test the sdist independently from the wheel's product installation, uninstall
only this candidate product in the same isolated dependency environment and
install the candidate sdist with ordinary dependency resolution. With the already
qualified build backend, `--no-build-isolation` is sufficient; the actual source
build still happens from the archive in a temporary directory, without a checkout.
Then repeat `pip check`, source admission and installed conformance. Do not use the
root stable sdist as though it were this candidate sdist.

## Artifacts and qualification

Profile/protocol v3 deliberately does not rebind0.3.0a1/core 4abc v1/v2 previews.
Keep the original environment and original packet bytes. The committed probe:

```sh
python cohorts/hypergan-hndl/check_legacy_packet.py \
  --legacy-python /absolute/stable-env/bin/python \
  --preview-python /absolute/original-a1-env/bin/python \
  --candidate-python /absolute/runtime-env/bin/python \
  --workdir /absolute/new-probe-directory
```

creates genuine two-update stable and original-preview packets. It calls
`read_packet` before `session.restore` and requires the exact source/cohort
refusal while preserving the current complete session state. Original-preview
raw/EMA native packets refuse at the factory/source guard before loading; their
bytes remain unchanged. Correction: earlier receipts passed a path directly to
`session.restore`, which established an input-type error rather than genuine
packet rejection. Those earlier rejection claims are superseded by this probe.

Installed tests cover exact split resume, raw/EMA native exports, OFF/fractional
response parity, actual native adapter parameter ownership, source/data tamper,
strict local loader arguments and model/cache refusals. An injected update error
closes and cleans the session; a fresh session resumes the saved valid packet and
retries to the exact expected result. Failed native observation restores modes,
gradient flags, transient buffers and RNG, and can be retried in the open session.

Prior review notes, [advisory limits](ACCELERATE-ADVISORY.md), immutable-input
requirements and fixed H/W remain in force. No verified official Accelerate patch
was found; pins are unchanged. The inherited full model environment still refuses
this CPU install, including Pillow==12.3.0 versus native-extra Pillow>=13. Shared
synthetic software qualification does not admit Anima pretrained/GPU execution,
rendering, new widths or quality claims. No GPU jobs, weights, release, deployment
or HNDL optimization is performed by this migration.
