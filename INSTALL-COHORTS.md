# Installation cohorts

The current packaged lifecycle uses `particle-sliders-core` 0.8.1 at
`32f6830f6402e85aa77b6631db64fcbbb2131f6b` with exact official ParticleGAN `0.9.1`.
Install from this repository root with `python -m pip install -r requirements.txt`.
The requirements entry delegates to `pyproject.toml`, so the two install paths
resolve the same immutable shared core. No unpublished PyPI core release is assumed.

`requirements-legacy.txt` preserves the previous installer verbatim for historical
trainer reproduction in a separate environment. Existing source locks, model
identities, checkpoints and receipts retain their original meaning. Do not combine
legacy requirements and the packaged lifecycle in one environment, or repoint a
historical lock to newer bytes. The product runtime still verifies its own
canonical source map; a successful installation alone does not qualify execution.

This repairs installation consistency. It does not adopt core0.11.0/PG0.9.3 or
PG0.9.4, migrate checkpoints, qualify pretrained/GPU output, or enable refused
continuous-response/render operations. Those require distinct reviewed cohorts.

The native extra declares torchvision, required by the pinned Cosmos model
implementation. CPU fixture verification pairs torch2.14.1+cpu with
torchvision0.29.1+cpu; native pretrained execution remains unqualified.

Keep this PG0.9.1 environment separate from core0.11.0/PG0.9.3 and
HyperGAN develop, which currently requires PG0.9.4/HNDL0.10.
These Git dependency wheels are for source installation, not a PyPI release.

The historical Comfy ZIP and Hub release intentionally expose the preserved
legacy bytes as both requirements.txt and requirements-legacy.txt. The source
checkout requirements.txt instead installs the packaged lifecycle.

Source distributions preserve these requirements, source locks, cohort receipts,
CPU fixtures and historical release helpers. After extracting a source archive,
run the current installer from its extracted root, exactly as for a checkout.
The wheel contains runtime code and configuration; it does not install the
repository's release/publication commands as executable entry points.

## Preparing a later shared-core cohort

The separate [Anima HyperGAN/HNDL candidate](HNDL-CANDIDATE.md) implements the
bounded product binding and build path. It does not change the default installer.

The jointly tested local shared candidate is core 0.12.0 at
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, with exact PG 0.9.4 and HyperGAN
develop `5f368a3770cf5bdae0448995c9f76e29af8049f7` / HNDL 0.10.0. It is
unpublished and has not qualified Anima's product factory, pretrained execution
or image rendering. Keep the current default until a separately reviewed Anima
cohort passes its own installed-package checks.

Prepare that adoption in another environment and branch. Record the exact core
source manifest and wheel provenance, then run the installed product's cache,
configuration, native adapter gradient ownership, zero-strength, raw/EMA,
checkpoint rejection and exact restoration checks against it. The shared width
admission remains authoritative: an unsupported native field must fail before
training, without reshaping it into a different domain or substituting a loss.
CPU synthetic fixtures do not admit pretrained or GPU work.

Any adopted source/profile change needs a new cohort and new checkpoint
provenance. Keep old locks and receipts byte-identical, verify rejection without
state mutation, and reproduce old checkpoints only in their recorded environment.
Do not relabel historical artifacts or change the compatibility import shim
`lumen_studio/vendor/reference.py` into a second mathematical implementation.
