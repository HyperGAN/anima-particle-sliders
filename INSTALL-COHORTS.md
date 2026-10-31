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
