# Independent public-source installation — 2026-10-31

Follow-up to [HNDL-CPU.md](HNDL-CPU.md). Exact shared core
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` is now publicly reachable.

A fresh Python3.12.15 venv physically installed every runtime dependency using
public PyPI, no dependency cache, ordinary isolated VCS builds and official CPU
Torch2.6.0/torchvision0.21.0 wheels. This independently tests the shared Torch
lower bound without changing any default or historical environment. Relevant
imports are all inside this new venv; there is no shared site-packages path,
dependency symlink or dependency-view `.pth`. Only the normal base interpreter
and standard library are shared by the venv.

The experimental candidate0.3 wheel's declared dependency resolved public core4abc0eb and its
`hypergan-hndl` extra normally. Pip reports and installed PEP610 metadata record
that exact core commit and HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`. Core0.12.0, official PG0.9.4 and
HNDL0.10.0 installed; the strict builder accepted the actual HNDL source closure
and clean HyperGAN build provenance.

The declared `native` extra also resolved and installed: Accelerate1.14.0,
Diffusers source `d5baa4fb548294f47dbca49890abd4b291204c60`, Transformers5.15.0,
Hub1.27.0, Pillow13.0.0 and matched CPU torchvision. Both base and native-extra
`pip check` calls reported no broken requirements. This is a real dependency
resolution/install result, not a `--no-deps` or preexisting-library check.

11 installed tests passed: the nine actual HNDL candidate contracts plus two
real local sharded-component loads with affected Accelerate APIs trapped.
Fixture files were copied byte-for-byte outside the source checkout to avoid
egg-info shadowing. No pretrained weights, Hub model downloads or GPU were used.

The review follow-up is an explicit0.3.0a1 prerelease in the `anima_hndl`
namespace with runtime environment admission and complete source-closure
admission. Isolation remains an operator installation requirement: the two
wheels share a distribution name, and pip can replace one with the other. The
runtime checks selected package roots after Python startup, not all imports or
arbitrary PYTHONPATH execution. It was installed with all native extras into a second completely
fresh venv using normal dependency resolution; 17 installed tests and `pip check`
pass there. The first eleven-check result above preserves the initial source lane.
The final lane additionally tests actual core/product file tampering, wrong Git
provenance, shared-environment refusal, post-load cache validation, fixed H/W and
the native inference adapter's actual parameter identities.

Official downloaded CPU artifact SHA256 values:

| Artifact | SHA256 |
| --- | --- |
| Torch2.6.0+cpu/cp312 Linux x86_64 | `59e78aa0c690f70734e42670036d6b541930b8eabbaa18d94e090abf14cc4d91` |
| Torchvision0.21.0+cpu/cp312 Linux x86_64 | `d6874431e678ba107b60a83f255c33f3755f06bad587b1b919aa514ec325dcd8` |

Both installs used `python -I -m pip install --isolated --index-url
https://pypi.org/simple --no-cache-dir --report ...`. The first selected the
candidate wheel, pytest and the hash-pinned official CPU Torch URL; the second
selected that wheel's `[native]` extra and the hash-pinned matching torchvision
URL. The review packet contains the exact commands, full install reports/logs,
package freeze, import paths, VCS metadata, source digests and test results.

This proves clean base/native-extra installation on this Python3.12/Linux CPU
combination. It does not reproduce the much larger original GPU model lock in
`configs/anima/requirements.lock`: `verify_environment` still refuses this CPU
environment, first for missing historical `annotated-types`. That lock includes
studio/CUDA and other exact dependencies outside the optional model extra.
There is also an unresolved declaration conflict: the candidate native extra
requires Pillow>=13, while the inherited full native environment lock requires
Pillow==12.3.0. No environment can satisfy both. Native owned-model execution
therefore remains blocked until a separate qualified native cohort resolves
this conflict. Neither lock nor any prior source receipt was relaxed or relabeled.

Pretrained/GPU admission, production-scale HNDL performance and sampler/decoder
parity remain pending. Accelerate is not claimed patched; the tested API
reachability and bounded mitigation are in [ACCELERATE-ADVISORY.md](ACCELERATE-ADVISORY.md).

The subsequent return-review changes reuse that physically isolated dependency
environment and reinstall the new candidate wheel with `--no-deps`; this is a
new source/artifact test, not another fresh dependency-resolution claim. The
original full resolver reports remain available in the review packet.
