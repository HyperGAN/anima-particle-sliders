# Anima HyperGAN/HNDL candidate

This separate source cohort uses the actual shared core 0.12/PG 0.9.4 game and
HyperGAN develop/HNDL 0.10 builder, for both the native routed adapter and the
conditional patch critic. It retains Anima's genuine five-dimensional velocity
fields at the public ParticleGAN penalty boundary. There is no new loss,
optimizer, copied GAN loop or continuous-domain admission bypass.

Current migration and reproduction: [MODERN-CORE-A794.md](MODERN-CORE-A794.md).
The prior [HNDL-CPU.md](HNDL-CPU.md) receipt remains historical.
The later [independent clean install](CLEAN-INSTALL-CPU.md) resolves the public
sources without dependency sharing. See the [Accelerate API assessment](ACCELERATE-ADVISORY.md)
for the pinned native loader's scope and operating boundary.

The root `pyproject.toml` remains the known CPU-tested core 0.8.1/PG 0.9.1 cohort.
`cohorts/hypergan-hndl/pyproject.toml` is an explicit prerelease build manifest:
Anima 0.3.0a2, exact shared source
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`, official PG 0.9.4, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL 0.10.0. Those exact sources are
now public and the base/native-extra dependency installation has passed in a
fresh CPU environment. Do not substitute another SHA, install this over an active
run, or combine the two environments.

The operator must create a dedicated virtual environment without system/shared
site-packages. Runtime admission checks that requirement and selected installed
package roots; it cannot stop pip from overwriting a stable installation. Both
wheels use the same distribution name, so explicitly installing this prerelease
can replace stable. Never install it into a stable environment. It is not a final
release selected by an ordinary stable upgrade. Use a trusted environment and
`python -I`: these checks run after Python startup, cover only the named roots
in `source_validation.py`, and do not sandbox PYTHONPATH or every dependency. Stable 0.2.0 retains only the `anima` entrypoints and
contains no modern `anima_hndl` implementation. The prerelease installs that
separate namespace and registers only `anima-hndl`.

Build the candidate locally in its own environment with patched setuptools:

```sh
python cohorts/hypergan-hndl/build.py --format both --out /absolute/path/to/candidate-artifacts
```

The helper stages a temporary source tree and uses the separate manifest; it
does not edit the root manifest. It builds both the candidate wheel and candidate
sdist. Install either artifact into a new isolated environment using normal
dependency resolution, following the exact CPU reproduction commands linked above.
Only the exact Git core source is admitted; the prior core 4abc wheel is not admitted.
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

Product profile/protocol v3 verifies exact core Git provenance, the complete canonical core file set, and the complete packaged
product closure (including particles, cache, PG 09 training and all imported
helpers/configuration). The committed `anima_hndl/cohort.json` is derived from
core a794e3b's Git tree and this reviewed product source; users must not regenerate
it to admit altered files. Its digest and both source-closure digests enter the
adapter/critic identities. Earlier experimental v1/v2 artifacts, including 0.3.0a1/core 4abc, require their
original environments and are not migrated. Every experimental artifact requires
its recorded source closure. The optional `--preview-python` probe below verifies
actual old-preview training and raw/EMA native-packet refusals without rewriting
packet bytes.

Training caches and the HNDL critic have fixed native H/W. Protocol v3 native
exports record `[1,H,W]` and reject a different spatial shape before loading.
Opening another resolution requires a separate qualified cache/session/export.
The bank shape is derived from the product's `ARCHITECTURE` stamp.

For an actual two-environment historical checkpoint rejection check:

```sh
python cohorts/hypergan-hndl/check_legacy_packet.py \
  --legacy-python /absolute/stable-env/bin/python \
  --candidate-python /absolute/prerelease-env/bin/python \
  --preview-python /absolute/original-a1-env/bin/python \
  --workdir /absolute/new-evidence-directory
```

The probe creates a genuine installed core 0.8.1/PG 0.9.1 packet after two tiny CPU
updates, then reads the packet before restore and requires the exact refusal reason
`packet kind, source, capabilities or cohort differs; no implicit rebinding`,
with complete state preservation. Passing a path directly to `session.restore`
is invalid; older receipts that did so did not prove checkpoint rejection.

Response fixtures use tiny random Cosmos state without model loaders. Separate
loader regressions load only their own generated weights. Native pretrained/GPU execution, sampler/decoder parity, HNDL performance at production
resolution, and model-weight publication remain unqualified. CPU construction
and response tests do not open accelerator admission; GPU0 remains reserved.

Candidate owned-model loading additionally admits only the committed converted
model file set and content digests, matching revision and portable index identity.
The local `anima-lock.json` is compared with that authority; it cannot authorize
extra files. Every `pretrained_model_name_or_path` must be an absolute string
exactly equal to the resolved admitted model root. Relative names, `.` and None
are refused rather than normalized. Both `ModularPipeline.from_pretrained` and
the actual `load_components` call receive `local_files_only=True` from the
candidate-only initializer; the inherited predict/encode/decode methods and
historical loader remain unchanged. Extra indexes, external paths, symlinks and special files
are refused before loading, and the directory is rechecked after loading, including inode/stat identities
that detect same-byte file replacement across the load boundary. The
operator must keep it immutable throughout use; this is not a filesystem sandbox.
The historical stable loader is unchanged. Cache index bytes are bound before
provider construction; all selected paths must be regular local files, and
`CachedResponses` independently verifies every shard digest despite the lower
reader using `verify=False`. Hash integrity is not teacher authenticity.

Borrowed CPU runtime exports retain the actual runtime class source identity.
Use the candidate `LocalTurboRuntime` implementation for candidate native export
and restoration. A legacy runtime class is not relabeled as that implementation.
