# Anima HyperGAN/HNDL candidate

This separate source cohort uses the actual shared core0.12/PG0.9.4 game and
HyperGAN develop/HNDL0.10 builder, for both the native routed adapter and the
conditional patch critic. It retains Anima's genuine five-dimensional velocity
fields at the public ParticleGAN penalty boundary. There is no new loss,
optimizer, copied GAN loop or continuous-domain admission bypass.

Bounded installed CPU results and remaining limits: [HNDL-CPU.md](HNDL-CPU.md).
The later [independent clean install](CLEAN-INSTALL-CPU.md) resolves the public
sources without dependency sharing. See the [Accelerate API assessment](ACCELERATE-ADVISORY.md)
for the pinned native loader's scope and operating boundary.

The root `pyproject.toml` remains the known CPU-tested core0.8.1/PG0.9.1 cohort.
`cohorts/hypergan-hndl/pyproject.toml` is an explicit prerelease build manifest:
Anima0.3.0a1, exact shared source
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, official PG0.9.4, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10.0. Those exact sources are
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
in `source_validation.py`, and do not sandbox PYTHONPATH or every dependency. Stable0.2.0 retains only the `anima` entrypoints and
contains no modern `anima_hndl` implementation. The prerelease installs that
separate namespace and registers only `anima-hndl`.

Build the candidate locally in its own environment with patched setuptools:

```sh
python cohorts/hypergan-hndl/build.py --out /absolute/path/to/candidate-wheels
```

The helper stages a temporary source tree and uses the separate manifest; it
does not edit the root manifest. For local qualification only, install the exact
joint core wheel and its pinned framework dependencies, then install the built
product wheel with `--no-deps`. That original source-qualification method is
separate from the later full dependency-resolution receipt linked above.
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

Product profile/protocol v2 verifies exact core Git provenance or the qualified
joint wheel SHA, the complete canonical core file set, and the complete packaged
product closure (including particles, cache, PG09 training and all imported
helpers/configuration). The committed `anima_hndl/cohort.json` is derived from
core4abc0eb's Git tree and this reviewed product source; users must not regenerate
it to admit altered files. Its digest and both source-closure digests enter the
adapter/critic identities. Earlier experimental v1 artifacts require their
original environment and are not migrated. Every experimental artifact, including
v2, requires its recorded source closure; a shared profile number alone does not
make changed draft sources checkpoint-compatible.

Training caches and the HNDL critic have fixed native H/W. Protocol v2 native
exports record `[1,H,W]` and reject a different spatial shape before loading.
Opening another resolution requires a separate qualified cache/session/export.
The bank shape is derived from the product's `ARCHITECTURE` stamp.

For an actual two-environment historical checkpoint rejection check:

```sh
python cohorts/hypergan-hndl/check_legacy_packet.py \
  --legacy-python /absolute/stable-env/bin/python \
  --candidate-python /absolute/prerelease-env/bin/python \
  --workdir /absolute/new-evidence-directory
```

The probe creates a genuine installed core0.8.1/PG0.9.1 packet after two tiny CPU
updates, then requires the exact refusal reason `unsupported session packet version; no
automatic migration` and verifies complete state preservation.

CPU fixtures use a tiny random Cosmos model without pretrained loaders. Native
pretrained/GPU execution, sampler/decoder parity, HNDL performance at production
resolution, and model-weight publication remain unqualified. CPU construction
and response tests do not open accelerator admission; GPU0 remains reserved.

Candidate owned-model loading additionally admits only the committed converted
model file set and content digests, matching revision and portable index identity.
The local `anima-lock.json` is compared with that authority; it cannot authorize
extra files. Extra indexes, external component paths, symlinks and special files
are refused before loading, and the directory is rechecked after loading. The
operator must keep it immutable throughout use; this is not a filesystem sandbox.
The historical stable loader is unchanged. Cache index bytes are bound before
provider construction; all selected paths must be regular local files, and
`CachedResponses` independently verifies every shard digest despite the lower
reader using `verify=False`. Hash integrity is not teacher authenticity.
