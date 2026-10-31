# Review follow-up: isolated prerelease and complete source admission

This follows the source review of 45ee9dc. The default installer remains stable
Anima 0.2.0/core 0.8.1/PG 0.9.1. Historical requirements and source/model locks are
unchanged. The candidate is now explicitly0.3.0a1, profile/protocol v2; earlier
experimental v1 artifacts require their original environment.

| Review issue | Change and evidence |
| --- | --- |
| Final0.3 could replace stable | Explicit prerelease 0.3.0a1 plus runtime admission of a dedicated venv and selected installed roots. Both artifacts still share a distribution name: pip can overwrite stable. Dedicated installation is an operator requirement; no install-time isolation is claimed. |
| Modern code leaked into stable wheel | Moved to candidate-only `anima_hndl`; stable wheel tests assert neither that namespace nor the old `lumen_studio.hndl_lifecycle` is present. Stable retains `anima`; candidate registers only `anima-hndl`. |
| Version-only core admission | Verify exact core Git commit4abc0eb or the reviewed joint wheel SHA, plus all 50 canonical core files from that Git tree. Wrong Git metadata and actual core-file tampering are rejected. |
| Imported product helpers unbound | Committed manifest covers all 65 packaged product files, including particles/cache/PG 09 training, native owner, source validator and configuration. Actual changes in those three helpers are rejected; source-closure digests and manifest hash enter network identity. Sdist-built candidate contents are checked against the complete manifest. |
| Installer/lock ambiguity | The a119-to-project installer change intentionally resolves the existing core 0.3/core 0.8.1 conflict. Torchvision is intentional for the pinned Cosmos import. README/formulation/reproduction/install docs and generator explicitly scope core.lock's original note/version to the historical archive. Its bytes remain unchanged. |
| Missing validation after load | Revalidate after native owner construction; a loader replacing the configured cache index is rejected before adapter/critic construction, with cleanup verified. |
| Magic bank dimensions | Allocation and HNDL input shape derive from `ARCHITECTURE['particles']` and `ARCHITECTURE['particle_dim']`. |
| Resolution contract unclear | Training critic/cache are fixed H/W. Protocol v2 exports `[1,H,W]`; native inference refuses another spatial shape before loading. |
| No genuine old-packet probe | Committed `cohorts/hypergan-hndl/check_legacy_packet.py` runs two installed environments, creates a real two-update core 0.8.1/PG 0.9.1 packet and requires the exact packet-version refusal and proves no state mutation. |
| Native adapter identity | Installed native inference checks actual candidate adapter type, shared wrapper parameter object identities, v2 protocol, product source identity and exact raw/EMA zero/fractional response parity. |

Return-review corrections add candidate-only admission against the committed
converted-model manifest and released portable identity. The local anima-lock
cannot authorize another revision, extra index, content or external path.
Regular no-follow opens refuse symlinks/FIFOs. Admission occurs before and after
owned loading; post-load mutation closes the owner. The historical stable loader
is unchanged. Directories must remain immutable; no filesystem sandbox is claimed.

Cache index admission is compared with the reader snapshot. All shard paths must
be regular and contained; `CachedResponses` hashes all shard bytes even when the
lower cache reader uses `verify=False`. Content corruption with an unchanged index
and index replacement before reader construction both refuse before attachment.
Selected loaded root `__file__` origins must match their admitted specs, but checks
run after Python startup and do not cover every dependency or arbitrary PYTHONPATH
execution. Use a trusted dedicated environment and `python -I`.

Executable source evidence:

- `tests/hndl/test_input_admission.py`: generated-file attacks before any loader,
  actual corrupt cache/index refusal, bounded FIFO opens and post-load cleanup.
- `tests/hndl/test_anima_hndl.py`: actual shared game, gradient ownership,
  raw/EMA responses, source/provenance and native inference adapter identities.
- `tests/hndl/test_loading_advisory.py`: actual local sharded Cosmos/Qwen3 loads;
  Accelerate API traps exist only inside these tests, not in the runtime.
- `cohorts/hypergan-hndl/check_legacy_packet.py`: real installed two-update
  core 0.8.1 packet, exact `unsupported session packet version; no automatic
  migration` refusal, complete state unchanged in core0.12.

Results: 88 stable installed checks, 123 archive/release checks and 49 installed
candidate/admission/loader checks pass. The genuine two-environment packet probe
passes with the exact refusal above. The updated wheel reuses the previously
physically isolated dependency environment; it is a new artifact/source test.
Earlier fresh public-source native-extra resolver reports remain separate.

The advisory assessment is [ACCELERATE-ADVISORY.md](ACCELERATE-ADVISORY.md).
No verified official patch was found;1.15 still has the unguarded API code. Pins
remain unchanged. Avoidance was tested only for the two pinned component paths.
Full owned pretrained/GPU/render qualification remains blocked: among other
full-lock differences, native-extra Pillow>=13 conflicts with inherited
Pillow==12.3.0. A separate qualified native cohort must resolve that conflict.
No checkpoint labels, historical locks, permissions or shared GAN math changed.

The subsequent return-review path correction rejects relative/bare/None component
roots instead of normalizing them. The literal must exactly equal the absolute
admitted root. A candidate-only `LocalTurboRuntime` initializer explicitly passes
`local_files_only=True` to both loading calls; all native execution methods are
inherited, and stable loader/source bytes stay unchanged. Actual pinned pipeline,
component and Cosmos loaders are exercised by the generated fixture regression,
which asserts literal arguments and local-only propagation with network forbidden.
Eight ambiguous path forms refuse before initialization; same-byte inode
replacement during loading also refuses with owned cleanup. The immutable-input
requirement remains: snapshots are not a concurrent filesystem sandbox. Native
identity retains the actual candidate runtime class; legacy classes are not
relabeled as equivalent.


Coordinated core adoption: the preceding review notes describe0.3.0a1/core 4abc.
The active candidate is now 0.3.0a2/profile/protocol v3 with exact core a794; see
MODERN-CORE-A794.md. Historical/default cohorts and earlier review evidence remain
unchanged. Correction to the old-packet probe: passing a path to session.restore
caused an input-type refusal, not a genuine decoded-packet compatibility check.
The current probe calls read_packet first and requires the actual source/cohort
refusal for both stable and old-preview packets, preserving state and old bytes.
