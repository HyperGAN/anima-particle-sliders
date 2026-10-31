# Review follow-up: isolated prerelease and complete source admission

This follows the source review of45ee9dc. The default installer remains stable
Anima0.2.0/core0.8.1/PG0.9.1. Historical requirements and source/model locks are
unchanged. The candidate is now explicitly0.3.0a1, profile/protocol v2; earlier
experimental v1 artifacts require their original environment.

| Review issue | Change and evidence |
| --- | --- |
| Final0.3 could replace stable | Explicit prerelease0.3.0a1 plus runtime admission of a dedicated venv and selected installed roots. Both artifacts still share a distribution name: pip can overwrite stable. Dedicated installation is an operator requirement; no install-time isolation is claimed. |
| Modern code leaked into stable wheel | Moved to candidate-only `anima_hndl`; stable wheel tests assert neither that namespace nor the old `lumen_studio.hndl_lifecycle` is present. Stable retains `anima`; candidate registers only `anima-hndl`. |
| Version-only core admission | Verify exact core Git commit4abc0eb or the reviewed joint wheel SHA, plus all50 canonical core files from that Git tree. Wrong Git metadata and actual core-file tampering are rejected. |
| Imported product helpers unbound | Committed manifest covers all64 packaged product files, including particles/cache/PG09 training, native owner, source validator and configuration. Actual changes in those three helpers are rejected; source-closure digests and manifest hash enter network identity. Sdist-built candidate contents are checked against the complete manifest. |
| Installer/lock ambiguity | The a119-to-project installer change intentionally resolves the existing core0.3/core0.8.1 conflict. Torchvision is intentional for the pinned Cosmos import. README/formulation/reproduction/install docs and generator explicitly scope core.lock's original note/version to the historical archive. Its bytes remain unchanged. |
| Missing validation after load | Revalidate after native owner construction; a loader replacing the configured cache index is rejected before adapter/critic construction, with cleanup verified. |
| Magic bank dimensions | Allocation and HNDL input shape derive from `ARCHITECTURE['particles']` and `ARCHITECTURE['particle_dim']`. |
| Resolution contract unclear | Training critic/cache are fixed H/W. Protocol v2 exports `[1,H,W]`; native inference refuses another spatial shape before loading. |
| No genuine old-packet probe | Committed `cohorts/hypergan-hndl/check_legacy_packet.py` runs two installed environments, creates a real two-update core0.8.1/PG0.9.1 packet and requires the exact packet-version refusal and proves no state mutation. |
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
  core0.8.1 packet, exact `unsupported session packet version; no automatic
  migration` refusal, complete state unchanged in core0.12.

Results: 88 stable installed checks, 123 archive/release checks and39 installed
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
