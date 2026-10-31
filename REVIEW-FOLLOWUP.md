# Review follow-up: isolated prerelease and complete source admission

This follows the source review of45ee9dc. The default installer remains stable
Anima0.2.0/core0.8.1/PG0.9.1. Historical requirements and source/model locks are
unchanged. The candidate is now explicitly0.3.0a1, profile/protocol v2; earlier
experimental v1 artifacts require their original environment.

| Review issue | Change and evidence |
| --- | --- |
| Final0.3 could replace stable | Explicit prerelease0.3.0a1; only a dedicated venv with no shared/system site-packages and exact installed product/dependency origins is admitted. Never install into a stable environment. |
| Modern code leaked into stable wheel | Moved to candidate-only `anima_hndl`; stable wheel tests assert neither that namespace nor the old `lumen_studio.hndl_lifecycle` is present. Stable retains `anima`; candidate registers only `anima-hndl`. |
| Version-only core admission | Verify exact core Git commit4abc0eb or the reviewed joint wheel SHA, plus all50 canonical core files from that Git tree. Wrong Git metadata and actual core-file tampering are rejected. |
| Imported product helpers unbound | Committed manifest covers all63 packaged product files, including particles/cache/PG09 training, native owner, source validator and configuration. Actual changes in those three helpers are rejected; source-closure digests and manifest hash enter network identity. Sdist-built candidate contents are checked against the complete manifest. |
| Installer/lock ambiguity | The a119-to-project installer change intentionally resolves the existing core0.3/core0.8.1 conflict. Torchvision is intentional for the pinned Cosmos import. README/formulation/reproduction/install docs and generator explicitly scope core.lock's original note/version to the historical archive. Its bytes remain unchanged. |
| Missing validation after load | Revalidate after native owner construction; a loader replacing the configured cache index is rejected before adapter/critic construction, with cleanup verified. |
| Magic bank dimensions | Allocation and HNDL input shape derive from `ARCHITECTURE['particles']` and `ARCHITECTURE['particle_dim']`. |
| Resolution contract unclear | Training critic/cache are fixed H/W. Protocol v2 exports `[1,H,W]`; native inference refuses another spatial shape before loading. |
| No genuine old-packet probe | Committed `cohorts/hypergan-hndl/check_legacy_packet.py` runs two installed environments, creates a real two-update core0.8.1/PG0.9.1 packet and proves candidate rejection without state mutation. |
| Native adapter identity | Installed native inference checks actual candidate adapter type, shared wrapper parameter object identities, v2 protocol, product source identity and exact raw/EMA zero/fractional response parity. |

Final results: 88 stable installed checks, 123 archive/release checks and17
fresh-installed candidate/loader checks pass. The genuine two-environment packet
probe passes. Final candidate installation resolves all declared native extras
from public sources and `pip check` passes, without dependency sharing.

The advisory assessment is [ACCELERATE-ADVISORY.md](ACCELERATE-ADVISORY.md).
No official patched Accelerate release was verified;1.15 still has the same
unguarded API code. Supported component paths avoid those APIs in actual local
sharded-model tests. The bounded mitigation is trusted fixed-revision staging,
not a claim that arbitrary model directories are sanitized. No dependency pin,
checkpoint label, permissions, shared GAN implementation or admission guard was
weakened. Full pretrained/GPU/render qualification remains pending.
