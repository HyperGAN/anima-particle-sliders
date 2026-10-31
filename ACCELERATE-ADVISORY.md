# Accelerate sharded-checkpoint assessment — 2026-10-31

[GHSA-4j2p-28q2-5m79](https://github.com/advisories/GHSA-4j2p-28q2-5m79)
lists Accelerate<=1.14.0 and no patched version. The affected APIs are
`accelerate.utils.load_checkpoint_in_model` and
`accelerate.load_checkpoint_and_dispatch`. The first takes shard filenames from
an index's `weight_map` and joins them to the checkpoint directory without
containment or regular-file checks; the second calls the first. Traversal and
absolute names can open outside files; a FIFO can block loading.

The provider's [issue4067](https://github.com/huggingface/accelerate/issues/4067)
and [PR4214](https://github.com/huggingface/accelerate/pull/4214) record the report
and proposed guard. That PR was closed without merge. The provider's
[inspected security policy](https://github.com/huggingface/accelerate/blob/3008af926ffe3e04ffe6fbe3d87494193fc85041/SECURITY.md)
treats selecting model artifacts as a trust decision. That policy does not remove
the observed path behavior or the published advisory.

PyPI's current official release is1.15.0. Both official1.14.0 and1.15.0 wheels
still contain the unguarded shard-path join. Being outside the advisory's listed
range does not establish that1.15.0 is fixed. No official patched version was
verified, so no dependency was upgraded, downgraded or replaced with a fork.
Official wheel SHA256 values are
`e94390c2863b873be18f623f9df48a0d8fe5eff13ea7f1a00092b0a7904888c6`
for1.14.0 and
`97eacca0b73e45cb867dbf8c5d5d4dc32219544300e0c8992c7334dc2ef33cec`
for1.15.0. An isolated1.14.0 diagnostic reproduced relative and absolute
out-of-directory loads using only self-created tiny safetensors in one temporary
directory. No system file, user data or FIFO was opened by that diagnostic.

## Repository reachability

The base/HNDL candidate has no Accelerate dependency; the optional `native` extra
and historical model environment select it. Both product session factories use
`TurboRuntime` for an owned model. It verifies the recorded environment and
staged hashes, then calls local `ModularPipeline.from_pretrained` and
`load_components`. Pinned Diffusers
`d5baa4fb548294f47dbca49890abd4b291204c60` routes through `ComponentSpec.load`
to each model class's `from_pretrained`. Diffusers model classes and
Transformers5.15.0 use their own state-dictionary loading implementations.
Neither pinned package's Python sources reference either affected Accelerate
API. Anima and shared HNDL construction do not call them either.

Two regression checks load actual locally saved, randomly initialized sharded
Cosmos and Qwen3 models through `ComponentSpec.load`, trapping every exposed
alias of both affected APIs to fail. Both loads reproduce the complete saved
state exactly. This exercises the component paths used by Anima; it does not
qualify pretrained Anima, all model classes/plugins or arbitrary artifacts.
Borrowed staged-runtime CPU sessions construct no model loader. Product routed
exports use safetensors with tensor-schema/identity checks; shared session and
native envelopes use the shared restore API, not either affected loader.

## Bounded candidate admission and remaining limits

A model directory's own `anima-lock.json` is self-attestation. The historical
stable loader's staged-hash check alone does not establish trusted file admission.
The candidate now compares it with packaged, committed
`configs/anima/converted-model.lock.json`, admits exactly that file set plus the
lock, hashes every non-index file against the committed digests, and binds the
complete portable modular index to `configs/anima/released-identity.json`.
Only local component paths within the admitted layout are accepted. Extra shard
indexes (even added to the local lock), different revisions or content, symlinks,
FIFOs and external component paths are refused before a model loader runs.
The directory is rechecked after loading. Descriptor-relative no-follow opens
refuse special files without blocking. Admission requires an immutable directory
throughout subsequent loading/use; it does not sandbox a concurrent malicious
writer. Historical stable code and locks are unchanged.

`tests/hndl/test_input_admission.py` uses only generated placeholder file attacks,
with the model loader forbidden. A separate local-component diagnostic found
that an extra index is considered by pinned Cosmos loading, which rejects its
parent-traversal name, while pinned Qwen3 selects its existing single weight file.
Neither component loaded the outside fixture. This does not negate the direct
Accelerate API reproduction above. The API traps in
`tests/hndl/test_loading_advisory.py` are **test-only assertions of avoidance**;
there is no installed runtime trap or global patch. Avoidance is qualified only
for those two actual pinned component paths and fixtures.

Cache admission binds configured index bytes before provider construction and
compares the reader's index with that admitted snapshot. It refuses traversal,
symlinks and special shard files; `CachedResponses` independently hashes every
shard even though `TargetCache(..., verify=False)` skips its own first hash pass.
Tests corrupt content without changing the index and replace the index between
admission and reader construction; both refuse before attachment. This proves
integrity checks, not teacher authenticity or protection from concurrent writers.

Do not use the generic affected Accelerate loaders on arbitrary directories or
uploaded indexes. Keep verification enabled and use the isolated base candidate
when only shared CPU contracts are required. No verified official patch was
found, so pins remain unchanged. No pretrained model was downloaded or loaded.
The full inherited native environment is still refused; in particular its
Pillow==12.3.0 conflicts with the candidate extra's Pillow>=13. A separately
qualified native cohort is required before owned pretrained/GPU execution.
See [CLEAN-INSTALL-CPU.md](CLEAN-INSTALL-CPU.md) for installation versus runtime
qualification. The candidate's selected import-root checks run after Python
startup and are not a universal PYTHONPATH or dependency sandbox.
