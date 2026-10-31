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

## Bounded mitigation

Continue using only the original operator-trusted fixed-revision Anima conversion
staged and checked by `release_tools/prepare_model.py`. Its recorded model uses
individual safetensors files rather than supplied sharded indexes. Keep
verification enabled. Do not use the generic affected Accelerate loaders on
arbitrary directories or uploaded indexes. Use the isolated base candidate when
only shared CPU contracts are required.

This is a trust/format boundary, not automatic sanitization: a locally supplied
checksum manifest is not a signature, and `local_files_only=True` does not
validate shard paths. Replacing or concurrently changing the trusted directory
invalidates that premise. No global monkeypatch, weaker check or permission
change is introduced. Supporting untrusted sharded directories would require a
separately reviewed containment/regular-file boundary or an official tested fix;
that capability is not admitted. Historical locks and identities stay unchanged.
Full pretrained/GPU qualification remains pending. See the separate
[clean-install receipt](CLEAN-INSTALL-CPU.md).
