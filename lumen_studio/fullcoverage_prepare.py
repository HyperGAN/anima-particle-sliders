"""Fresh full-coverage teacher producer for an externally admitted runtime.

No model factory, scheduler replacement, optimizer, cache reuse, or automatic
successor lives here. The separate CPU conditioning producer must have finished
and unloaded its encoder before this GPU teacher phase is admitted.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from .contracts import atomic_json, digest, file_hash
from .coverage import (compile_coverage, field_metadata, validate_field_membership,
                       validate_index, require_production_rate_coverage,
                       freeze_teacher_membership)

CORE = "3dd672a7f9b8c169daef90fd63e4bb42b9ddc423"
CPU_PROFILE = "anima-native-CPU-BF16-conditioning-v1"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _prompt_sha(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def prepare_fullcoverage(runtime, manifests, directory, *, variation,
                         conditioning_provider, owner_check, admission,
                         source_authority, native_profile, mode="production",
                         pilot_rows=None, pilot_seeds=None):
    """Produce both pristine teachers on fresh native trajectories at every field.

    ``conditioning_provider.identity()`` and ``pair(row)`` consume only a new,
    authenticated CPU conditioning cache. They must not instantiate an encoder.
    The pair receipt binds exact UTF-8 prompt digests and the provider identity.
    Actual admission/source/lease/whole-phase ownership remains external.
    """
    _require(set(manifests) == {"train", "dev"}, "both declared unsealed splits required")
    contracts = {s: compile_coverage(m, variation, mode=mode,
        pilot_rows=None if pilot_rows is None else pilot_rows[s],
        pilot_seeds=None if pilot_seeds is None else pilot_seeds[s])
        for s, m in manifests.items()}
    _require(set(contracts["train"]["characters"]).isdisjoint(contracts["dev"]["characters"]),
             "training/development subjects overlap")
    _require(callable(owner_check) and callable(admission),
             "actual owner and source-bound admission callbacks required")
    _require(source_authority.get("core_commit") == CORE
             and source_authority.get("PG") == "0.9.5"
             and source_authority.get("core_version") == "0.13.0",
             "exact official core0.13/PG095 authority required")
    _require(source_authority.get("session_weight_EMA") is None
             and source_authority.get("materialized_EMA_shadows") is False
             and source_authority.get("selection") == "raw",
             "None/no weight EMA/no shadows/RAW required")
    directory = Path(directory)
    _require(not directory.exists(), "fresh source/cohort output required; no implicit resume")
    owner_check()
    condition_identity = deepcopy(conditioning_provider.identity())
    _require(condition_identity.get("profile") == CPU_PROFILE
             and condition_identity.get("CPU_encoder_unloaded") is True,
             "source-qualified CPU conditioning must finish/unload before GPU phase")
    condition_sha = digest(condition_identity)
    request = dict(phase="fresh-fullcoverage-teachers", mode=mode,
        coverage_sha256={s: c["sha256"] for s, c in contracts.items()},
        conditioning_identity_sha256=condition_sha,
        source_authority=deepcopy(source_authority), native_profile=deepcopy(native_profile),
        session_weight_EMA=None, materialized_EMA_shadows=False, selection="raw")
    _require(admission(deepcopy(request)) is True, "fresh reviewed external admission refused")
    # Native/model libraries are imported only after the actual admission callback.
    import torch
    from .cache import save_tensor_file
    from .core_source import verify_core
    from .numerical_profiles import NAME, require_numerical_profile
    from particle_sliders.pg095 import PG_VERSION, PG_SOURCE_PROFILE
    verify_core()
    _require((PG_VERSION, PG_SOURCE_PROFILE) == ("0.9.5", "release-0.9.5"),
             "actual installed official PG095 source differs")
    _require(require_numerical_profile(NAME) == native_profile,
             "actual fresh native numerical profile differs")
    _require(not runtime.mixer.adapters and not runtime.mixer.handles,
             "teacher runtime must be pristine with no attached adapters")
    _require(not any(p.requires_grad for p in runtime.transformer.parameters()),
             "teacher base must be frozen")
    directory.mkdir(parents=True)
    atomic_json(directory / "preparation-authority.json", request)
    indexes, memberships = {}, {}
    with torch.no_grad(), runtime.mixer.scales({}):
        for split in ("train", "dev"):
            contract = contracts[split]
            root = directory / split
            root.mkdir()
            atomic_json(root / "coverage.json", contract)
            identity = dict(model=deepcopy(runtime.identity), split=split, variation=variation,
                manifest_sha256=manifests[split]["sha256"], resolution=512, steps=10, cfg=1,
                coverage_sha256=contract["sha256"], conditioning_identity=condition_identity,
                conditioning_identity_sha256=condition_sha,
                numerical_profile=deepcopy(native_profile), teacher="fresh-current-runtime",
                layout="both-trajectories/all-positions/full-velocity/shared-state-v1",
                coordinate_origin="fresh native trajectories under this model/math/conditioning",
                source_authority=deepcopy(source_authority), mode=mode,
                production_rate_eligible=contract["production_rate_eligible"])
            fingerprint = digest(identity)
            shards, observed, pair_receipts = [], [], []
            by_id = {r["id"]: r for r in manifests[split]["rows"]}
            for row_id in contract["row_ids"]:
                row = by_id[row_id]
                owner_check()
                neutral, positive, receipt = conditioning_provider.pair(deepcopy(row))
                _require(receipt.get("row_id") == row_id
                    and receipt.get("neutral_prompt_sha256") == _prompt_sha(row["neutral"])
                    and receipt.get("positive_prompt_sha256") == _prompt_sha(row["positive"])
                    and receipt.get("conditioning_identity_sha256") == condition_sha,
                    "conditioning pair differs from exact declared prompts/source")
                for value in (neutral, positive):
                    _require(value.device.type == "cpu" and value.dtype == torch.bfloat16
                        and tuple(value.shape) == (1, 512, 1024) and not value.requires_grad
                        and torch.isfinite(value).all(), "invalid source-qualified CPU embedding")
                pair_receipts.append(deepcopy(receipt))
                for seed in contract["seeds_by_row"][row_id]:
                    owner_check()
                    initial = runtime.noise(seed, 512, 512)
                    _require(tuple(initial.shape) == (1, 16, 1, 64, 64),
                             "native512 geometry differs")
                    records = []
                    for trajectory in ("neutral", "positive"):
                        latent = initial.clone()
                        scheduler = runtime.scheduler(10)
                        _require(len(scheduler.timesteps) == 10, "all ten native steps required")
                        for position, timestep in enumerate(scheduler.timesteps):
                            owner_check()
                            require_numerical_profile(NAME)
                            n = runtime.predict(latent, timestep, neutral)
                            t = runtime.predict(latent, timestep, positive)
                            _require(torch.isfinite(n).all() and torch.isfinite(t).all(),
                                     "nonfinite pristine teacher field")
                            n_cpu, t_cpu = n.cpu().clone(), t.cpu().clone()
                            records.append(dict(position=position, timestep=float(timestep),
                                trajectory=trajectory, latent=latent.cpu().clone(),
                                neutral=n_cpu, positive=t_cpu))
                            teacher_sse = float((t_cpu.double() - n_cpu.double()).square().sum())
                            observed.append(dict(field_metadata(row, seed, trajectory, position,
                                index=len(observed)), teacher_sse=teacher_sse, numel=n_cpu.numel()))
                            latent = runtime.step(scheduler, n if trajectory == "neutral" else t,
                                                  timestep, latent)
                    path = root / f"{row_id}-{seed}.pt"
                    save_tensor_file(path, dict(fingerprint=fingerprint, row=deepcopy(row), seed=seed,
                        embedding=neutral.clone(), positive_embedding=positive.clone(), records=records))
                    shards.append(dict(path=path.name, sha256=file_hash(path), count=20,
                                       row=row_id, seed=seed))
            validate_field_membership(observed, contract)
            index = dict(identity=identity, input_fingerprint=fingerprint,
                         fingerprint=digest(dict(input=fingerprint, shards=shards)), shards=shards)
            validate_index(index, contract)
            _require(conditioning_provider.identity() == condition_identity,
                     "conditioning identity changed during teacher production")
            atomic_json(root / "conditioning-receipts.json", pair_receipts)
            atomic_json(root / "field-membership.json", observed)
            atomic_json(root / "teacher-membership.json", freeze_teacher_membership(observed, contract))
            atomic_json(root / "index.json", index)
            indexes[split] = index
            memberships[split] = observed
    owner_check()
    coverage_authority = (require_production_rate_coverage(manifests, contracts, indexes, memberships)
                          if mode == "production" else
                          dict(status="BOUNDED_PILOT_INELIGIBLE_FOR_RATE_PUBLICATION"))
    atomic_json(directory / "coverage-authority.json", coverage_authority)
    result = dict(schema="anima-pg095-fresh-fullcoverage-teachers-v1",
        status="complete", indexes={s: dict(path=str(directory / s / "index.json"),
            sha256=file_hash(directory / s / "index.json")) for s in indexes},
        coverage_sha256={s: c["sha256"] for s, c in contracts.items()},
        mode=mode, production_rate_eligible=(mode == "production"),
        old_teacher_or_embedding_reuse=False, fresh_native_coordinates=True,
        conditioning_identity_sha256=condition_sha, new_update_calls=0,
        teacher_membership={s: dict(path=str(directory / s / "teacher-membership.json"),
            sha256=file_hash(directory / s / "teacher-membership.json")) for s in indexes},
        field_membership={s: dict(path=str(directory / s / "field-membership.json"),
            sha256=file_hash(directory / s / "field-membership.json")) for s in indexes},
        coverage_authority=dict(path=str(directory / "coverage-authority.json"),
                                sha256=file_hash(directory / "coverage-authority.json")),
        session_weight_EMA=None, materialized_EMA_shadows=False, selection="raw",
        actual_training_RAW_or_rate_qualification=False)
    atomic_json(directory / "preparation-complete.json", result)
    return result
