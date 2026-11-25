"""Remove scene and lighting contrasts from the exterior of the soft45 target."""

from pathlib import Path
import sys
from unittest.mock import patch

import optimization_worker as harness
from optimization_targets import apply_target, mattes, plan


METHOD = "soft45-background-orthogonal"
RECIPE = {"method": METHOD, "teacher": "background-orthogonal-v1", "floor": .45, "donor": None, "blend": 0}
SOFT45 = harness.ROOT / "cohorts/soft-floor45"
EXTERIOR_THRESHOLD = .15
COUNTERFACTUALS = {
    "lighting": "hard theatrical rim light with deep shadows",
    "background": "a plain undecorated studio wall behind the subject",
}
NORM_TOLERANCE = 1e-5
PROJECTION_TOLERANCE = 1e-6


def recipe_for(method):
    if method != METHOD or plan(method) != RECIPE:
        raise ValueError(f"Invalid exterior orthogonalization recipe: {method}")
    return RECIPE


def source_manifest(original):
    sources = {}
    for item in original["identity"]["teachers"]:
        row = item["row"]
        neutral = row["neutral"]
        probes = {}
        for clause, replacement in COUNTERFACTUALS.items():
            field = row[clause]
            if neutral.count(field) != 1:
                raise ValueError(f"Expected exactly one matched clause in {row['id']}: {clause}")
            probes[clause] = neutral.replace(field, replacement, 1)
        sources[row["id"]] = {
            "neutral": neutral, "positive": row["positive"], "mask": mattes()[row["id"]], "probes": probes,
        }
    return sources


def cache_identity(original, soft45, method):
    sources = source_manifest(original)
    if (soft45["identity"]["parent"] != original["fingerprint"]
            or soft45["identity"]["recipe"] != plan("soft-floor45")
            or soft45["identity"]["sources"] != {
                name: {key: field for key, field in source.items() if key != "probes"}
                for name, source in sources.items()
            }):
        raise ValueError("The source is not the matching soft45 target cache")
    replay_path = SOFT45 / "native-parity.json"
    replay = harness.read(replay_path)
    if not replay["passed"] or replay["request"]["cache_fingerprint"] != soft45["fingerprint"]:
        raise ValueError("The soft45 source did not pass native replay")
    return original["identity"] | {
        "parent": original["fingerprint"],
        "layout": f"exterior-orthogonal/{method}/v1",
        "implementation": [harness.pin(harness.ROOT / name) for name in (
            "background_orthogonal_teacher_worker.py", "optimization_worker.py", "optimization_targets.py",
            f"plans/{method}.json",
        )],
        "recipe": recipe_for(method), "donor_source": None,
        "soft45_cache": {"fingerprint": soft45["fingerprint"],
                         "index": harness.pin(SOFT45 / "target-cache/index.json"),
                         "native_replay": harness.pin(replay_path)},
        "sources": sources,
        "exterior": {"matte_max_inclusive": EXTERIOR_THRESHOLD, "space": "native area-resampled matte"},
        "target_norm": "preserve soft45 interior exactly and restore the exterior residual L2 after projection",
    }


def context(runtime, source, shape):
    from fit import mask_for

    path = Path(source["mask"]["path"])
    if harness.pin(path) != source["mask"]:
        raise ValueError("The source subject matte changed")
    matte = mask_for(path, shape, runtime.device)
    return matte, [runtime.encode(source["probes"][name]) for name in COUNTERFACTUALS]


def project_exterior(neutral, positive, probe_responses, matte):
    import torch

    support = (matte <= EXTERIOR_THRESHOLD).expand_as(positive)
    neutral_ext = neutral.masked_select(support).double()
    positive_ext = positive.masked_select(support).double()
    if neutral_ext.numel() < 3 or neutral_ext.numel() == positive.numel():
        raise ValueError("The target needs both an interior and an exterior support")
    probes = [response.masked_select(support).double() - neutral_ext for response in probe_responses]
    if len(probes) != len(COUNTERFACTUALS):
        raise ValueError("Expected both pinned nuisance responses")
    probe_norms = [probe.norm() for probe in probes]
    basis = []
    for probe, norm in zip(probes, probe_norms, strict=True):
        if norm.item() <= 1e-10:
            continue
        direction = probe / norm
        for axis in basis:
            direction = direction - torch.dot(direction, axis) * axis
        length = direction.norm()
        if length.item() > 1e-10:
            basis.append(direction / length)
    original = positive_ext - neutral_ext
    radius = original.norm()
    removed = sum(torch.dot(original, axis) * axis for axis in basis)
    remainder = original - removed
    remaining = remainder.norm()
    if radius.item() <= 1e-10 or remaining.item() <= 1e-8 * radius.item():
        raise ValueError("No orthogonal exterior edit with the requested source norm")
    exterior = (neutral_ext + remainder * (radius / remaining)).to(positive.dtype)
    target = positive.masked_scatter(support, exterior)
    realized = target.masked_select(support).double() - neutral_ext
    interior_difference = (target.masked_select(~support) - positive.masked_select(~support)).abs().max()
    residual_norm = realized.norm()
    raw = positive.double() - neutral.double()
    output = target.double() - neutral.double()
    details = {
        "source_l2": raw.norm().item(), "target_l2": output.norm().item(),
        "exterior_source_l2": radius.item(), "exterior_target_l2": residual_norm.item(),
        "exterior_renormalization": (radius / remaining).item(),
        "exterior_element_count": neutral_ext.numel(), "field_element_count": positive.numel(),
        "max_interior_target_difference": interior_difference.item(), "basis_rank": len(basis),
        "removed_l2": removed.norm().item() if basis else 0.,
        "exterior_change_l2": (realized - original).norm().item(),
        "nuisances": {
            name: {
                "norm": norm.item(),
                "absolute_cosine_before": (torch.dot(probe, original).abs() / (norm * radius)).item()
                if norm.item() > 1e-10 else 0.,
                "absolute_cosine_after": (torch.dot(probe, realized).abs() / (norm * residual_norm)).item()
                if norm.item() > 1e-10 else 0.,
            } for name, probe, norm in zip(COUNTERFACTUALS, probes, probe_norms, strict=True)
        },
    }
    return target, details


def check_field(row):
    import math

    for key in ("source_l2", "target_l2", "exterior_source_l2", "exterior_target_l2",
                "exterior_renormalization", "removed_l2", "exterior_change_l2"):
        if not math.isfinite(row[key]) or row[key] < 0:
            raise ValueError(f"Invalid exterior diagnostic {key}")
    if (row["source_l2"] <= 0 or row["exterior_source_l2"] <= 0
            or abs(row["target_l2"] / row["source_l2"] - 1) >= NORM_TOLERANCE
            or abs(row["exterior_target_l2"] / row["exterior_source_l2"] - 1) >= NORM_TOLERANCE
            or not 0 < row["exterior_element_count"] < row["field_element_count"]
            or row["max_interior_target_difference"] != 0
            or row["basis_rank"] not in (0, 1, 2)
            or set(row["nuisances"]) != set(COUNTERFACTUALS)):
        raise ValueError("The target did not preserve the soft45 interior and edit norms")
    for entry in row["nuisances"].values():
        if (not math.isfinite(entry["norm"]) or entry["norm"] < 0
                or not 0 <= entry["absolute_cosine_before"] <= 1 + PROJECTION_TOLERANCE
                or not 0 <= entry["absolute_cosine_after"] < PROJECTION_TOLERANCE):
            raise ValueError("Exterior target is not orthogonal to both measured probes")


def diagnostics_proof(fields):
    if len(fields) != 120:
        raise ValueError("Expected 120 original native states")
    for row in fields:
        check_field(row)
    removed = max(row["removed_l2"] / row["exterior_source_l2"] for row in fields)
    if removed <= PROJECTION_TOLERANCE:
        raise ValueError("The captured nuisance plane removed no exterior edit")
    return {
        "maximum_exterior_removed_fraction": removed,
        "maximum_remaining_absolute_cosine": max(entry["absolute_cosine_after"] for row in fields
                                                 for entry in row["nuisances"].values()),
        "maximum_interior_target_difference": 0.,
        "maximum_exterior_relative_norm_difference": max(
            abs(row["exterior_target_l2"] / row["exterior_source_l2"] - 1) for row in fields),
    }


def caches(runtime, method):
    from lumen_studio.cache import TargetCache

    original = TargetCache(harness.BASE_CACHE, verify=True)
    soft45 = TargetCache(SOFT45 / "target-cache", verify=True)
    if (original.index["identity"]["model"] != runtime.identity
            or soft45.index["identity"]["model"] != runtime.identity
            or len(original) != 120 or len(soft45) != len(original)):
        raise ValueError("The native runtime does not match both original sources")
    return original, soft45, cache_identity(original.index, soft45.index, method)


def prepare_cache(runtime, method):
    import torch
    from lumen_studio.cache import TargetCache, save_tensor_file
    from lumen_studio.contracts import digest

    original, soft45, identity = caches(runtime, method)
    directory = harness.ROOT / "cohorts" / method / "target-cache"
    input_sha = digest(identity)
    existing = directory / "index.json"
    if existing.is_file():
        cache = TargetCache(directory, verify=True)
        if cache.index["identity"] != identity or len(cache) != len(original):
            raise ValueError("A different exterior target is already pinned")
        diagnostics_proof(harness.read(directory / "diagnostics.json"))
        return cache
    directory.mkdir(parents=True, exist_ok=True)
    shards, fields = [], []
    with torch.no_grad(), runtime.mixer.scales({}):
        for shard, source_shard in zip(original.index["shards"], soft45.index["shards"], strict=True):
            harness.guarded()
            parent = torch.load(harness.BASE_CACHE / shard["path"], map_location="cpu", weights_only=True)
            source = torch.load(SOFT45 / "target-cache" / source_shard["path"], map_location="cpu", weights_only=True)
            if (shard["path"] != source_shard["path"] or parent["row"] != source["row"]
                    or not torch.equal(parent["embedding"], source["embedding"])
                    or len(parent["records"]) != 20 or parent["fingerprint"] != original.index["input_fingerprint"]
                    or source["fingerprint"] != soft45.index["input_fingerprint"]):
                raise ValueError("The soft45 source lost its original native state layout")
            scene = parent["row"]["id"]
            matte, probes = context(runtime, identity["sources"][scene], parent["records"][0]["latent"].shape[-2:])
            records = []
            for row, match in zip(parent["records"], source["records"], strict=True):
                harness.guarded()
                if (any(row[key] != match[key] for key in ("timestep", "position", "trajectory"))
                        or any(not torch.equal(row[key], match[key]) for key in ("latent", "neutral"))):
                    raise ValueError("The soft45 source does not match the original record")
                latent = row["latent"].to(runtime.device)
                neutral = row["neutral"].to(runtime.device)
                positive = match["positive"].to(runtime.device)
                responses = [runtime.predict(latent, row["timestep"], embedding) for embedding in probes]
                target, detail = project_exterior(neutral, positive, responses, matte)
                check_field(detail)
                records.append(row | {"positive": target.cpu()})
                fields.append({"scene": scene, "position": row["position"], "trajectory": row["trajectory"], **detail})
            destination = directory / shard["path"]
            updated = parent | {"records": records, "fingerprint": input_sha}
            if destination.exists():
                previous = torch.load(destination, map_location="cpu", weights_only=True)
                if previous["fingerprint"] != input_sha or any(
                    not torch.equal(old["positive"], new["positive"])
                    for old, new in zip(previous["records"], records, strict=True)
                ):
                    raise ValueError(f"Partial exterior target differs: {destination}")
            else:
                save_tensor_file(destination, updated)
            shards.append(shard | {"sha256": harness.pin(destination)["sha256"]})
            print(f"{method} exterior teacher cache: {scene} verified", flush=True)
    diagnostics_proof(fields)
    harness.write(directory / "diagnostics.json", fields)
    harness.write(existing, {"identity": identity, "input_fingerprint": input_sha, "shards": shards,
                             "fingerprint": digest({"input": input_sha, "shards": shards})})
    return TargetCache(directory, verify=True)


def parity(runtime, method, cache, request):
    import torch

    expected = {"model_identity": runtime.identity, "cache_fingerprint": cache.index["fingerprint"],
                "response_shape": list(cache[0]["positive"].shape)}
    if request != expected:
        return False
    original, soft45, identity = caches(runtime, method)
    evidence = []
    with torch.no_grad(), runtime.mixer.scales({}):
        for index in (3, 19, 46, 117):
            source, baseline, current = original[index], soft45[index], cache[index]
            prompts = identity["sources"][source["row"]["id"]]
            latent, step = source["latent"].to(runtime.device), source["timestep"]
            matte, probes = context(runtime, prompts, latent.shape[-2:])
            neutral = runtime.predict(latent, step, runtime.encode(prompts["neutral"]))
            positive = runtime.predict(latent, step, runtime.encode(prompts["positive"]))
            original_target = neutral + (positive - neutral)
            floor = identity["recipe"]["floor"]
            preserved, _ = apply_target({"weight": floor + (1 - floor) * matte, "blend": 0},
                                        neutral, original_target, None)
            responses = [runtime.predict(latent, step, embedding) for embedding in probes]
            target, detail = project_exterior(neutral, preserved, responses, matte)
            check_field(detail)
            checks = [torch.equal(neutral.cpu(), source["neutral"]),
                      torch.equal(original_target.cpu(), source["positive"]),
                      torch.equal(preserved.cpu(), baseline["positive"]),
                      torch.equal(latent.cpu(), current["latent"]),
                      torch.equal(neutral.cpu(), current["neutral"]),
                      torch.equal(target.cpu(), current["positive"])]
            evidence.append({"index": index, "checks": checks,
                             "max_target_difference": (target.cpu() - current["positive"]).abs().max().item()})
    success = all(all(item["checks"]) for item in evidence)
    harness.write(harness.ROOT / "cohorts" / method / "native-parity.json",
                  {"passed": success, "request": request, "fields": evidence})
    return success


if __name__ == "__main__":
    stage, _, method, *_ = sys.argv[1:]
    if stage not in ("train", "prepare"):
        raise ValueError(f"The exterior teacher does not own {stage}")
    recipe_for(method)
    with patch.multiple(harness, prepare_cache=prepare_cache, parity=parity):
        harness.main()
