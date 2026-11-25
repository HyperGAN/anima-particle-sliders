"""Capture teacher fields on the frozen soft-floor45 adapter's deployed path."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from unittest.mock import patch

import optimization_worker as harness
from optimization_targets import encode_context, plan, sources


METHOD = "soft45-adapter-rollout"
RECIPE = {"method": METHOD, "teacher": "adapter-rollout-v1", "floor": .45, "donor": None, "blend": 0}
REFERENCE = harness.ROOT / "cohorts" / "soft-floor45"
RAW_PATH = harness.STORAGE / "soft-floor45" / "training" / "run-a1" / "RAW000192.safetensors"
RAW = {"path": str(RAW_PATH), "bytes": 33737392,
       "sha256": "5ada5db653b1ebfe543075ed52ad47e1118cfa975fee86ab4f132eb5d793c92b"}
ALPHA = 2 / 3
STEPS = 10
SCENES = ("bridge", "cafe", "greenhouse", "harbor", "library", "mall")
WITNESSES = (3, 19, 46, 117)
INPUT_PINS = {
    harness.BASE_CACHE / "index.json": "fb5bb6ae35fcc7cf161826d725e66542ae28e880c4ce65f21dd5e2459b09b718",
    REFERENCE / "target-cache" / "index.json": "3eeab399fca2673bc7ef9cd92b5ed6d4496378042349a15c87cc877279608c6b",
    REFERENCE / "result.json": "92b523522f9de65303faace80eec9edd3add337227b8ce6e00d89669840b16e3",
    REFERENCE / "native-parity.json": "29b4f223a8294404c789eb2a373ba4a4d8127b23e4e8c9ec55bf12c9879ef787",
    REFERENCE / "milestones" / "RAW000192.json": "d3dc793cb550939ea7d6c848c6a0a45a8e5c94fdde6dd785a40e4565177f0029",
    RAW_PATH.with_suffix(".scoped.json"): "4ac9d321334bca9f504f7275ba784e6d7fc3bb781ada8c444df766c79d675ab4",
    harness.PRODUCT / "schedules.json": "d68b0f93ebcdff8f52172f4b268d11f5fd3904b7df831f6702281cf6fc7c156a",
}


def recipe_for(method):
    if method != METHOD or plan(method) != RECIPE:
        raise ValueError(f"Expected the exact {METHOD} training recipe")
    return RECIPE.copy()


def rollout_identity(original, control):
    pins = [harness.pin(path) for path in INPUT_PINS]
    if any(pin["sha256"] != sha for pin, sha in zip(pins, INPUT_PINS.values(), strict=True)):
        raise ValueError("An input to the frozen adapter rollout changed")
    result = harness.read(REFERENCE / "result.json")
    milestone = harness.read(REFERENCE / "milestones" / "RAW000192.json")
    parity = harness.read(REFERENCE / "native-parity.json")
    schedule = harness.read(harness.PRODUCT / "schedules.json")
    sidecar = harness.read(RAW_PATH.with_suffix(".scoped.json"))
    if (harness.pin(RAW_PATH)["sha256"] != RAW["sha256"] or RAW_PATH.stat().st_size != RAW["bytes"]
            or result["raw"] != RAW or result["step"] != 192 or result["ema_decay"] is not None
            or sidecar["native_RAW_sha256"] != RAW["sha256"] or sidecar["completed_steps"] != 192
            or sidecar["selection"] != "raw" or sidecar["ema_decay"] is not None
            or sidecar["model_identity"] != original["identity"]["model"]
            or sidecar["cache_identity"]["index"] != control
            or result["cache"] != control["fingerprint"] or not parity["passed"]
            or control["identity"]["parent"] != original["fingerprint"]
            or milestone["step"] != 192 or milestone["gains"] != schedule["strengths"]["early"]
            or schedule["weights"] != "raw" or schedule["ema_decay"] is not None):
        raise ValueError("Rollout donor is not the pinned completed soft-floor45 RAW192")
    return {"raw": RAW.copy(), "source": "soft-floor45", "inputs": pins, "checkpoint_step": 192,
            "source_cache": control["fingerprint"], "weights": "raw", "ema_decay": None,
            "profile": "early", "alpha": ALPHA, "sigmas": schedule["sigmas"], "gains": milestone["gains"],
            "conditioning": "native neutral prompt", "position_zero": "native training seed noise"}


def cache_identity(original, method):
    recipe = recipe_for(method)
    control = harness.read(REFERENCE / "target-cache" / "index.json")
    rows = {item["row"]["id"]: item["row"] for item in original["identity"]["teachers"]}
    if tuple(rows) != SCENES or original["identity"]["steps"] != STEPS:
        raise ValueError("Unexpected native scene order or number of training positions")
    inputs = [Path(__file__).resolve(), harness.ROOT / "optimization_worker.py",
              harness.ROOT / "optimization_targets.py", harness.ROOT / "plans" / f"{method}.json",
              harness.PRODUCT / "fit.py", harness.PRODUCT / "replay.py"]
    return json.loads(json.dumps(original["identity"] | {
        "parent": original["fingerprint"], "layout": f"adapter-rollout/{method}/v1",
        "implementation": [harness.pin(path) for path in inputs], "recipe": recipe,
        "donor_source": None, "sources": sources(rows), "rollout": rollout_identity(original, control),
        "target_norm": "L2 of the archived native contrast at the same scene, trajectory and position",
        "neutral_records": "exact soft-floor45 records",
    }))


@contextmanager
def frozen_adapter(runtime, identity):
    import torch
    from replay import loaded

    mixer = runtime.mixer
    key = f"{harness.VARIATION}-192"
    if key in mixer.adapters:
        raise ValueError("Adapter rollout needs an unused mixer name")
    old_handles = tuple(mixer.handles)
    old_strengths = mixer.strengths.copy()
    try:
        # Constructing the temporary inference adapter must not consume the fit RNG.
        with torch.random.fork_rng(devices=[]):
            name = loaded(runtime, identity["raw"], identity["checkpoint_step"])
        if name != key:
            raise ValueError("Unexpected frozen rollout attachment")
        mixer.adapters[key].eval().requires_grad_(False)
        yield key
    finally:
        mixer.adapters.pop(key, None)
        mixer.strengths = old_strengths
        for handle in tuple(mixer.handles):
            if handle not in old_handles:
                handle.remove()
                mixer.handles.remove(handle)


def normalized_target(weight, neutral, positive, source_norm):
    import torch

    edit = (positive - neutral) * weight
    measured = edit.norm()
    if source_norm.item() <= 1e-10 or measured.item() <= 1e-10:
        raise ValueError("A native teacher contrast is degenerate")
    target = neutral + edit * (source_norm / measured)
    if not bool(torch.isfinite(target).all()):
        raise ValueError("The native teacher target is nonfinite")
    return target, measured.item()


def capture_scene(runtime, key, identity, parent, prior):
    import torch
    from lumen_studio_scoped.policy import tensor_sha
    from lumen_studio_scoped.training import equal_state

    row = parent["row"]
    scene = row["id"]
    scheduler = runtime.scheduler(STEPS)
    schedule = identity["rollout"]
    if scheduler.sigmas.tolist() != schedule["sigmas"] or harness.gains(runtime, ALPHA, "early") != schedule["gains"]:
        raise ValueError("The live sampler differs from the predeclared common early schedule")
    if (len(parent["records"]) != 2 * STEPS or len(prior["records"]) != 2 * STEPS
            or any(not equal_state(parent[name], prior[name])
                   for name in ("row", "seed", "embedding", "positive_embedding"))):
        raise ValueError("The original and soft-floor45 teacher rows differ")
    context = encode_context(runtime, row, "soft-floor45", parent["records"][0]["latent"].shape[-2:])
    if any(not equal_state(context[name], parent[archive]) for name, archive in
           (("neutral", "embedding"), ("positive", "positive_embedding"))):
        raise ValueError("The native conditioning differs from its exact archived embeddings")
    latent = runtime.noise(parent["seed"], 512, 512)
    if not equal_state(latent, parent["records"][0]["latent"]):
        raise ValueError("The original training-seed noise differs")

    fields = list(prior["records"][:STEPS])
    diagnostics, trace = [], []
    changed = changed_from_neutral = 0
    for position, (step, gain) in enumerate(zip(scheduler.timesteps, schedule["gains"], strict=True)):
        harness.guarded()
        for track, index in (("neutral", position), ("positive", STEPS + position)):
            base, control = parent["records"][index], prior["records"][index]
            if (base["trajectory"], base["position"], base["timestep"]) != (track, position, float(step)) or any(
                not equal_state(base[name], control[name]) for name in base if name != "positive"
            ):
                raise ValueError("Original state layout differs from the frozen soft-floor45 teacher")
            source_norm = (base["positive"].to(runtime.device) - base["neutral"].to(runtime.device)).norm()
            if track == "neutral":
                record = control
            else:
                with runtime.mixer.scales({}):
                    neutral = runtime.predict(latent, step, context["neutral"])
                    positive = runtime.predict(latent, step, context["positive"])
                # The archived full-frame teacher constructs its positive field this way,
                # including the float32 rounding from subtracting and adding the base.
                inherited = neutral + (positive - neutral)
                target, weighted_norm = normalized_target(context["weight"], neutral, inherited, source_norm)
                record = base | {"latent": latent.cpu().clone(), "neutral": neutral.cpu(), "positive": target.cpu()}
                fields.append(record)
                if position == 0 and (not equal_state(record, control)
                                      or not equal_state(inherited, base["positive"])):
                    raise ValueError("The rollout changed fields at the identical initial state")
                different = not equal_state(record["latent"], base["latent"])
                neutral_path = parent["records"][position]["latent"]
                differs_from_neutral = not equal_state(record["latent"], neutral_path)
                changed += int(different)
                changed_from_neutral += int(differs_from_neutral)
                with runtime.mixer.scales({key: gain}):
                    velocity = runtime.predict(latent, step, context["neutral"])
                if not bool(torch.isfinite(velocity).all()):
                    raise ValueError("The frozen adapter rollout is nonfinite")
                next_latent = runtime.step(scheduler, velocity, step, latent)
                if not bool(torch.isfinite(next_latent).all()):
                    raise ValueError("The native rollout step produced a nonfinite state")
                trace.append({"position": position, "timestep": float(step), "gain": gain,
                              "original_latent_sha256": tensor_sha(base["latent"]),
                              "neutral_path_latent_sha256": tensor_sha(neutral_path),
                              "latent_sha256": tensor_sha(latent), "next_latent_sha256": tensor_sha(next_latent),
                              "rollout_velocity_sha256": tensor_sha(velocity), "neutral_sha256": tensor_sha(neutral),
                              "positive_teacher_sha256": tensor_sha(positive), "target_sha256": tensor_sha(target),
                              "weighted_norm_before_match": weighted_norm, "replaced": different,
                              "differs_from_neutral": differs_from_neutral})
                latent = next_latent
            source_l2 = source_norm.item()
            target_l2 = (record["positive"].to(runtime.device) - record["neutral"].to(runtime.device)).norm().item()
            if source_l2 <= 1e-10 or abs(target_l2 / source_l2 - 1.) >= 1e-5:
                raise ValueError("A target differs from the original same-position native edit norm")
            diagnostics.append({"scene": scene, "position": position, "trajectory": track,
                                "source_l2": source_l2, "target_l2": target_l2})
    if not changed or not changed_from_neutral:
        raise ValueError(f"The adapter reproduced an original trajectory for {scene}")
    diagnostics.sort(key=lambda row: (row["trajectory"] != "neutral", row["position"]))
    return fields, diagnostics, {"scene": scene, "seed": parent["seed"], "neutral_prompt": row["neutral"],
                                "positive_prompt": row["positive"], "steps": trace,
                                "changed_positive_states": changed,
                                "changed_from_neutral_states": changed_from_neutral}


def verified_sources(runtime):
    from lumen_studio.cache import TargetCache

    original = TargetCache(harness.BASE_CACHE, verify=True)
    prior = TargetCache(REFERENCE / "target-cache", verify=True)
    for cache in (original, prior):
        if cache.index["identity"]["model"] != runtime.identity or len(cache) != 120:
            raise ValueError("The exact full-frame training cache differs from this runtime")
        if tuple(row["row"] for row in cache.index["shards"]) != SCENES:
            raise ValueError("The teacher shard order differs from the declared six scenes")
    return original, prior


def captures(runtime, original, prior, identity):
    import torch

    with torch.no_grad(), frozen_adapter(runtime, identity["rollout"]) as key:
        for source, control in zip(original.index["shards"], prior.index["shards"], strict=True):
            if {k: v for k, v in source.items() if k != "sha256"} != {
                k: v for k, v in control.items() if k != "sha256"
            }:
                raise ValueError("The two archives disagree on training coordinates")
            parent = torch.load(original.directory / source["path"], map_location="cpu", weights_only=True)
            saved = torch.load(prior.directory / control["path"], map_location="cpu", weights_only=True)
            if any(data["fingerprint"] != cache.index["input_fingerprint"] or data["seed"] != source["seed"]
                   or data["row"]["id"] != source["row"] for data, cache in ((parent, original), (saved, prior))):
                raise ValueError("A source shard is bound to a different training context")
            records, diagnostics, trace = capture_scene(runtime, key, identity, parent, saved)
            yield source, parent | {"records": records, "adapter_rollout": trace}, diagnostics, trace


def write_binding(path, value):
    if path.exists():
        if harness.read(path) != value:
            raise ValueError(f"Frozen adapter-rollout receipt differs: {path}")
    else:
        harness.write(path, value)


def prepare_cache(runtime, method):
    import torch
    from lumen_studio.cache import TargetCache, save_tensor_file
    from lumen_studio.contracts import digest
    from lumen_studio_scoped.training import equal_state

    folder = harness.ROOT / "cohorts" / method / "target-cache"
    original, prior = verified_sources(runtime)
    identity = cache_identity(original.index, method)
    fingerprint = digest(identity)
    if (folder / "index.json").exists():
        cache = TargetCache(folder, verify=True)
        if cache.index["identity"] != identity or cache.index["input_fingerprint"] != fingerprint or len(cache) != 120:
            raise ValueError("An existing cache differs from the predeclared adapter rollout")
        return cache

    folder.mkdir(parents=True, exist_ok=True)
    shards, diagnostics, traces = [], [], []
    for source, data, detail, trace in captures(runtime, original, prior, identity):
        path = folder / source["path"]
        data["fingerprint"] = fingerprint
        if path.exists():
            if not equal_state(torch.load(path, map_location="cpu", weights_only=True), data):
                raise ValueError(f"A partial cache shard disagrees with native replay: {path}")
        else:
            save_tensor_file(path, data)
        shards.append(source | {"sha256": harness.pin(path)["sha256"]})
        diagnostics.extend(detail)
        traces.append(trace)
        print(f"{method} teacher cache: {trace['scene']} verified", flush=True)
    index = {"identity": identity, "input_fingerprint": fingerprint, "shards": shards,
             "fingerprint": digest({"input": fingerprint, "shards": shards})}
    write_binding(folder / "diagnostics.json", diagnostics)
    write_binding(folder / "rollout-provenance.json", {"schema": "anima-adapter-rollout-v1",
                  "cache_fingerprint": index["fingerprint"], "source": identity["rollout"], "scenes": traces})
    write_binding(folder / "index.json", index)
    return TargetCache(folder, verify=True)


def parity(runtime, method, cache, request):
    import torch
    from lumen_studio.contracts import digest
    from lumen_studio_scoped.training import equal_state

    expected = {"model_identity": runtime.identity, "cache_fingerprint": cache.index["fingerprint"],
                "response_shape": list(cache[0]["positive"].shape)}
    if request != expected:
        return False
    original, prior = verified_sources(runtime)
    identity = cache_identity(original.index, method)
    if cache.index["identity"] != identity or cache.index["input_fingerprint"] != digest(identity):
        raise ValueError("Native admission received a different rollout identity")
    evidence, traces, diagnostics = [], [], []
    for scene_index, (source, data, detail, trace) in enumerate(captures(runtime, original, prior, identity)):
        actual = torch.load(cache.directory / source["path"], map_location="cpu", weights_only=True)
        data["fingerprint"] = cache.index["input_fingerprint"]
        if not equal_state(actual, data):
            raise ValueError(f"Native RAW192 rollout replay differs for {trace['scene']}")
        traces.append(trace)
        diagnostics.extend(detail)
        for index in WITNESSES:
            if index // (2 * STEPS) == scene_index:
                evidence.append({"index": index, "checks": [True], "max_target_difference": 0.})

    provenance = cache.directory / "rollout-provenance.json"
    if (harness.read(provenance) != {"schema": "anima-adapter-rollout-v1",
                                    "cache_fingerprint": cache.index["fingerprint"],
                                    "source": identity["rollout"], "scenes": traces}
            or harness.read(cache.directory / "diagnostics.json") != diagnostics):
        raise ValueError("The rollout provenance differs from deterministic native replay")
    write_binding(cache.directory.parent / "native-parity.json", {
        "passed": True, "request": request, "fields": evidence,
        "rollout": {"provenance": harness.pin(provenance), "replayed_positive_records": 60,
                    "exact_neutral_records": 60,
                    "changed_positive_states": sum(trace["changed_positive_states"] for trace in traces),
                    "changed_from_neutral_states": sum(trace["changed_from_neutral_states"] for trace in traces)},
    })
    return True


if __name__ == "__main__":
    stage, _, method, *_ = sys.argv[1:]
    if stage not in ("prepare", "train"):
        raise ValueError(f"The adapter rollout teacher does not own {stage}")
    recipe_for(method)
    with patch.multiple(harness, prepare_cache=prepare_cache, parity=parity):
        harness.main()
