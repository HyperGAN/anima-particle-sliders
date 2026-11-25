"""shared-best-seed-split: the shared main-selected teacher cache, half of each scene's fields from a second seed.

Same five scenes, prompts, teacher law (lumen_studio.cache.prepare_targets from the pinned main checkout),
100-record count and order, training options, harness and evaluation as shared-best-stock. The only change:
in each admitted scene the neutral trajectory takes seed B at odd positions and the positive trajectory
takes seed B at even positions (seed A everywhere else). Seed B is the first candidate (A+1, A+2, A+3) that
the selection's own machinery qualifies and that keeps the pooled edit norm (G3) in bounds.

GPU stages (run under shared_guard through ``optimization_guard.py``):
  assets   R0 re-render the 10 selected teacher images (sha256 must match the manifest),
           R1 rebuild the seed-A cache with prepare_targets (tensors and fingerprint must equal the shared cache),
           C  render neutral/positive for the 15 candidate (scene, seed) pairs.
  targets  T  prepare_targets on one derived single-row manifest per qualified candidate.
``prepare_cache``/``parity`` are the training-time hooks for both methods of this directory.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
STORE = Path("/mnt/nvme/sliders/experiments") / ROOT.name
SOURCE = STORE / "source"
SOURCE_COMMIT = "aaaf41adc02b7d1564c601864ea392b949389a71"
VIBE_DEPS = STORE / "vibe-deps"
OPT = WORK / "anima-final-boss-optimization-20261123"
PRIOR = WORK / "anima-final-boss-research-raw192-20261123"
SHARED = WORK / "anima-final-boss-shared-teacher-20261124"
SHARED_STORAGE = (SHARED / "storage").resolve()
SHARED_CACHE = SHARED_STORAGE / "targets/best-teacher"
SHARED_RECEIPT = SHARED_STORAGE / "targets/receipt.json"
STUDY = WORK / "anima-vibe-teacher-targets-20261124" / "research/vibe-targets/run/portfolio-policy-v2"
SELECTION = STUDY / "attempts/cli-policy-boss-verified.log"
MANIFEST = STUDY / "cli-confirm-boss/training/6a527d50d6bb2fd6c972b5349d44c3c47480731953b72ae368884655f1de84d5/manifest.json"

METHOD = "shared-best-seed-split"
STOCK = "shared-best-stock"
METHODS = (METHOD, STOCK)
LABELS = {
    METHOD: "Selected teacher fields · half of each scene from a second qualified seed",
    STOCK: "All selected teacher fields · stock sampling",
}
STEPS = 10
RESOLUTION = 512
ASSIGNMENT = ("neutral trajectory: seed A at even positions, seed B at odd positions; "
              "positive trajectory: seed B at even positions, seed A at odd positions")
LAYOUT = "shared-immutable-teacher-seed-split-v1"
G3_BOUNDS = (0.85, 1.15)  # amendment 1: applied to the overall seed-B/seed-A edit strength, not to each position
MINIMUM_SCORE = 0.75
MINIMUM_SPLIT_SCENES = 3

SEED_PLAN = ROOT / "seed-plan.json"
EVIDENCE = ROOT / "seed-b"
ASSETS_RECEIPT = EVIDENCE / "assets.json"
JUDGMENTS = EVIDENCE / "judgments.json"
TARGETS_RECEIPT = EVIDENCE / "targets.json"
QUALIFICATION = EVIDENCE / "qualification.json"
AUDIT = EVIDENCE / "target-audit.json"
QUALIFY_ROOT = STORE / "qualify"
R1_CACHE = STORE / "r1/targets/seed-a"
CACHE = STORE / "targets" / METHOD
COHORT = ROOT / "cohorts" / METHOD


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(path):
    return {"path": str(path), "sha256": file_sha(path)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def canonical_digest(value):
    """lumen_studio.contracts.digest, without importing the package (same canonical JSON)."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


# --- crossed slot rule -------------------------------------------------------------------------

def slot_order():
    return [(trajectory, position) for trajectory in ("neutral", "positive") for position in range(STEPS)]


def source_for(trajectory, position):
    """Crossed rule: neutral even->A, neutral odd->B, positive even->B, positive odd->A."""
    if trajectory not in ("neutral", "positive") or not 0 <= position < STEPS:
        raise ValueError(f"Unknown slot {trajectory}/{position}")
    return "A" if (trajectory == "neutral") == (position % 2 == 0) else "B"


def b_trajectory(position):
    """The trajectory whose slot comes from seed B at this position."""
    return "positive" if position % 2 == 0 else "neutral"


def g3_ratios(split, a_norms, b_norms):
    """split: {row: seed}. Returns pooled ratio per position and the per-scene ratios."""
    pooled, individual = [], {row: [] for row in split}
    for position in range(STEPS):
        slot = (b_trajectory(position), position)
        b = [b_norms[(row, seed)][slot] for row, seed in split.items()]
        a = [a_norms[row][slot] for row in split]
        pooled.append((sum(b) / len(b)) / (sum(a) / len(a)))
        for row, x, y in zip(split, b, a):
            individual[row].append(x / y)
    return pooled, individual


def g3_overall(split, a_norms, b_norms):
    """Overall edit strength of the seed-B slots over the seed-A slots they replace (gate amendment 1).

    geometric_mean weighs every teacher position equally (stock sampling draws positions uniformly);
    norm_weighted is total seed-B norm over total replaced seed-A norm. Per-position pooled ratios stay
    in the record as a diagnostic: inside the shared seed-A cache itself, two same-seed slots at one
    position already differ by 0.60-1.63x per scene and 0.89-1.18x pooled over its five scenes.
    """
    import math

    pooled, individual = g3_ratios(split, a_norms, b_norms)
    total_b = sum(b_norms[(row, seed)][(b_trajectory(k), k)] for row, seed in split.items() for k in range(STEPS))
    total_a = sum(a_norms[row][(b_trajectory(k), k)] for row in split for k in range(STEPS))
    return {"geometric_mean": math.exp(sum(math.log(x) for x in pooled) / len(pooled)),
            "norm_weighted": total_b / total_a,
            "per_scene_geometric_mean": {row: math.exp(sum(math.log(x) for x in individual[row]) / STEPS)
                                         for row in split}}


def g3_in_bounds(overall):
    return all(G3_BOUNDS[0] <= overall[key] <= G3_BOUNDS[1] for key in ("geometric_mean", "norm_weighted"))


def b_slots():
    return [(t, k) for t, k in slot_order() if source_for(t, k) == "B"]


# --- declared inputs ---------------------------------------------------------------------------

def shared_receipt():
    return read(SHARED_RECEIPT)


def checked_manifest():
    """The shared study's manifest, byte-pinned by its receipt (validation runs where main's modules are loaded)."""
    receipt = shared_receipt()
    manifest = read(MANIFEST)
    if file_sha(MANIFEST) != receipt["manifest_file_sha256"] or manifest["sha256"] != receipt["manifest"]:
        raise ValueError("The manifest differs from the shared study's verified teacher receipt")
    if len(manifest["rows"]) != 5 or any(len(row["seeds"]) != 1 for row in manifest["rows"]):
        raise ValueError("Expected five single-seed selected teacher rows")
    return manifest


def seed_plan(manifest=None):
    manifest = checked_manifest() if manifest is None else manifest
    plan = read(SEED_PLAN)
    rows = [row["id"] for row in manifest["rows"]]
    expected_a = {row["id"]: row["seeds"][0] for row in manifest["rows"]}
    if (plan["method"] != METHOD or plan["scene_order"] != rows or plan["seed_a"] != expected_a
            or plan["candidates"] != {row: [expected_a[row] + k for k in (1, 2, 3)] for row in rows}
            or plan["manifest"]["sha256"] != file_sha(MANIFEST)
            or plan["manifest"]["manifest_sha256"] != manifest["sha256"]):
        raise ValueError("seed-plan.json differs from the declared fixed candidate order")
    return plan


def rows_by_id(manifest):
    return {row["id"]: row for row in manifest["rows"]}


def render_key(prompt, seed):
    """The collection's render request id: digest(kind=render, inputs=(prompt, seed))."""
    return canonical_digest({"kind": "render", "inputs": {"prompt": prompt, "seed": seed}})


def r0_folder(row_id, side):
    return STORE / "assets/r0" / row_id / side


def candidate_folder(row_id, seed, side):
    return STORE / "assets/candidates" / row_id / str(seed) / side


def candidate_manifest_path(row_id, seed):
    return STORE / "candidates/manifests" / f"{row_id}-{seed}.json"


def candidate_cache(row_id, seed):
    return STORE / "candidates/targets" / f"{row_id}-{seed}"


def candidate_output(image, prompt, seed):
    """Selection-style output record of one of our renders."""
    return {"path": image["path"], "prompt": prompt, "request_id": render_key(prompt, seed), "seed": seed,
            "sha256": image["sha256"]}


def derived_manifest(manifest, row_id, seed, judged, digest):
    """Single-row manifest for one qualified candidate seed, with that seed's own judged selection.

    Passes lumen_studio.vibe.validate_manifest unchanged: the row keeps its id (round + prompt), carries
    seeds == [seed] == selection.seed, and the judgment scores, gain and outputs are the candidate's own.
    ``fitness`` is the parent collection's prompt-level value (seed independent) and is kept as is.
    """
    row = rows_by_id(manifest)[row_id]
    sides = {}
    for side in ("neutral", "positive"):
        reading = judged[side]
        sides[side] = {"decision": reading["decision"], "judgment": {"score": reading["readings"]["effect"]},
                       "output": reading["output"], "readings": reading["readings"]}
    selection = dict(row["selection"]) | sides | {
        "seed": seed, "gain": sides["positive"]["judgment"]["score"] - sides["neutral"]["judgment"]["score"]}
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    body["rows"] = [dict(row) | {"seeds": [seed], "selection": selection}]
    body["derivation"] = {
        "kind": "shared-seed-split candidate seed", "parent_manifest_sha256": manifest["sha256"],
        "row": row_id, "seed": seed, "seed_a": row["seeds"][0],
        "qualification": "lumen_studio.vibe_teacher_selection.qualifies(row, 0.75) plus the selector's second read",
        "fitness": "parent collection prompt fitness, unchanged (seed independent)",
    }
    body["sha256"] = digest(body)
    return body


# --- main checkout modules and the native runtime (GPU stages) ---------------------------------

def source_head():
    head = (SOURCE / ".git/HEAD").read_text().strip()
    if head != SOURCE_COMMIT:
        raise ValueError(f"The private main checkout is not at {SOURCE_COMMIT}: {head}")
    return head


def load_main_modules():
    """Exactly the shared prepare() order: captured runtime first, then main's cache/vibe entry points."""
    import lumen_studio
    import torch  # noqa: F401
    from lumen_studio.backends.anima import TurboRuntime  # noqa: F401
    from lumen_studio.contracts import atomic_json, file_hash  # noqa: F401

    source_head()
    if "lumen_studio.dataset" in __import__("sys").modules:
        raise RuntimeError("lumen_studio.dataset was imported from the captured product before main's modules")
    if str(SOURCE / "lumen_studio") not in lumen_studio.__path__:
        lumen_studio.__path__.insert(0, str(SOURCE / "lumen_studio"))
    from lumen_studio import cache, dataset, vibe
    from lumen_studio.contracts import digest

    files = {name: Path(module.__file__).resolve() for name, module in
             (("cache", cache), ("dataset", dataset), ("vibe", vibe))}
    if any(not path.is_relative_to(SOURCE) for path in files.values()):
        raise RuntimeError(f"main's cache entry points did not load from {SOURCE}: {files}")
    return {"cache": cache, "vibe": vibe, "digest": digest,
            "files": {name: pin(path) for name, path in files.items()}}


def native_runtime(read_json=read):
    """The shared prepare() runtime: same model mirror, memory cap, TF32 off, default TurboRuntime."""
    import torch
    from lumen_studio.backends.anima import TurboRuntime
    from lumen_studio.provenance import model_identity

    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(19 * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    expected = checked_manifest()["collection_identity"]["product"]
    model = Path(read_json(PRIOR / "nvme-execution.json")["mirror_model"])
    if model_identity(model) != expected["model"] or (expected["resolution"], expected["steps"], expected["cfg"]) != (512, 10, 1):
        raise ValueError("The merged teacher selection uses a different native model or sampler")
    return TurboRuntime(model, device="cuda:0", verify=True)


def vibe_render(runtime, prompt, seed, folder):
    """lumen_studio.vibe.AnimaVibe.render, byte for byte: scales({}), render(512, 512, 10 steps), .save(image.png)."""
    receipt = folder / "render.json"
    png = folder / "image.png"
    args = {"call": "AnimaVibe.render", "prompt": prompt, "seed": seed, "width": RESOLUTION,
            "height": RESOLUTION, "steps": STEPS, "model": runtime.identity}
    if receipt.exists():
        saved = read(receipt)
        if saved != args | {"image": pin(png)}:
            raise RuntimeError(f"Render identity changed: {folder}")
        return saved["image"]
    folder.mkdir(parents=True, exist_ok=True)
    with runtime.mixer.scales({}):
        runtime.render(prompt, seed, width=RESOLUTION, height=RESOLUTION, steps=STEPS).save(png)
    image = pin(png)
    write_json(receipt, args | {"image": image})
    return image


def same(left, right):
    import torch

    if isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(right, dict):
        return isinstance(left, dict) and left.keys() == right.keys() and all(same(left[k], v) for k, v in right.items())
    if isinstance(right, list):
        return isinstance(left, list) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def compare_caches(mine, theirs):
    """Every shard tensor torch.equal and the cache fingerprint equal (R1)."""
    import torch

    left, right = read(Path(mine) / "index.json"), read(Path(theirs) / "index.json")
    mismatches = []
    for key in ("identity", "input_fingerprint", "fingerprint"):
        if left[key] != right[key]:
            mismatches.append(f"index/{key}")
    shape = [(s["path"], s["row"], s["seed"], s["count"]) for s in left["shards"]]
    if shape != [(s["path"], s["row"], s["seed"], s["count"]) for s in right["shards"]]:
        mismatches.append("index/shards")
    else:
        for item in left["shards"]:
            a = torch.load(Path(mine) / item["path"], map_location="cpu", weights_only=True)
            b = torch.load(Path(theirs) / item["path"], map_location="cpu", weights_only=True)
            for key in b:
                if key == "records":
                    for j, (x, y) in enumerate(zip(a["records"], b["records"], strict=True)):
                        mismatches += [f"{item['row']}/{j}/{name}" for name in y if not same(x.get(name), y[name])]
                elif not same(a.get(key), b[key]):
                    mismatches.append(f"{item['row']}/{key}")
            if a.keys() != b.keys():
                mismatches.append(f"{item['row']}/keys")
            del a, b
    file_equal = [s["sha256"] for s in left["shards"]] == [s["sha256"] for s in right["shards"]]
    return {"passed": not mismatches, "fingerprint": left["fingerprint"], "expected_fingerprint": right["fingerprint"],
            "shard_files_equal": file_equal, "mismatch_count": len(mismatches), "mismatches": mismatches[:40]}


def assets(runtime, guard, modules):
    """GPU stage ``assets``: R0, R1, then candidate renders (resumable; any reproduction mismatch stops)."""
    manifest = checked_manifest()
    modules["vibe"].validate_manifest(manifest)
    plan = seed_plan(manifest)
    if runtime.identity != shared_receipt()["model"]:
        raise ValueError("The native runtime differs from the shared teacher cache's runtime")
    receipt = {"passed": False, "model": runtime.identity, "seed_plan": pin(SEED_PLAN), "manifest": pin(MANIFEST),
               "source_commit": source_head(), "modules": modules["files"],
               "implementation": pin(Path(__file__).resolve())}
    r0 = []
    for row in manifest["rows"]:
        for side in ("neutral", "positive"):
            guard()
            output = row["selection"][side]["output"]
            if output["prompt"] != row[side] or output["seed"] != row["seeds"][0]:
                raise ValueError(f"Selected output does not belong to {row['id']}/{side}")
            image = vibe_render(runtime, row[side], output["seed"], r0_folder(row["id"], side))
            r0.append({"row": row["id"], "side": side, "seed": output["seed"], "produced": image,
                       "expected_sha256": output["sha256"], "request_id": render_key(row[side], output["seed"]),
                       "expected_request_id": output["request_id"]})
    receipt["R0"] = {"passed": all(item["produced"]["sha256"] == item["expected_sha256"]
                                   and item["request_id"] == item["expected_request_id"] for item in r0), "images": r0}
    if not receipt["R0"]["passed"]:
        write_json(ASSETS_RECEIPT, receipt | {"failed_gate": "R0"})
        raise RuntimeError("R0: a re-rendered selected teacher image differs from the manifest sha256")
    print("R0 passed: 10 selected teacher images reproduced byte for byte", flush=True)
    guard()
    index = modules["cache"].prepare_targets(runtime, manifest, manifest["variation"], R1_CACHE, resolution=RESOLUTION,
                                             progress=lambda done, total: print(f"R1 teacher {done}/{total}", flush=True))
    receipt["R1"] = compare_caches(R1_CACHE, SHARED_CACHE) | {"index": pin(R1_CACHE / "index.json"),
                                                               "prepared_fingerprint": index["fingerprint"]}
    if not receipt["R1"]["passed"]:
        write_json(ASSETS_RECEIPT, receipt | {"failed_gate": "R1"})
        raise RuntimeError("R1: the rebuilt seed-A cache differs from the shared teacher cache")
    print("R1 passed: rebuilt seed-A cache equals the shared cache", flush=True)
    candidates = {}
    rows = rows_by_id(manifest)
    for row_id in plan["scene_order"]:
        row = rows[row_id]
        candidates[row_id] = {}
        for seed in plan["candidates"][row_id]:
            entry = {}
            for side in ("neutral", "positive"):
                guard()
                image = vibe_render(runtime, row[side], seed, candidate_folder(row_id, seed, side))
                entry[side] = candidate_output(image, row[side], seed)
            candidates[row_id][str(seed)] = entry
            print(f"candidate renders {row_id[:13]} seed {seed}", flush=True)
    receipt |= {"passed": True, "candidates": candidates}
    write_json(ASSETS_RECEIPT, receipt)
    print("assets complete", flush=True)
    return receipt


def qualified_candidates(judgments):
    return [(row, item["seed"]) for row in judgments["scene_order"]
            for item in judgments["candidates"][row] if item["qualified"]]


def targets(runtime, guard, modules):
    """GPU stage ``targets``: prepare_targets on the derived manifest of every qualified candidate."""
    manifest = checked_manifest()
    seed_plan(manifest)
    judgments = read(JUDGMENTS)
    if judgments.get("status") != "judged" or judgments["assets"] != pin(ASSETS_RECEIPT):
        raise RuntimeError("Run qualify.py judge after the assets stage first")
    if runtime.identity != shared_receipt()["model"]:
        raise ValueError("The native runtime differs from the shared teacher cache's runtime")
    caches = []
    for row_id, seed in qualified_candidates(judgments):
        entry = next(item for item in judgments["candidates"][row_id] if item["seed"] == seed)
        path = candidate_manifest_path(row_id, seed)
        derived = read(path)
        if derived != derived_manifest(manifest, row_id, seed, entry, modules["digest"]) or \
                pin(path) != entry["derived_manifest"]:
            raise ValueError(f"Derived manifest changed for {row_id} seed {seed}")
        modules["vibe"].validate_manifest(derived)
        guard()
        output = candidate_cache(row_id, seed)
        index = modules["cache"].prepare_targets(runtime, derived, derived["variation"], output, resolution=RESOLUTION)
        if (len(index["shards"]) != 1 or index["shards"][0]["row"] != row_id or index["shards"][0]["seed"] != seed
                or index["shards"][0]["count"] != 20 or index["identity"]["manifest_sha256"] != derived["sha256"]
                or index["identity"]["model"] != runtime.identity or derived["variation"] != manifest["variation"]):
            raise ValueError(f"Candidate cache for {row_id} seed {seed} is not one 20-field shard")
        caches.append({"row": row_id, "seed": seed, "directory": str(output), "fingerprint": index["fingerprint"],
                       "index": pin(output / "index.json"), "manifest": pin(path)})
        print(f"candidate fields {row_id[:13]} seed {seed}: {index['fingerprint'][:12]}", flush=True)
    receipt = {"passed": True, "model": runtime.identity, "judgments": pin(JUDGMENTS), "caches": caches,
               "modules": modules["files"], "source_commit": source_head(),
               "implementation": pin(Path(__file__).resolve())}
    write_json(TARGETS_RECEIPT, receipt)
    print(f"targets complete: {len(caches)} candidate caches", flush=True)
    return receipt


# --- training-time hooks -----------------------------------------------------------------------

def checked_shared_cache(runtime):
    """shared_data.prepare_cache's receipt checks, verbatim."""
    from lumen_studio.cache import TargetCache

    receipt = shared_receipt()
    manifest = read(MANIFEST)
    source = TargetCache(SHARED_CACHE, verify=True)
    if (manifest["sha256"] != receipt["manifest"] or file_sha(MANIFEST) != receipt["manifest_file_sha256"]
            or source.index["fingerprint"] != receipt["cache"] or len(source) != receipt["targets"]
            or file_sha(SHARED_CACHE / "index.json") != receipt["cache_index_sha256"]
            or (runtime is not None and runtime.identity != receipt["model"])):
        raise ValueError("Expected the single verified main-selected teacher cache")
    return source


def admitted_cache(runtime=None):
    """Load and verify the assembled crossed cache; never builds anything."""
    from lumen_studio.cache import TargetCache

    problem = None
    try:
        source = checked_shared_cache(runtime)
        qualification, audit = read(QUALIFICATION), read(AUDIT)
        cache = TargetCache(CACHE, verify=True)
        identity = cache.index["identity"]
        if not qualification.get("passed"):
            problem = "qualification has not passed"
        elif not audit.get("passed"):
            problem = "target audit has not passed"
        elif audit.get("cache_fingerprint") != cache.index["fingerprint"] or audit.get("index") != pin(CACHE / "index.json"):
            problem = "target audit is pinned to a different cache"
        elif identity.get("qualification_sha256") != file_sha(QUALIFICATION):
            problem = "cache is pinned to a different qualification"
        elif identity.get("layout") != LAYOUT or identity.get("parent") != source.index["fingerprint"]:
            problem = "cache layout or parent differs"
        elif len(cache) != len(source) or [s["path"] for s in cache.index["shards"]] != [s["path"] for s in source.index["shards"]]:
            problem = "cache does not hold the shared record count and order"
        elif runtime is not None and identity.get("model") != runtime.identity:
            problem = "cache model identity differs from the native runtime"
    except (OSError, KeyError, ValueError) as error:
        problem = f"{type(error).__name__}: {error}"
    if problem:
        raise RuntimeError(f"Run assemble.py after qualification ({problem})")
    return cache


def prepare_cache(runtime, method):
    if method == STOCK:
        return checked_shared_cache(runtime)  # shared_data.derive returns the source cache for stock
    if method == METHOD:
        return admitted_cache(runtime)
    raise ValueError(f"Unknown teacher cache {method}")


def parity_indices(cache):
    """shared_data.parity's one slot per scene, plus two A and two B slots in every split scene."""
    seed_map = cache.index["identity"].get("seed_map", {})
    stride = cache.index["shards"][0]["count"]
    order = slot_order()
    chosen = []
    for scene, shard in enumerate(cache.index["shards"]):
        picks = [(scene + 3) % stride]
        if seed_map.get(shard["row"], {}).get("b") is not None:
            picks += [order.index(slot) for slot in (("neutral", 2), ("positive", 7), ("neutral", 5), ("positive", 8))]
        chosen += [scene * stride + j for j in dict.fromkeys(picks)]
    return chosen


def parity(runtime, method, cache, request):
    import torch
    from lumen_studio.cache import TargetCache

    if request != {"model_identity": runtime.identity, "cache_fingerprint": cache.index["fingerprint"],
                   "response_shape": list(cache[0]["positive"].shape)}:
        return False
    guard = __import__("optimization_worker").guarded
    seed_map = cache.index["identity"].get("seed_map", {})
    fields = []
    indices = parity_indices(cache) if method == METHOD else [
        scene * cache.index["shards"][0]["count"] + (scene + 3) % cache.index["shards"][0]["count"]
        for scene in range(len(cache.index["shards"]))]
    with torch.no_grad(), runtime.mixer.scales({}):
        for index in indices:
            guard()
            row = cache[index]
            neutral, positive = (runtime.encode(row["row"][key]) for key in ("neutral", "positive"))
            latent = row["latent"].to(runtime.device)
            seed = seed_map.get(row["row"]["id"], {})
            fields.append({
                "index": index, "row": row["row"]["id"], "position": row["position"], "trajectory": row["trajectory"],
                "seed": ("B" if seed.get("b") is not None and source_for(row["trajectory"], row["position"]) == "B"
                         else "A"),
                "embedding": torch.equal(neutral.cpu(), row["embedding"]),
                "neutral": torch.equal(runtime.predict(latent, row["timestep"], neutral).cpu(), row["neutral"]),
                "positive": torch.equal(runtime.predict(latent, row["timestep"], positive).cpu(), row["positive"]),
            })
    passed = all(all(row[key] for key in ("embedding", "neutral", "positive")) for row in fields)
    evidence = {"passed": passed, "request": request, "fields": fields}
    if method == METHOD:
        source = TargetCache(SHARED_CACHE, verify=True)
        copied = split = 0
        for index in range(len(cache)):
            record, other = cache[index], source[index]
            if record["row"]["id"] != other["row"]["id"] or (record["trajectory"], record["position"], record["timestep"]) != (
                    other["trajectory"], other["position"], other["timestep"]):
                passed = False
                break
            is_b = seed_map[record["row"]["id"]]["b"] is not None and source_for(record["trajectory"], record["position"]) == "B"
            equal = all(torch.equal(record[key], other[key]) for key in ("latent", "neutral", "positive"))
            copied += equal and not is_b
            split += is_b and not torch.equal(record["latent"], other["latent"])
        crossed = [row for row, entry in seed_map.items() if entry["b"] is not None]
        scenes = len(crossed)
        covered = all(sum(f["row"] == row and f["seed"] == seed for f in fields) >= 2
                      for row in crossed for seed in ("A", "B"))
        passed = passed and covered and copied == len(cache) - 10 * scenes and split == 10 * scenes
        evidence |= {"passed": passed, "seed_a_records_equal_to_shared": copied, "seed_b_records": split}
    write_json(ROOT / "cohorts" / method / "native-parity.json", evidence)
    return passed
