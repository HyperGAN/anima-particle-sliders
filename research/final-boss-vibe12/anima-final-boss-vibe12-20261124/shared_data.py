"""Use the selected teacher fields verbatim, changing only training row exposure."""

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from lumen_studio.cache import TargetCache, save_tensor_file
from lumen_studio.contracts import atomic_json, digest, file_hash

from control import CACHE, MANIFEST, STORAGE
from optimization_worker import guarded, read


LABELS = {
    "shared-best-stock": "All selected teacher fields · stock sampling",
    "shared-best-opening": "Repeat the opening four positions",
    "shared-best-middle": "Repeat the middle four positions",
    "shared-best-neutral-path": "Neutral trajectory only",
    "shared-best-mixed-path": "Alternate neutral and positive trajectories",
    "shared-best-uniform-steps": "Six evenly spaced timestep positions",
    "shared-best-divergence": "Six highest-divergence timestep positions",
}
COORDINATES = [(trajectory, step) for trajectory in ("neutral", "positive") for step in range(10)]


def select_records(method, scene, rows):
    if method in ("shared-best-opening", "shared-best-middle"):
        start = 0 if method == "shared-best-opening" else 3
        return rows + [row for row in rows if start <= row["position"] < start + 4]
    if method == "shared-best-neutral-path":
        return rows[:10]
    if method == "shared-best-mixed-path":
        return [rows[step + 10 * ((step + scene) % 2)] for step in range(10)]
    if method == "shared-best-uniform-steps":
        positions = (0, 2, 4, 5, 7, 9)
    elif method == "shared-best-divergence":
        energies = [sum((rows[index]["positive"] - rows[index]["neutral"]).square().mean().item()
                        for index in (step, step + 10)) for step in range(10)]
        positions = sorted(range(10), key=lambda step: (-energies[step], step))[:6]
    else:
        raise ValueError(f"Unknown sampling recipe {method}")
    return [row for row in rows if row["position"] in positions]


def equal(left, right):
    if isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(right, dict):
        return left.keys() == right.keys() and all(equal(left[key], value) for key, value in right.items())
    if isinstance(right, list):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def derive(source, method):
    if method == "shared-best-stock":
        return source
    dest = STORAGE / "targets" / method
    identity = source.index["identity"] | {
        "layout": "shared-immutable-teacher-sampling-v1",
        "parent": source.index["fingerprint"],
        "sampling_recipe": method,
        "sampling_source_sha256": file_hash(Path(__file__)),
    }
    fingerprint = digest(identity)
    existing = TargetCache(dest) if dest.is_dir() else None
    entries = []
    counts = Counter()
    with TemporaryDirectory(dir=dest.parent, prefix=".sampling-") as temporary:
        staging = Path(temporary) / "cache"
        staging.mkdir()
        for scene, item in enumerate(source.index["shards"]):
            if Path(item["path"]).name != item["path"]:
                raise ValueError(f"Invalid teacher shard path {item['path']}")
            shard = torch.load(source.directory / item["path"], map_location="cpu", weights_only=True)
            records = shard["records"]
            if (shard["fingerprint"] != source.index["input_fingerprint"]
                    or (shard["row"]["id"], shard["seed"]) != (item["row"], item["seed"])
                    or [(row["trajectory"], row["position"]) for row in records] != COORDINATES):
                raise ValueError(f"Teacher fields changed for {item['row']}")
            chosen = select_records(method, scene, records)
            counts.update(row["position"] for row in chosen)
            target = (dest if existing else staging) / item["path"]
            expected = shard | {"fingerprint": fingerprint, "records": chosen}
            if existing is None:
                save_tensor_file(target, expected)
            if not equal(torch.load(target, map_location="cpu", weights_only=True), expected):
                raise ValueError(f"Sampled teacher fields differ for {item['row']}")
            entries.append(item | {"count": len(chosen), "sha256": file_hash(target)})
        index = {"identity": identity, "input_fingerprint": fingerprint, "shards": entries,
                 "fingerprint": digest({"input": fingerprint, "shards": entries})}
        if existing is None:
            atomic_json(staging / "index.json", index)
            staging.rename(dest)
        elif existing.index != index:
            raise ValueError(f"Sampling cache identity changed for {method}")
    atomic_json(STORAGE / "targets" / f"{method}.json", {
        "source_cache": source.index["fingerprint"], "cache": index["fingerprint"],
        "target_tensors_verified": True, "rows_by_position": counts,
    })
    return TargetCache(dest, verify=True)


def prepare_cache(runtime, method):
    receipt = read(STORAGE / "targets/receipt.json")
    manifest = read(MANIFEST)
    source = TargetCache(CACHE, verify=True)
    if (manifest["sha256"] != receipt["manifest"] or file_hash(MANIFEST) != receipt["manifest_file_sha256"]
            or source.index["fingerprint"] != receipt["cache"] or len(source) != receipt["targets"]
            or file_hash(CACHE / "index.json") != receipt["cache_index_sha256"]
            or runtime.identity != receipt["model"]):
        raise ValueError("Expected the single verified main-selected teacher cache")
    return derive(source, method)


def parity(runtime, method, cache, request):
    from control import ROOT

    if request != {"model_identity": runtime.identity, "cache_fingerprint": cache.index["fingerprint"],
                   "response_shape": list(cache[0]["positive"].shape)}:
        return False
    fields = []
    stride = cache.index["shards"][0]["count"]
    with torch.no_grad(), runtime.mixer.scales({}):
        for scene in range(len(cache.index["shards"])):
            guarded()
            index = scene * stride + (scene + 3) % stride
            row = cache[index]
            neutral, positive = (runtime.encode(row["row"][key]) for key in ("neutral", "positive"))
            latent = row["latent"].to(runtime.device)
            fields.append({
                "index": index, "position": row["position"], "trajectory": row["trajectory"],
                "embedding": torch.equal(neutral.cpu(), row["embedding"]),
                "neutral": torch.equal(runtime.predict(latent, row["timestep"], neutral).cpu(), row["neutral"]),
                "positive": torch.equal(runtime.predict(latent, row["timestep"], positive).cpu(), row["positive"]),
            })
    passed = all(all(row[key] for key in ("embedding", "neutral", "positive")) for row in fields)
    atomic_json(ROOT / "cohorts" / method / "native-parity.json",
                {"passed": passed, "request": request, "fields": fields})
    return passed
