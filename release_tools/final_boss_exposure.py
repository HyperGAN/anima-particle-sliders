"""Reweight frozen Anima teacher rows for the stock balanced sampler."""

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from lumen_studio.cache import TargetCache, save_tensor_file
from lumen_studio.contracts import atomic_json, digest, file_hash


POSITIONS = {
    "exposure-opening4": (0, 1, 2, 3),
    "exposure-middle4": (3, 4, 5, 6),
}
COORDINATES = [(trajectory, position) for trajectory in ("neutral", "positive") for position in range(10)]


def freeze(source, method, output):
    source = TargetCache(source)
    if len(source) != 120 or [shard["count"] for shard in source.index["shards"]] != [20] * 6:
        raise ValueError("Expected six original Anima contexts, each with two ten-position trajectories")
    positions = POSITIONS[method]
    parent = source.index["identity"]
    recipe = {
        "source_cache": source.index["fingerprint"],
        "source_manifest": parent["manifest_sha256"],
        "repeated_positions": list(positions),
        "row_order": "twenty originals then eight repeats per source shard",
    }
    identity = parent | {
        "layout": "frozen-teacher-phase-exposure/v1",
        "phase_exposure": recipe,
        "phase_exposure_source_sha256": file_hash(Path(__file__).resolve()),
        "manifest_sha256": digest(recipe),
    }
    input_id = digest(identity)
    output = Path(output)
    existing = TargetCache(output) if output.exists() else None
    if existing is not None and (existing.index["identity"] != identity
                                 or existing.index["input_fingerprint"] != input_id):
        raise ValueError("A different exposure cache already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".exposure-") as temporary:
        staging = Path(temporary) / "cache"
        staging.mkdir()
        shard_entries = []
        for shard in source.index["shards"]:
            name = Path(shard["path"])
            if name.name != shard["path"]:
                raise ValueError("Expected a plain source shard filename")
            original = torch.load(source.directory / name, map_location="cpu", weights_only=True)
            records = original["records"]
            if (original["fingerprint"] != source.index["input_fingerprint"]
                    or (original["row"]["id"], original["seed"]) != (shard["row"], shard["seed"])
                    or [(row["trajectory"], row["position"]) for row in records] != COORDINATES):
                raise ValueError(f"Unexpected source coordinates in {name}")
            repeats = [row for row in records if row["position"] in positions]
            derived = original | {"fingerprint": input_id, "records": records + repeats}
            if existing is None:
                save_tensor_file(staging / name, derived)
                shard_path = staging / name
            else:
                shard_path = output / name
                found = torch.load(shard_path, map_location="cpu", weights_only=True)
                verify_shard(found, derived, name)
            shard_entries.append(shard | {"count": 28, "sha256": file_hash(shard_path)})
        index = {"identity": identity, "input_fingerprint": input_id, "shards": shard_entries,
                 "fingerprint": digest({"input": input_id, "shards": shard_entries})}
        if existing is None:
            atomic_json(staging / "index.json", index)
            staging.rename(output)
        elif existing.index != index:
            raise ValueError("The cached exposure manifest differs from the deterministic recipe")
    return TargetCache(output)


def verify_shard(actual, expected, name):
    def equal(left, right):
        if isinstance(right, torch.Tensor):
            return isinstance(left, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
        if isinstance(right, dict):
            return left.keys() == right.keys() and all(equal(left[key], value) for key, value in right.items())
        if isinstance(right, list):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        return left == right

    if not equal(actual, expected):
        raise ValueError(f"Stored exposure fields differ from the original coordinates or tensors in {name}")


def counts_by_position(cache, exposures):
    result = [0] * 10
    for index, count in enumerate(exposures.tolist()):
        result[cache[index]["position"]] += count
    return result


def sampler_receipt(cache, seed, steps, batch_size):
    """Exercise the actual stock draw routine without running model math."""
    from particle_sliders.engine import ConditionalParticleGAN

    sampler = SimpleNamespace(
        sampling="balanced", rows=range(len(cache)), batch_size=batch_size,
        streams={phase + "_rows": torch.Generator().manual_seed(seed + offset)
                 for phase, offset in (("d", 10), ("g", 30))},
        row_orders={phase: torch.empty(0, dtype=torch.int64) for phase in ("d", "g")},
        row_cursors={"d": 0, "g": 0},
    )
    result = {}
    for phase in ("d", "g"):
        ids = torch.cat([ConditionalParticleGAN._sample_ids(sampler, phase) for _ in range(steps)])
        counts = ids.bincount(minlength=len(cache))
        result[phase] = {
            "presentations": ids.numel(),
            "by_position": counts_by_position(cache, counts),
            "by_row": counts.tolist(),
            "rows_seen": int(counts.count_nonzero()),
            "row_sequence_sha256": digest(ids.tolist()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("method", choices=POSITIONS)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cache = freeze(args.source, args.method, args.output)
    # Reopen the committed files and verify every original and repeated field.
    freeze(args.source, args.method, args.output)
    counts = Counter(cache[index]["position"] for index in range(len(cache)))
    expected = {position: 24 if position in POSITIONS[args.method] else 12 for position in range(10)}
    if counts != expected:
        raise ValueError(f"Unexpected row counts by position: {counts}")
    sampler = sampler_receipt(cache, seed=1, steps=192, batch_size=2)
    if any(receipt["presentations"] != 384 or receipt["rows_seen"] != 168 for receipt in sampler.values()):
        raise ValueError("The stock sampler did not deliver the registered training budget and coverage")
    atomic_json(args.receipt, {
        "method": args.method,
        "cache_fingerprint": cache.index["fingerprint"],
        "recipe": cache.index["identity"]["phase_exposure"],
        "rows_by_position": dict(sorted(counts.items())),
        "bitwise_verified_teacher_rows": len(cache),
        "unique_source_rows": 120,
        "seed": 1, "steps": 192, "batch_size": 2,
        "sampler": sampler,
    })


if __name__ == "__main__":
    main()
