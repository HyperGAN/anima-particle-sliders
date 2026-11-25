"""Verify and bind the shared twelve-scene cache before admitting training."""
import json
from pathlib import Path

from control import CACHE, MANIFEST, ORIGINAL, ROOT, SOURCE, STORAGE
from lumen_studio.cache import TargetCache
from lumen_studio.contracts import atomic_json, digest, file_hash
from lumen_studio.vibe import validate_manifest
from optimization_worker import options, pin, read


def main():
    manifest = read(MANIFEST)
    validate_manifest(manifest)
    cache = TargetCache(CACHE, verify=True)
    checked = cache.index
    audit_path = MANIFEST.parents[2] / "audits" / f"{manifest['portfolio_round_sha256']}.json"
    audit = read(audit_path)
    if (len(manifest["rows"]) != 12 or len(cache) != 240 or checked["identity"]["manifest_sha256"] != manifest["sha256"]
            or digest(audit) != manifest["portfolio_round_sha256"] or len(audit["retained"]) != 12):
        raise ValueError("The expanded teacher cache does not match the verified twelve-scene cohort")
    receipt = {
        "manifest": manifest["sha256"], "manifest_file_sha256": file_hash(MANIFEST),
        "selection_audit": pin(audit_path), "cache": checked["fingerprint"],
        "cache_index_sha256": file_hash(CACHE / "index.json"), "targets": len(cache),
        "model": checked["identity"]["model"], "scenes": 12,
    }
    atomic_json(STORAGE / "targets/receipt.json", receipt)
    original = read(ORIGINAL / "search-protocol.json")
    labels = read(ROOT / "recipes.json")
    original["formulation_comparison"]["methods"] = list(labels)
    original["training_options"] = options() | {"variation": manifest["variation"]}
    original["experiment_target"] = "Twelve distinct Final Boss vibe scenes; shared RAW192 and the pinned eight-case score."
    atomic_json(ROOT / "search-protocol.json", original)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
