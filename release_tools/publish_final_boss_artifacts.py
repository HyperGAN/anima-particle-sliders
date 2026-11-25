"""Publish verified RAW checkpoints and the leaderboard winner's frozen target cache."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from release_tools.publish_final_boss_vibe12 import read


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(campaign, folder, repository, target_cache):
    panel = read(campaign / "storage/distinct-alpha-standings.json")
    winner = panel["rows"][0]["method"]
    target_cache = target_cache or campaign / "storage/targets" / winner
    api = HfApi()
    info = api.model_info(repository, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    operations = []
    catalog_path = folder / "checkpoints.json"
    previous = read(catalog_path) if catalog_path.exists() else {}

    def verify_and_stage(path, name, expected):
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Local artifact changed: {path}")
        if name in remote:
            record = remote[name]
            remote_sha = (record.lfs.sha256 if record.lfs else
                          sha256(Path(hf_hub_download(repository, name, revision=info.sha))))
            if remote_sha != actual:
                raise ValueError(f"Published artifact already exists with different bytes: {name}")
        else:
            operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)))
        return dict(file=name, sha256=actual, bytes=path.stat().st_size)

    checkpoints = {}
    for row in panel["rows"]:
        name = row["method"]
        result = read(campaign / "cohorts" / name / "result.json")
        if result["step"] != 192 or result["ema_decay"] is not None:
            raise ValueError(f"Expected a RAW192 result for {name}")
        raw = result["raw"]
        artifact = verify_and_stage(Path(raw["path"]), f"vibe12/{name}/RAW000192.safetensors", raw["sha256"])
        if artifact["bytes"] != raw["bytes"]:
            raise ValueError(f"Checkpoint size differs for {name}")
        checkpoints[name] = dict(**artifact, training_cache_sha256=result["cache"])
        old = previous.get("checkpoints", {}).get(name, {})
        if (previous.get("repository") == repository and old.get("sha256") == raw["sha256"]
                and "target_index" in old and old["target_index"]["file"] in remote):
            checkpoints[name]["target_index"] = old["target_index"]

    index_file = target_cache / "index.json"
    index = read(index_file)
    if index["fingerprint"] != checkpoints[winner]["training_cache_sha256"]:
        raise ValueError("Supply the exact target cache used to train the winning adapter")
    prefix = f"vibe12/{winner}/targets"
    for shard in index["shards"]:
        name = shard["path"]
        if Path(name).name != name:
            raise ValueError(f"Invalid target shard path {name}")
        verify_and_stage(target_cache / name, f"{prefix}/{name}", shard["sha256"])
    checkpoints[winner]["target_index"] = verify_and_stage(index_file, f"{prefix}/index.json", sha256(index_file))
    revision = info.sha
    if operations:
        revision = api.create_commit(repo_id=repository, parent_commit=info.sha, operations=operations,
                                     commit_message=f"Publish measured Final Boss adapters and {winner} targets").oid
    pinned = {s.rfilename: s for s in api.model_info(repository, revision=revision, files_metadata=True).siblings}
    if any(pinned[r["file"]].lfs.sha256 != r["sha256"] for r in checkpoints.values()):
        raise ValueError("Published RAW artifacts failed verification")
    catalog_path.write_text(json.dumps(dict(repository=repository, checkpoints=checkpoints, revision=revision),
                                       indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--repository", default="ntc-ai/anima-final-boss-research")
    parser.add_argument("--target-cache", type=Path, help="Override the winning cache location on the research host")
    args = parser.parse_args()
    publish(args.campaign, args.folder, args.repository, args.target_cache)
