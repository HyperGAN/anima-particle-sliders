"""Export native post-training control fields and their matched teacher images."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

import torch
from safetensors.torch import load_file, save_file


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def export(source, destination):
    index = json.loads((source / "index.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "images").mkdir(exist_ok=True)
    output = {"source_cache_fingerprint": index["fingerprint"], "parent_cache_fingerprint": index["identity"]["parent"],
              "model": index["identity"]["model"], "manifest_sha256": index["identity"]["manifest_sha256"], "scenes": []}
    for shard in index["shards"]:
        path = source / shard["path"]
        if digest(path) != shard["sha256"]:
            raise ValueError("Native teacher shard changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        row = payload["row"]
        if (row["id"], payload["seed"], len(payload["records"])) != (shard["row"], shard["seed"], shard["count"]):
            raise ValueError("Teacher row identity changed")
        tensors = {key: payload[key].contiguous() for key in ("embedding", "positive_embedding")}
        records = []
        for step, record in enumerate(payload["records"]):
            keys = ("latent", "neutral", "positive")
            if set(record) != {*keys, "trajectory", "position", "timestep"}:
                raise ValueError("Unexpected native field schema")
            records.append({key: record[key] for key in ("trajectory", "position", "timestep")})
            tensors.update({f"{step}.{key}": record[key].contiguous() for key in keys})
        tensor_file = destination / path.with_suffix(".safetensors").name
        save_file(tensors, tensor_file)
        check = load_file(tensor_file)
        if check.keys() != tensors.keys() or any(check[k].dtype != v.dtype or not torch.equal(check[k], v)
                                                 for k, v in tensors.items()):
            raise ValueError("The portable fields differ from their training tensors")
        scene = {"id": row["id"], "seed": payload["seed"], "records": records, "file": tensor_file.name,
                 "sha256": digest(tensor_file), "source_shard_sha256": shard["sha256"]}
        for key in ("neutral", "positive"):
            image = row["selection"][key]["output"]
            name = f"images/{image['sha256']}.png"
            if digest(Path(image["path"])) != image["sha256"]:
                raise ValueError("Teacher image differs from the frozen selection")
            copyfile(image["path"], destination / name)
            scene[key] = {"prompt": row[key], "image": name, "sha256": image["sha256"]}
        output["scenes"].append(scene)
    (destination / "index.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
