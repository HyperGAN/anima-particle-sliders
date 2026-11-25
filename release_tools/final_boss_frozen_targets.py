"""Restore the exact training targets referenced by a published Final Boss winner."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import hf_hub_download

from lumen_studio.cache import TargetCache
from lumen_studio.contracts import digest, file_hash
from lumen_studio.provenance import same_portable_model


def restore_targets(config, model, output):
    raw = config["checkpoints"][config["recipe"]]
    pin = raw["target_index"]
    prefix = Path(pin["file"]).parent

    def download(filename):
        return Path(hf_hub_download(config["checkpoint_repository"], str(filename),
                                    revision=config["checkpoint_revision"]))

    index_file = download(pin["file"])
    if file_hash(index_file) != pin["sha256"]:
        raise ValueError("The published target index has changed")
    index = json.loads(index_file.read_text())
    fingerprint = digest(dict(input=index["input_fingerprint"], shards=index["shards"]))
    if (index["fingerprint"] != raw["training_cache_sha256"] or index["fingerprint"] != fingerprint
            or digest(index["identity"]) != index["input_fingerprint"]
            or not same_portable_model(index["identity"]["model"], model)):
        raise ValueError("The target cache does not match the winning training run and native model")
    if output.exists():
        raise FileExistsError(f"Use a new target directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "targets"
        staged.mkdir()
        shutil.copyfile(index_file, staged / "index.json")
        for shard in index["shards"]:
            name = shard["path"]
            if Path(name).name != name:
                raise ValueError("The target index contains an invalid shard name")
            shutil.copyfile(download(prefix / name), staged / name)
        TargetCache(staged, verify=True)
        staged.rename(output)
    return TargetCache(output, verify=False)
