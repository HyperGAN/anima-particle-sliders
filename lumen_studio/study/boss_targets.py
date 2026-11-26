"""Apply the measured Final Boss field recipe to a pinned set of matched teaching images."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import torch

from ..cache import TargetCache, save_tensor_file
from ..contracts import atomic_json, digest, file_hash


def subject_matte(path, shape, device):
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255
    matte = torch.as_tensor(values, device=device)[None, None]
    return torch.nn.functional.interpolate(matte, size=shape, mode="area").unsqueeze(2)


def edited_target(neutral, positive, weight, cue_mix, explicit=None, alternate=None):
    residual = positive - neutral
    mixed = residual
    if explicit is not None:
        direction = explicit - neutral
        if alternate is not None:
            direction = (direction + (alternate - neutral)) * 0.5
        mixed = residual.lerp(direction, cue_mix)
    weighted = mixed * weight
    norm = weighted.norm()
    if not torch.isfinite(norm) or norm.item() <= 1e-10:
        raise ValueError("Degenerate weighted teacher contrast")
    return neutral + weighted * (residual.norm() / norm)


def field_context(runtime, recipe, directory, row, shape):
    selected = row["selection"]
    entry = recipe["rows"].get(selected["positive"]["output"]["sha256"])
    if (
        entry is None
        or entry["neutral_sha256"] != selected["neutral"]["output"]["sha256"]
    ):
        raise ValueError(
            "This Final Boss field recipe has no matte and explicit cue for the selected image pair"
        )
    path = directory / entry["matte"]
    if (
        Path(entry["matte"]).name != entry["matte"]
        or file_hash(path) != entry["matte_sha256"]
    ):
        raise ValueError("Final Boss foreground matte differs from the bound recipe")
    return 0.45 + 0.55 * subject_matte(path, shape, runtime.device), runtime.encode(
        entry["explicit"]
    )


def target_field(runtime, record, neutral, positive, context):
    weight, embedding = context
    cue = None
    if record["position"] >= 7:
        cue = runtime.predict(
            record["latent"].to(runtime.device), record["timestep"], embedding
        )
    return edited_target(neutral, positive, weight, 0.35, explicit=cue)


@torch.no_grad()
def prepare_boss_targets(runtime, source, destination, recipe_path):
    recipe = json.loads(recipe_path.read_text())
    source_identity = source.index["identity"]
    if (
        recipe["method"] != "late-cues-v1"
        or source_identity["model"] != runtime.identity
        or source_identity["layout"]
        != "both-trajectories/all-positions/full-velocity/shared-state-v1"
    ):
        raise ValueError("The Final Boss teacher recipe or native runtime differs")
    identity = dict(
        source.index["identity"],
        layout="final-boss/late-cues-v1",
        source_cache=source.index["fingerprint"],
        field_recipe=recipe,
    )
    fingerprint = digest(identity)
    if destination.exists():
        cache = TargetCache(destination, verify=True)
        if cache.index["identity"] != identity:
            raise ValueError(
                "Existing Final Boss targets differ from this source cache or recipe"
            )
        return cache
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp, runtime.mixer.scales({}):
        staged = Path(tmp) / "targets"
        staged.mkdir()
        shards = []
        for shard in source.index["shards"]:
            data = torch.load(
                source.directory / shard["path"], map_location="cpu", weights_only=True
            )
            row, records = data["row"], data["records"]
            context = field_context(
                runtime,
                recipe,
                recipe_path.parent,
                row,
                records[0]["latent"].shape[-2:],
            )
            entry = recipe["rows"][row["selection"]["positive"]["output"]["sha256"]]
            (staged / entry["matte"]).write_bytes(
                (recipe_path.parent / entry["matte"]).read_bytes()
            )
            for record in records:
                record["positive"] = target_field(
                    runtime,
                    record,
                    record["neutral"].to(runtime.device),
                    record["positive"].to(runtime.device),
                    context,
                ).cpu()
            data["fingerprint"] = fingerprint
            output = staged / shard["path"]
            save_tensor_file(output, data)
            shards.append(dict(shard, sha256=file_hash(output)))
        atomic_json(
            staged / "index.json",
            dict(
                identity=identity,
                input_fingerprint=fingerprint,
                shards=shards,
                fingerprint=digest(dict(input=fingerprint, shards=shards)),
            ),
        )
        staged.rename(destination)
    return TargetCache(destination, verify=True)


@torch.no_grad()
def teacher_records(runtime, neutral, positive, seed, *, resolution=512):
    """Both frozen teachers at every state of both native trajectories for one scene."""
    records, first = [], None
    initial = runtime.noise(seed, resolution, resolution)
    for trajectory in ("neutral", "positive"):
        latent = initial.clone()
        scheduler = runtime.scheduler(10)
        for index, timestep in enumerate(scheduler.timesteps):
            if trajectory == "positive" and index == 0:
                vn, vp = first
            else:
                vn = runtime.predict(latent, timestep, neutral)
                vp = runtime.predict(latent, timestep, positive)
            if first is None:
                first = vn, vp
            records.append(
                dict(
                    position=index,
                    timestep=float(timestep),
                    trajectory=trajectory,
                    latent=latent.cpu().clone(),
                    neutral=vn.cpu(),
                    positive=vp.cpu(),
                )
            )
            latent = runtime.step(
                scheduler, vn if trajectory == "neutral" else vp, timestep, latent
            )
    return records


@torch.no_grad()
def replace_boss_teachers(runtime, parent, destination, replacements):
    """Derive a late-cues cache in which chosen scenes learn from another Boss teacher.

    ``replacements`` maps a scene id to its new ``positive`` prompt, ``explicit`` cue
    prompt and the selected teacher image record ``selected``. The neutral prompt,
    seed, foreground matte, row order and field recipe stay those of the parent, so
    both caches expose every scene equally. Each unchanged scene is regenerated
    through the same code and must equal the parent tensors before anything is written.
    """
    identity = parent.index["identity"]
    recipe = json.loads(json.dumps(identity["field_recipe"]))
    scenes = [shard["row"] for shard in parent.index["shards"]]
    if (
        identity["layout"] != "final-boss/late-cues-v1"
        or identity["model"] != runtime.identity
        or not replacements
        or not set(replacements) < set(scenes)
        or len(set(scenes)) != len(scenes)
    ):
        raise ValueError(
            "Replace some, not all, scenes of a late-cues cache for this runtime"
        )
    loaded, changed = {}, {}
    for shard in parent.index["shards"]:
        data = torch.load(
            parent.directory / shard["path"], map_location="cpu", weights_only=True
        )
        loaded[shard["row"]] = data
        new = replacements.get(shard["row"])
        if new is None:
            continue
        row = data["row"]
        previous = row["selection"]["positive"]
        entry = recipe["rows"].pop(previous["output"]["sha256"])
        selected = new["selected"]
        if (
            selected["output"]["prompt"] != new["positive"]
            or selected["output"]["seed"] != shard["seed"]
            or file_hash(selected["output"]["path"]) != selected["output"]["sha256"]
            or selected["output"]["sha256"] in recipe["rows"]
        ):
            raise ValueError(f"Replacement teacher image differs for {shard['row']}")
        recipe["rows"][selected["output"]["sha256"]] = dict(entry, explicit=new["explicit"])
        data["row"] = dict(
            row,
            positive=new["positive"],
            selection=dict(row["selection"], positive=selected, replaced=previous),
        )
        changed[shard["row"]] = dict(
            previous=previous["output"]["sha256"], positive=selected["output"]["sha256"]
        )
    new_identity = dict(
        identity,
        field_recipe=recipe,
        teacher_replacement=dict(
            schema="final-boss-teacher-replacement-v1",
            parent_cache=parent.index["fingerprint"],
            rows=changed,
        ),
    )
    fingerprint = digest(new_identity)
    if destination.exists():
        cache = TargetCache(destination, verify=True)
        if cache.index["identity"] != new_identity:
            raise ValueError("Existing replacement targets differ from this selection")
        return cache
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp, runtime.mixer.scales({}):
        staged = Path(tmp) / "targets"
        staged.mkdir()
        shards = []
        for shard in parent.index["shards"]:
            data = loaded[shard["row"]]
            row = data["row"]
            entry = recipe["rows"][row["selection"]["positive"]["output"]["sha256"]]
            (staged / entry["matte"]).write_bytes(
                (parent.directory / entry["matte"]).read_bytes()
            )
            positive = runtime.encode(row["positive"])
            records = teacher_records(
                runtime, data["embedding"].to(runtime.device), positive, shard["seed"]
            )
            context = field_context(
                runtime, recipe, staged, row, records[0]["latent"].shape[-2:]
            )
            for record in records:
                record["positive"] = target_field(
                    runtime,
                    record,
                    record["neutral"].to(runtime.device),
                    record["positive"].to(runtime.device),
                    context,
                ).cpu()
            if shard["row"] in changed:
                # The neutral trajectory and its base velocity belong to the scene.
                for new, old in zip(records[:10], data["records"][:10], strict=True):
                    if not all(torch.equal(new[k], old[k]) for k in ("latent", "neutral")):
                        raise ValueError("Neutral trajectory was not reproduced")
                data["records"] = records
                data["positive_embedding"] = positive.cpu()
            else:
                for new, old in zip(records, data["records"], strict=True):
                    if new.keys() != old.keys() or not all(
                        torch.equal(new[k], old[k]) if torch.is_tensor(old[k]) else new[k] == old[k]
                        for k in old
                    ):
                        raise ValueError(
                            f"Unchanged scene {shard['row']} was not reproduced exactly"
                        )
            data["fingerprint"] = fingerprint
            output = staged / shard["path"]
            save_tensor_file(output, data)
            shards.append(dict(shard, sha256=file_hash(output)))
        atomic_json(
            staged / "index.json",
            dict(
                identity=new_identity,
                input_fingerprint=fingerprint,
                shards=shards,
                fingerprint=digest(dict(input=fingerprint, shards=shards)),
            ),
        )
        staged.rename(destination)
    return TargetCache(destination, verify=True)
