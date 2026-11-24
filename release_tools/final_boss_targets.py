"""Capture the fixed Anima Final Boss teacher targets for the RAW192 study."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from lumen_studio.cache import TargetCache, save_tensor_file
from lumen_studio.contracts import atomic_json, digest, file_hash


STUDY = Path(__file__).resolve().parents[1] / "docs/final-boss-raw192"
FIXTURE = STUDY / "training-pairs.json"
VARIATION = "character-final-boss"
REGIONAL_RADII = {"regional-cue-matte": 0, "regional-cue-dilated2": 2}
RECIPES = {"prompt-only": (1.0, 0.0), "soft-floor45": (0.45, 0.0),
           "strong-boss-45-mix35": (0.45, 0.35), "strong-cues-early3": (0.45, 0.35),
           "strong-cues-late3": (0.45, 0.35), "diverse-mean35": (0.45, 0.35),
           **dict.fromkeys(REGIONAL_RADII, (0.45, 0.35))}
RESOLUTION = 512
STEPS = 10
CUE_POSITIONS = {"strong-cues-early3": (0, 1, 2), "strong-cues-late3": (7, 8, 9)}


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


def regional_target(neutral, baseline, stronger, matte, radius):
    """Rotate the edit in the allowed region, preserving both comparison norms."""
    if radius:
        matte = F.max_pool2d(matte.squeeze(2), 2 * radius + 1, stride=1, padding=radius).unsqueeze(2)
    active = matte > 0.01
    base = baseline.double() - neutral.double()
    delta = stronger.double() - baseline.double()
    local_base = torch.where(active, base, 0.)
    local_delta = torch.where(active, matte.double() * delta, 0.)
    magnitude = local_base.square().sum()
    desired_distance_sq = delta.square().sum()
    if not 0 < desired_distance_sq < 4 * magnitude:
        raise ValueError("The strong-cue edit distance does not fit the local baseline norm")
    tangent = local_delta - local_base * ((local_delta * local_base).sum() / magnitude)
    tangent_norm = tangent.norm()
    if not tangent_norm > 0:
        raise ValueError("The masked stronger cue has no tangent to the existing edit")
    cosine_loss = desired_distance_sq / (2 * magnitude)
    sine = (cosine_loss * (2 - cosine_loss)).sqrt()
    shift = tangent * (magnitude.sqrt() * sine / tangent_norm) - local_base * cosine_loss
    target = torch.where(active, (baseline.double() + shift).to(baseline.dtype), baseline)
    observed_norm_ratio = (target.double() - neutral.double()).norm() / base.norm()
    cue_distance_ratio = (target.double() - baseline.double()).norm() / desired_distance_sq.sqrt()
    if (not torch.isfinite(target).all() or abs(observed_norm_ratio - 1.) > 1e-6
            or abs(cue_distance_ratio - 1.) > 1e-6):
        raise ValueError("The regional cue failed its fixed-dose comparison")
    return target, {
        "full_edit_norm_ratio": observed_norm_ratio.item(),
        "stronger_cue_distance_ratio": cue_distance_ratio.item(),
        "cue_direction_cosine": ((shift * delta).sum() / desired_distance_sq).item(),
        "editable_area": active.double().mean().item(),
    }


def row_metadata(row):
    return dict(id=row["id"], variation=VARIATION, split="train",
                neutral=row["neutral"], positive=row["positive"], seeds=[row["seed"]])


def teacher_rows(fixture, recipe):
    rows = json.loads(fixture.read_text())["pairs"]
    if recipe != "diverse-mean35":
        return rows
    alternates = json.loads((fixture.parent / "ablations/two-cue-prompts.json").read_text())
    if set(alternates) != {row["id"] for row in rows}:
        raise ValueError("Two-cue prompts must cover the training pairs exactly")
    return [row | {"alternate_positive": alternates[row["id"]]} for row in rows]


def teacher_cues(runtime, row, recipe):
    if not RECIPES[recipe][1]:
        return ()
    keys = ("explicit_positive", "alternate_positive") if recipe == "diverse-mean35" else ("explicit_positive",)
    return tuple(runtime.encode(row[key]) for key in keys)


def final_target(runtime, latent, timestep, neutral, positive, weight, cue_mix, cue_embeddings, *, regional=None):
    # Preserve the rounding of the original full-frame capture.
    positive = neutral + (positive - neutral)
    if weight is None:
        return positive
    cues = (runtime.predict(latent, timestep, embedding) for embedding in cue_embeddings)
    target = edited_target(neutral, positive, weight, cue_mix, *cues)
    if regional is None:
        return target
    matte, radius = regional
    baseline = edited_target(neutral, positive, weight, 0.)
    return regional_target(neutral, baseline, target, matte, radius)[0]


def capture_row(runtime, row, matte_path, recipe):
    floor, cue_mix = RECIPES[recipe]
    neutral, positive = (runtime.encode(row[key]) for key in ("neutral", "positive"))
    cues = teacher_cues(runtime, row, recipe)
    cue_positions = CUE_POSITIONS.get(recipe, range(STEPS))
    n = p = runtime.noise(row["seed"], RESOLUTION, RESOLUTION)
    matte = subject_matte(matte_path, n.shape[-2:], runtime.device) if floor < 1 else None
    weight = floor + (1 - floor) * matte if matte is not None else None
    regional = (matte, REGIONAL_RADII[recipe]) if recipe in REGIONAL_RADII else None
    n_scheduler, p_scheduler = runtime.scheduler(STEPS), runtime.scheduler(STEPS)
    records = {"neutral": [], "positive": []}
    for position, timestep in enumerate(n_scheduler.timesteps):
        vn = runtime.predict(n, timestep, neutral)
        vp = runtime.predict(n, timestep, positive)
        vnp, vpp = ((vn, vp) if position == 0 else
                    (runtime.predict(p, timestep, neutral), runtime.predict(p, timestep, positive)))
        n_next = runtime.step(n_scheduler, vn, timestep, n)
        p_next = runtime.step(p_scheduler, vnp + (vpp - vnp), timestep, p)
        for track, latent, base, positive_field in (("neutral", n, vn, vp), ("positive", p, vnp, vpp)):
            active = cues if position in cue_positions else ()
            target = final_target(runtime, latent, timestep, base, positive_field, weight, cue_mix,
                                  active, regional=regional)
            records[track].append(dict(position=position, timestep=float(timestep), trajectory=track,
                                       latent=latent.cpu().clone(), neutral=base.cpu().clone(),
                                       positive=target.cpu().clone()))
        n, p = n_next, p_next
    return dict(row=row_metadata(row), seed=row["seed"], embedding=neutral.cpu(),
                positive_embedding=positive.cpu(), records=records["neutral"] + records["positive"])


@torch.no_grad()
def capture(runtime, recipe, output, fixture=FIXTURE):
    """Capture both frozen paths and replace only the positive response fields."""
    fixture = Path(fixture)
    output = Path(output)
    rows = teacher_rows(fixture, recipe)
    for row in rows:
        if file_hash(fixture.parent / row["matte"]["file"]) != row["matte"]["sha256"]:
            raise ValueError(f"Subject matte changed for {row['id']}")
    output.mkdir(parents=True, exist_ok=False)
    identity = dict(model=runtime.identity, layout=f"final-boss-raw192/{recipe}/v1",
                    manifest_sha256=digest(rows), split="train", variation=VARIATION,
                    resolution=RESOLUTION, steps=STEPS, cfg=1, recipe=recipe)
    fingerprint = digest(identity)
    shards = []
    with runtime.mixer.scales({}):
        for row in rows:
            data = capture_row(runtime, row, fixture.parent / row["matte"]["file"], recipe)
            path = output / f"{row['id']}-{row['seed']}.pt"
            save_tensor_file(path, data | {"fingerprint": fingerprint})
            shards.append(dict(path=path.name, sha256=file_hash(path), count=len(data["records"]),
                               row=row["id"], seed=row["seed"]))
    atomic_json(output / "index.json", dict(identity=identity, input_fingerprint=fingerprint,
                fingerprint=digest(dict(input=fingerprint, shards=shards)), shards=shards))
    return TargetCache(output, verify=True)


@torch.no_grad()
def verify_native(runtime, cache, request, fixture=FIXTURE):
    """Recheck four captured fields through the frozen native teacher."""
    identity = cache.index["identity"]
    expected = dict(model_identity=runtime.identity, cache_fingerprint=cache.index["fingerprint"],
                    response_shape=list(cache[0]["positive"].shape))
    if request != expected or identity["model"] != runtime.identity:
        return False
    fixture = Path(fixture)
    recipe = identity["recipe"]
    rows = {row["id"]: row for row in teacher_rows(fixture, recipe)}
    floor, cue_mix = RECIPES[recipe]
    cue_positions = CUE_POSITIONS.get(recipe, range(STEPS))
    indices = (1, 19, 42, 117) if recipe in CUE_POSITIONS else (3, 19, 46, 117)
    with runtime.mixer.scales({}):
        for index in indices:
            record = cache[index]
            row = rows[record["row"]["id"]]
            latent, timestep = record["latent"].to(runtime.device), record["timestep"]
            neutral = runtime.predict(latent, timestep, runtime.encode(row["neutral"]))
            positive = runtime.predict(latent, timestep, runtime.encode(row["positive"]))
            matte_path = fixture.parent / row["matte"]["file"]
            if file_hash(matte_path) != row["matte"]["sha256"]:
                return False
            matte = subject_matte(matte_path, latent.shape[-2:], runtime.device) if floor < 1 else None
            weight = floor + (1 - floor) * matte if matte is not None else None
            regional = (matte, REGIONAL_RADII[recipe]) if recipe in REGIONAL_RADII else None
            cues = teacher_cues(runtime, row, recipe) if record["position"] in cue_positions else ()
            target = final_target(runtime, latent, timestep, neutral, positive, weight, cue_mix,
                                  cues, regional=regional)
            if not (torch.equal(neutral.cpu(), record["neutral"]) and torch.equal(target.cpu(), record["positive"])):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = json.loads((STUDY / "default.json").read_text())["recipe"]
    parser.add_argument("recipe", nargs="?", choices=RECIPES, default=default,
                        help=f"Training recipe (default: {default})")
    parser.add_argument("--model", type=Path, required=True, help="Converted Anima Turbo v1.1 directory")
    parser.add_argument("--out", type=Path, required=True, help="New target-cache directory")
    args = parser.parse_args()

    from lumen_studio.backends.anima import TurboRuntime
    from lumen_studio.core_source import verify_core

    verify_core()
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = TurboRuntime(args.model, device="cuda:0", verify=True)
    try:
        cache = capture(runtime, args.recipe, args.out)
        request = dict(model_identity=runtime.identity, cache_fingerprint=cache.index["fingerprint"],
                       response_shape=list(cache[0]["positive"].shape))
        if not verify_native(runtime, cache, request):
            raise RuntimeError("Native Final Boss target parity failed")
        print(f"Captured {len(cache)} native fields to {args.out}")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
