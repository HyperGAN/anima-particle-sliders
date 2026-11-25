"""Interpolate frozen matched teacher edits before applying a soft subject matte."""

from functools import lru_cache
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / "anima-final-boss-signal-raw192-20261123"
MASK_CACHE = ROOT.parent / "anima-character-final-boss-20261122/target-cache/index.json"
METHODS = tuple(path.stem for path in sorted((ROOT / "plans").glob("*.json")))


@lru_cache
def plan(method):
    if method not in METHODS:
        raise ValueError(f"Unknown target recipe {method}")
    recipe = json.loads((ROOT / "plans" / f"{method}.json").read_text())
    if (recipe["method"] != method or not 0 < recipe["floor"] <= 1 or not 0 <= recipe["blend"] <= 1
            or (recipe["donor"] is None) != (recipe["blend"] == 0)
            or recipe["donor"] not in (None, "protected-contexts", "nuisance-projection", "prompt-consensus")):
        raise ValueError(f"Invalid target recipe {recipe}")
    return recipe


@lru_cache
def mattes():
    data = json.loads(MASK_CACHE.read_text())
    return {item["row"]["id"]: item["mask"] for item in data["identity"]["teachers"]}


def sources(rows):
    return {name: {"neutral": row["neutral"], "positive": row["positive"], "mask": mattes()[name]}
            for name, row in rows.items()}


def donor_cache(method):
    from lumen_studio.cache import TargetCache

    donor = plan(method)["donor"]
    if donor is None:
        return None
    folder = ARCHIVE / "cohorts" / donor
    parity = json.loads((folder / "native-parity.json").read_text())
    if not parity["passed"]:
        raise ValueError(f"Donor {donor} lacks native target parity")
    return TargetCache(folder / "target-cache", verify=True)


def encode_context(runtime, row, method, shape):
    from fit import mask_for
    from native import pin

    mask = mattes()[row["id"]]
    if pin(Path(mask["path"])) != mask:
        raise ValueError("The source subject matte changed")
    recipe = plan(method)
    matte = mask_for(Path(mask["path"]), shape, runtime.device)
    floor = recipe["floor"]
    return {"neutral": runtime.encode(row["neutral"]), "positive": runtime.encode(row["positive"]),
            "weight": floor + (1 - floor) * matte, "blend": recipe["blend"]}


def apply_target(context, neutral, positive, donor_target):
    residual = positive - neutral
    mixed = residual if donor_target is None else residual.lerp(donor_target - neutral, context["blend"])
    edit = mixed * context["weight"]
    norm = edit.norm()
    return neutral + edit * (residual.norm() / norm.clamp_min(1e-10)), {
        "blend": context["blend"], "weighted_norm_before_match": norm.item(),
    }
