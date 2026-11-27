"""Rank measured treatments across rounds, with one training pair per scene."""
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

LIMIT = 16
MIN_VIBE_SCORE = 0.35


def has_vibe_score(option):
    score = option.get("boss")
    return isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1


def subject_matte(root, neutral, python, script):
    # A mask belongs to a neutral image. Different seeds of a scene have different masks.
    folder = root / "mattes"
    folder.mkdir(exist_ok=True)
    destination = folder / (neutral["sha256"] + ".png")
    if not destination.exists():
        with TemporaryDirectory(dir=folder) as temporary:
            output = Path(temporary) / "mask.png"
            cut = subprocess.run([python, str(script), neutral["path"], str(output)],
                                 capture_output=True, text=True)
            if cut.returncode:
                if "Unusable foreground region" in cut.stderr or "No character region" in cut.stderr:
                    return None
                raise RuntimeError(cut.stderr)
            output.replace(destination)
    return str(destination)


def select_teachers(rounds, scenes, matte_for, limit=LIMIT):
    """Resolve ties by recorded order. All treatments in a row share one neutral mask."""
    ranked = []
    for row in rounds:
        row["chosen"] = None
        eligible = [o for o in row["options"] if has_vibe_score(o) and o["boss"] >= MIN_VIBE_SCORE]
        if eligible:
            ranked.append((row, max(eligible, key=lambda o: o["boss"])))

    kept = {}
    for row, option in sorted(ranked, key=lambda pair: pair[1]["boss"], reverse=True):
        name = row["scene"]
        if name in kept or row.get("mask_usable") is False:
            continue
        neutral = row["neutral"]
        matte = matte_for(neutral)
        row["mask_usable"] = matte is not None
        if matte is None:
            continue
        row["chosen"] = option["positive"]
        kept[name] = dict(
            neutral=scenes[name]["neutral"], positive=option["positive"], explicit=option["explicit"],
            seed=row["seed"], framing=row["framing"], boss=option["boss"], case=option.get("case"), matte=matte,
            neutral_image={k: neutral[k] for k in ("path", "sha256")},
            positive_image={k: option["image"][k] for k in ("path", "sha256")})
        if len(kept) == limit:
            break
    return kept
