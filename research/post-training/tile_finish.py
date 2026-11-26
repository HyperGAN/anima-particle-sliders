"""Test whether the judge catches broken drawing better on enlarged crops than on the whole image."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
import contextlib
import io
import sys

from PIL import Image
from particle_sliders.alpha_tune_jev import _Image
from particle_sliders.candidate_failure import FileRecord

from lumen_studio.contracts import atomic_json

from campaign import ROOT

with contextlib.redirect_stdout(io.StringIO()):
    from working_metric import cases, judge, ranking, read, readings, root

tiles = root / "tiles"
tiles.mkdir(exist_ok=True)
STATE = "This is an enlarged crop from an anime illustration."
QUESTIONS = {
    "broken_drawing": {
        "type": "score",
        "instructions": (
            "How cleanly is this crop drawn? A finished drawing has continuous outlines, smooth deliberate "
            "shading and shapes that read as real things such as fabric, armor, skin, hair, walls or sky. A "
            "broken drawing shatters into jagged shards or facets with no meaning, has stray streaks of "
            "color, or smears forms together. Plain flat areas, and deliberate sharp designs such as spikes "
            "on armor, are fine when their outlines are clean."
        ),
        "criteria": [
            "Cleanly drawn throughout",
            "Mostly clean with a small jagged or smeared spot",
            "Noticeable shards, streaks or smears in part of the crop",
            "Shards, streaks or smears across most of the crop",
            "Fragments only; nothing is drawn properly",
        ],
    }
}


def crops(path, sha):
    with Image.open(path) as source:
        picture = source.convert("RGB")
    half = picture.width // 2
    out = []
    for index, (x, y) in enumerate(((0, 0), (half, 0), (0, half), (half, half))):
        target = tiles / f"{sha}-{index}.png"
        if not target.exists():
            picture.crop((x, y, x + half, y + half)).resize((512, 512), Image.Resampling.LANCZOS).save(target)
        out.append(target)
    return out


def read_tile(path):
    answer = judge._decide((_Image.read(FileRecord.from_path(path)),), STATE, QUESTIONS)
    return None if answer is None else answer["broken_drawing"]


whole_image = read(root / "readings-with-finish.json")
wanted = sys.argv[1:] or ["cleaner-36", "parent-0", "control-low-g-48", "control-48", "cleanest-seed2-12"]
sets = {"OFF": [(cases[c]["off"]["path"], cases[c]["off"]["sha256"]) for c in ranking]}
comparison = read(ROOT / "damage-definition/comparison.json")
sets["current teachers"] = [(r["on"]["path"], r["on"]["sha256"]) for r in comparison["current teachers"]]
for key in wanted:
    sets[key] = [(str(ROOT / "gallery/images" / (v["on"] + ".png")), v["on"]) for v in readings[key].values()]
with ThreadPoolExecutor(8) as pool:
    jobs = {label: [[pool.submit(read_tile, t) for t in crops(*item)] for item in items]
            for label, items in sets.items()}
    result = {label: [[j.result() for j in image] for image in rows] for label, rows in jobs.items()}
atomic_json(root / "tile-finish.json", {label: dict(zip([s for _, s in sets[label]], rows))
                                        for label, rows in result.items()})
print(f'{"set":<22}{"tile mean":>10}{"worst tile":>12}{"images ≥50 worst":>18}{"whole-image":>13}')
for label, rows in result.items():
    worst = [max(v for v in image if v is not None) for image in rows]
    whole = (f'{100 * (1 - fmean(v["finish"] for v in whole_image[label].values())):.1f}' if label in readings else "—")
    print(f'{label:<22}{100 * fmean(v for image in rows for v in image if v is not None):>10.1f}'
          f'{100 * fmean(worst):>12.1f}{sum(w >= 0.5 for w in worst):>12}/{len(rows)}{whole:>13}')
print("cleaner-36 worst tile per image:", [round(100 * max(i)) for i in result["cleaner-36"]])
print("OFF worst tile per image:", [round(100 * max(i)) for i in result["OFF"]])
