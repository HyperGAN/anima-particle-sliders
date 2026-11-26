"""Read drawing quality on native-resolution tiles, so no model sees a shrunken image."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from statistics import fmean
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import base64
import hashlib
import json
import os
import sys

from PIL import Image

from campaign import PARENT_STUDY, ROOT

URL = "https://api.nano-gpt.com/api/v1/decisions"
root = ROOT / "working-metric"
store = root / "native-tiles"
store.mkdir(exist_ok=True)
STATE = "This is an unscaled crop from an anime illustration."
QUESTIONS = {
    "broken_drawing": {
        "type": "score",
        "instructions": (
            "How cleanly is this crop drawn? A finished drawing has continuous outlines, smooth deliberate "
            "shading and shapes that read as real things such as fabric, armor, skin, hair, walls or sky. A "
            "broken drawing shatters into jagged shards or facets with no meaning, has stray streaks of "
            "color, a smeared, missing or mask-like face, or smears forms together. Plain flat areas, and "
            "deliberate sharp designs such as spikes on armor, are fine when their outlines are clean."
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


def read(path):
    return json.loads(Path(path).read_bytes())


def tiles(path, grid, size=None):
    """Native pixels only: an overlapping grid of crops, never resized."""
    with Image.open(path) as source:
        picture = source.convert("RGB")
    size = size or (picture.width // 2 if grid == 2 else picture.width * 3 // 8)
    step = (picture.width - size) // (grid - 1)
    out = []
    for row in range(grid):
        for column in range(grid):
            buffer = BytesIO()
            picture.crop((column * step, row * step, column * step + size, row * step + size)).save(
                buffer, "PNG", optimize=True)
            out.append(buffer.getvalue())
    return out


def ask(model, raw):
    key = hashlib.sha256(json.dumps([model, STATE, QUESTIONS, hashlib.sha256(raw).hexdigest()],
                                    sort_keys=True).encode()).hexdigest()
    receipt = store / f"{key}.json"
    if receipt.exists():
        body = read(receipt)
    else:
        image = {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}}
        payload = dict(model=model, state=[STATE, "image_1:", image], questions=QUESTIONS)
        request = Request(URL, data=json.dumps(payload).encode(), method="POST",
                          headers={"Content-Type": "application/json"})
        request.add_unredirected_header("Authorization", "Bearer " + os.environ["NANOGPT_API_KEY"])
        try:
            with urlopen(request, timeout=120) as response:
                body = json.load(response)
        except HTTPError as error:
            body = {"http_status": error.code, "detail": error.read().decode(errors="replace")[:300]}
        receipt.write_text(json.dumps(body))
    try:
        return body["answers"]["broken_drawing"]["score"] / 4
    except (KeyError, TypeError):
        return None


if __name__ == "__main__":
    grid, size = int(sys.argv[1]), int(sys.argv[2])
    models = sys.argv[3:]
    panels = read(PARENT_STUDY / "cohort-retest/panels.json")
    cases = {c["id"]: c for c in panels["cases"]}
    comparison = read(ROOT / "damage-definition/comparison.json")
    ranked = read(root / "readings-with-finish.json")
    sets = {
        "OFF (clean)": [cases[c]["off"]["path"] for c in panels["panels"]["ranking"]],
        "teachers (clean)": [r["on"]["path"] for r in comparison["current teachers"]],
    }
    for key in ("control-low-g-48", "parent-0", "cleaner-36", "cleanest-12"):
        sets[key] = [str(ROOT / "gallery/images" / (v["on"] + ".png")) for v in ranked[key].values()]
    cut = {label: [tiles(p, grid, size) for p in paths] for label, paths in sets.items()}
    sizes = [len(t) for rows in cut.values() for image in rows for t in image]
    print(f"{grid}x{grid} native {size}px tiles, PNG bytes {min(sizes)} to {max(sizes)}")
    with ThreadPoolExecutor(8) as pool:
        jobs = {m: {label: [[pool.submit(ask, m, t) for t in image] for image in rows]
                    for label, rows in cut.items()} for m in models}
        table = {m: {label: [[j.result() for j in image] for image in rows] for label, rows in sets_.items()}
                 for m, sets_ in jobs.items()}
    (root / f"native-tiles-{grid}-{size}.json").write_text(json.dumps(table))
    print(f'{"model · worst tile":<34}' + "".join(f"{label[:16]:>18}" for label in sets)
          + f'{"unanswered":>12}{"busted above clean":>20}')
    for m, rows in table.items():
        worst = {label: [max((v for v in image if v is not None), default=None) for image in images]
                 for label, images in rows.items()}
        missing = sum(v is None for images in rows.values() for image in images for v in image)
        cells = "".join(f"{100 * fmean(v for v in values if v is not None):>18.1f}" if any(v is not None for v in values)
                        else f'{"none":>18}' for values in worst.values())
        clean = [v for label in ("OFF (clean)", "teachers (clean)") for v in worst[label] if v is not None]
        busted = [v for label in ("cleaner-36", "cleanest-12") for v in worst[label] if v is not None]
        flag = f"{sum(v > max(clean) for v in busted)}/{len(busted)}" if clean and busted else "—"
        print(f"{m:<34}{cells}{missing:>12}{flag:>20}")
        mean = "".join(f"{100 * fmean(v for image in images for v in image if v is not None):>18.1f}"
                       if any(v is not None for image in images for v in image) else f'{"none":>18}'
                       for images in rows.values())
        print(f'{"  mean tile":<34}{mean}')
        print("  flagged row, worst tile per image:", [None if v is None else round(100 * v) for v in worst["cleaner-36"]])
