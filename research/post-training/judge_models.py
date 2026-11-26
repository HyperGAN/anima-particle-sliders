"""Compare decision models on one whole-image drawing-quality question, using saved images."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import base64
import hashlib
import json
import os
import sys

from campaign import PARENT_STUDY, ROOT

URL = "https://api.nano-gpt.com/api/v1/decisions"
root = ROOT / "working-metric"
store = root / "model-comparison"
store.mkdir(exist_ok=True)
STATE = "Judge how well this anime illustration is drawn, not what it depicts."
QUESTIONS = {
    "broken_drawing": {
        "type": "score",
        "instructions": (
            "How cleanly is this illustration drawn? A finished drawing has continuous outlines, smooth "
            "deliberate shading, and a face and body that hold together. A broken drawing shatters into "
            "jagged shards or facets with no meaning, has stray streaks of color, a smeared, missing or "
            "mask-like face, or forms that dissolve into abstract fragments. Deliberate designs such as "
            "spiked armor, crowns and capes are fine when they are cleanly drawn."
        ),
        "criteria": [
            "Cleanly drawn throughout",
            "Mostly clean with a small jagged or smeared spot",
            "Noticeable shards, streaks or a face that does not hold together",
            "Shards, streaks or smears across most of the picture",
            "Fragments only; nothing is drawn properly",
        ],
    }
}


def read(path):
    return json.loads(Path(path).read_bytes())


def ask(model, path):
    raw, media = Path(path).read_bytes(), "image/png"
    if model.startswith("inception/"):
        # This route takes PNG only and refuses the saved 512-pixel files as too large.
        from io import BytesIO

        from PIL import Image

        buffer = BytesIO()
        with Image.open(path) as picture:
            picture.convert("RGB").resize((288, 288), Image.Resampling.LANCZOS).save(buffer, "PNG", optimize=True)
        raw = buffer.getvalue()
    if model.startswith("cloudflare/"):
        # These routes accept at most 384 KiB of decoded image; the saved PNGs are larger.
        from io import BytesIO

        from PIL import Image

        buffer = BytesIO()
        with Image.open(path) as picture:
            picture.convert("RGB").save(buffer, "JPEG", quality=95)
        raw, media = buffer.getvalue(), "image/jpeg"
    key = hashlib.sha256(json.dumps([model, STATE, QUESTIONS, hashlib.sha256(raw).hexdigest()],
                                    sort_keys=True).encode()).hexdigest()
    receipt = store / f"{key}.json"
    if receipt.exists():
        body = read(receipt)
    else:
        image = {"type": "image_url",
                 "image_url": {"url": f"data:{media};base64," + base64.b64encode(raw).decode()}}
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


panels = read(PARENT_STUDY / "cohort-retest/panels.json")
cases = {c["id"]: c for c in panels["cases"]}
comparison = read(ROOT / "damage-definition/comparison.json")
ranked = read(root / "readings-with-finish.json")
sets = {
    "OFF (clean)": [cases[c]["off"]["path"] for c in panels["panels"]["ranking"]],
    "teachers (clean)": [r["on"]["path"] for r in comparison["current teachers"]],
}
for key in ("parent-0", "control-low-g-48", "cleaner-36", "cleanest-12"):
    sets[key] = [str(ROOT / "gallery/images" / (v["on"] + ".png")) for v in ranked[key].values()]
models = sys.argv[1:]
with ThreadPoolExecutor(6) as pool:
    jobs = {m: {label: [pool.submit(ask, m, p) for p in paths] for label, paths in sets.items()} for m in models}
    table = {m: {label: [j.result() for j in row] for label, row in rows.items()} for m, rows in jobs.items()}
(root / "model-comparison.json").write_text(json.dumps(table, indent=1))
print(f'{"model":<30}' + "".join(f"{label[:16]:>18}" for label in sets) + f'{"busted above clean":>20}')
for m, rows in table.items():
    cells = []
    for label, values in rows.items():
        seen = [v for v in values if v is not None]
        cells.append(f"{100 * fmean(seen):>12.1f} ({len(seen):>2})" if seen else f'{"no answer":>17}')
    clean = [v for label in ("OFF (clean)", "teachers (clean)") for v in rows[label] if v is not None]
    busted = [v for label in ("cleaner-36", "cleanest-12") for v in rows[label] if v is not None]
    flag = f"{sum(v > max(clean) for v in busted)}/{len(busted)}" if clean and busted else "—"
    print(f"{m:<30}" + " ".join(cells) + f"{flag:>20}")
