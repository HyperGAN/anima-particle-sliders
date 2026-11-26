"""Final Boss ability on images that are still finished drawings.

The distinct Boss score multiplied each character's effect by one minus a damage
reading that barely moved between clean and shattered drawings. Here the same
effect term is multiplied by a quality term that reaches zero for a broken
drawing. Quality never asks whether the character still looks like the original
person, so a strong redesign is not penalized for being one.

Drawing quality is read on overlapping tiles at native resolution. A shrunken
image hides the shattered-shard defect this term exists to catch.
"""

from io import BytesIO
from pathlib import Path
from statistics import fmean
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import base64
import hashlib
import json
import os

from PIL import Image

from .design_score import summarize

VERSION = "final-boss-working-v1"
URL = "https://api.nano-gpt.com/api/v1/decisions"
MODEL = "inception/mercury-decide"
TILE = 320
# Quality falls from one at the OFF image's own reading to zero this far above it.
SPAN = 0.30
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


def tiles(path, size=TILE):
    """Four overlapping corner crops as PNG bytes. Pixels are never resampled."""
    with Image.open(path) as source:
        picture = source.convert("RGB")
    width, height = picture.size
    if min(width, height) < size:
        raise ValueError("Image is smaller than one native tile")
    out = []
    for top in (0, height - size):
        for left in (0, width - size):
            buffer = BytesIO()
            picture.crop((left, top, left + size, top + size)).save(buffer, "PNG", optimize=True)
            out.append(buffer.getvalue())
    return out


class DrawingReader:
    """Ask one decision model how cleanly each tile is drawn, keeping every answer."""

    def __init__(self, receipts, model=MODEL):
        self.receipts, self.model = Path(receipts), model
        self.receipts.mkdir(parents=True, exist_ok=True)

    def tile(self, raw):
        binding = dict(version=VERSION, model=self.model, state=STATE, questions=QUESTIONS,
                       tile=hashlib.sha256(raw).hexdigest())
        key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        receipt = self.receipts / f"{key}.json"
        if receipt.exists():
            body = json.loads(receipt.read_bytes())["response"]
        else:
            image = {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}}
            payload = dict(model=self.model, state=[STATE, "image_1:", image], questions=QUESTIONS)
            request = Request(URL, data=json.dumps(payload).encode(), method="POST",
                              headers={"Content-Type": "application/json"})
            request.add_unredirected_header("Authorization", "Bearer " + os.environ["NANOGPT_API_KEY"])
            try:
                with urlopen(request, timeout=120) as response:
                    body = json.load(response)
            except HTTPError as error:
                if error.code in (408, 429, 500, 502, 503, 504):
                    return None  # Transient. Leave no receipt, so a later run asks again.
                body = {"http_status": error.code}
            except OSError:
                return None
            temporary = receipt.with_suffix(".pending")
            temporary.write_text(json.dumps(dict(request=binding, response=body)))
            temporary.replace(receipt)
        try:
            value = body["answers"]["broken_drawing"]["score"]
        except (KeyError, TypeError):
            return None
        return value / (len(QUESTIONS["broken_drawing"]["criteria"]) - 1)

    def broken(self, path):
        """Mean reading over the four tiles; zero is clean and one is fragments."""
        values = [self.tile(raw) for raw in tiles(path)]
        return None if None in values else fmean(values)


def quality(broken, off_broken, one_figure):
    """One for a finished single-figure drawing, zero for a broken or split one."""
    if None in (broken, off_broken, one_figure):
        return None
    drawn = 1 - min(1.0, max(0.0, broken - off_broken) / SPAN)
    return min(drawn, one_figure)


def score_case(boss, original, quality_):
    """The distinct Boss effect term on an image that is still a finished drawing."""
    if None in (boss, original, quality_):
        return None
    if not all(0 <= value <= 1 for value in (boss, original, quality_)):
        raise ValueError("Readings must lie in the unit interval")
    return max(0.0, boss - original) * boss**3 * quality_


def working_score(cases, pairs):
    """Mean over character pairs, discounted where two characters received one design."""
    if None in cases.values() or any(pair["repeat"] is None for pair in pairs):
        return None
    return summarize(cases, pairs)["score"]
