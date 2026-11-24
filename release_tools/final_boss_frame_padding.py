"""Measure straight near-white margins in native Final Boss renders."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


PIXEL_MIN = 247
LINE_FRACTION_MIN = 0.995
MIN_WIDTH = 8


def width_from_edge(white_fraction):
    gaps = np.flatnonzero(white_fraction < LINE_FRACTION_MIN)
    width = int(gaps[0]) if len(gaps) else len(white_fraction)
    return width if MIN_WIDTH <= width < len(white_fraction) else 0


def white_margins(image):
    """Return the widths of near-white bands attached to the four image edges."""
    white = (np.asarray(image.convert("RGB")) >= PIXEL_MIN).all(axis=2)
    rows = white.mean(axis=1)
    columns = white.mean(axis=0)
    return {"top": width_from_edge(rows), "bottom": width_from_edge(rows[::-1]),
            "left": width_from_edge(columns), "right": width_from_edge(columns[::-1])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.images:
        with Image.open(path) as image:
            print(json.dumps({"image": str(path), "edges": white_margins(image)}))
