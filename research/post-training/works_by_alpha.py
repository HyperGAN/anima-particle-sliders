"""Read the working-edit metric at every searched strength of one RAW, on its saved search images."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
import json
import sys

from particle_sliders.candidate_failure import FileRecord

from lumen_studio.contracts import atomic_json
from lumen_studio.vibe_integrity import FrameIntegrity

from campaign import PARENT_STUDY, POLICY, ROOT
from working_metric import D, ask, cases, judge, read, root

name = sys.argv[1]
base = (PARENT_STUDY / POLICY["parent"]["calibration"] / D if name == "parent-0"
        else ROOT / "arms" / name.rsplit("-", 1)[0] / "calibration" / name.rsplit("-", 1)[1] / D)
integrity = FrameIntegrity(root / "judgments")


def structure(ref):
    return max(integrity.check(FileRecord.from_path(Path(ref["path"])))["checks"].values())


docs = sorted((read(p) for p in (base / "search").glob("*.json")), key=lambda d: d["identity"]["alpha"])
with ThreadPoolExecutor(6) as pool:
    floors = {c: pool.submit(structure, cases[c]["off"]) for c in docs[0]["rows"]}
    jobs = [(d, {c: (r, pool.submit(ask, cases[c]["off"], r["on"]), pool.submit(structure, r["on"]))
                 for c, r in d["rows"].items()}) for d in docs]
    out = []
    for d, row in jobs:
        works = []
        for c, (r, answer, broken) in row.items():
            a = answer.result()
            intact = 1 - max(0.0, broken.result() - floors[c].result())
            works.append(0.0 if a is None else min(r["boss_distinction"], intact, a["same_person"], a["one_figure"]))
        out.append(dict(alpha=d["identity"]["alpha"], works=works, gain=d["mean_gain"], score=d["score"]))
atomic_json(root / f"by-alpha-{name}.json", out)
print(name)
print(f'{"alpha":>8}{"working":>9}{"mean":>7}{"gain":>7}{"current score":>15}')
for o in out:
    print(f'{o["alpha"]:>8.4g}{sum(w >= 0.5 for w in o["works"]):>7}/4{100 * fmean(o["works"]):>7.1f}'
          f'{100 * o["gain"]:>7.1f}{100 * (o["score"] or 0):>15.2f}')
