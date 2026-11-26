"""Read whether each edited image is still drawn to the standard of its OFF image."""

from concurrent.futures import ThreadPoolExecutor
from statistics import fmean
import contextlib
import io

from lumen_studio.contracts import atomic_json

from campaign import ROOT

with contextlib.redirect_stdout(io.StringIO()):
    from working_metric import cases, image, judge, read, readings, root

STATE = (
    "Image 1 is the original illustration. Image 2 is an edited version. Compare only how well "
    "each is drawn, not what it depicts."
)
QUESTIONS = {
    "broken_drawing": {
        "type": "score",
        "instructions": (
            "Is image 2 drawn to the same finished standard as image 1? Look at the line art, shading and "
            "anatomy across the whole picture. A finished drawing has continuous clean outlines, smooth "
            "deliberate shading and a face and body that hold together. A broken drawing has shapes "
            "shattered into jagged shards or facets, streaks of stray color, smeared or missing facial "
            "features, or forms that dissolve into abstract fragments. A new costume, pose or design is "
            "not a defect."
        ),
        "criteria": [
            "Image 2 is as cleanly drawn as image 1",
            "Slightly rougher than image 1, with small jagged or smeared areas",
            "Clearly rougher: noticeable shards, streaks or a face that no longer holds together",
            "Mostly broken into shards, streaks or fragments; only part of the figure is drawn properly",
            "Not a finished drawing: fragments and abstract shapes",
        ],
    }
}


def finish(off, on):
    answer = judge._decide((image(off), image(on)), STATE, QUESTIONS)
    return None if answer is None else 1 - answer["broken_drawing"]


paths = {}
for entry in read(ROOT / "gallery/results.json")["entries"]:
    pass
comparison = read(ROOT / "damage-definition/comparison.json")
with ThreadPoolExecutor(6) as pool:
    jobs = {}
    for key, row in readings.items():
        for c, v in row.items():
            on = dict(path=str(ROOT / "gallery/images" / (v["on"] + ".png")), sha256=v["on"])
            jobs[key, c] = pool.submit(finish, cases[c]["off"], on)
    teachers = {label: [pool.submit(finish, r["off"], r["on"]) for r in comparison[label]]
                for label in ("current teachers", "teacher proposals")}
    for (key, c), job in jobs.items():
        value = job.result()
        v = readings[key][c]
        v["finish"] = value
        v["works2"] = None if None in (value, v["works"]) else min(v["works"], value)
atomic_json(root / "readings-with-finish.json", readings)
for label, rows in teachers.items():
    values = [j.result() for j in rows]
    print(f"{label:<22} finish {100 * fmean(v for v in values if v is not None):.1f}  min {100 * min(v for v in values if v is not None):.1f}")
print(f'{"row":<22}{"working":>8}{"mean":>7}{"finish":>8}{"was":>6}')
for key in sorted(readings, key=lambda k: -fmean(v["works2"] or 0 for v in readings[k].values())):
    v = list(readings[key].values())
    w = [x["works2"] or 0 for x in v]
    print(f'{key:<22}{sum(x >= 0.5 for x in w):>6}/8{100 * fmean(w):>7.1f}{100 * fmean(x["finish"] or 0 for x in v):>8.1f}'
          f'{sum((x["works"] or 0) >= 0.5 for x in v):>4}/8')
print("cleaner-36", {c: round(100 * (v["finish"] or 0)) for c, v in readings["cleaner-36"].items()})
