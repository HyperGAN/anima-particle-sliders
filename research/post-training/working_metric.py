"""Read, for every ranked image, whether the edit works: the same person, as one intact final boss."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
import json

from particle_sliders.alpha_tune_jev import _Image
from particle_sliders.candidate_failure import FileRecord

from lumen_studio.contracts import atomic_json
from lumen_studio.study.judge import StudyJudge

from campaign import PARENT_STUDY, POLICY, ROOT

D = "distinct_boss_score"
root = ROOT / "working-metric"
root.mkdir(exist_ok=True)
judge = StudyJudge(root / "judgments", "Final Boss")
STATE = (
    "Image 1 is the original illustration of one person. Image 2 is an edit meant to give that "
    "same person a final-boss design while keeping the picture."
)
QUESTIONS = {
    "same_person": {
        "type": "noul",
        "instructions": (
            "Is the central character of image 2 recognizably the same individual as in image 1, with "
            "their face and hair still visible, now wearing a different design? Answer near zero when the "
            "face is hidden, missing or replaced, when the person has become a faceless cloak, creature, "
            "statue, silhouette or abstract shape, or when the head can no longer be found."
        ),
    },
    "one_figure": {
        "type": "noul",
        "instructions": (
            "Does image 2 show exactly one central character, as image 1 does? Answer near zero when "
            "there are two or more figures, a duplicated or split figure, or no clear figure at all."
        ),
    },
    "same_scene": {
        "type": "noul",
        "instructions": (
            "Is image 2 still set in the same place as image 1, with the same surroundings visible "
            "behind the character? A change of costume, pose or camera distance is fine. Answer near zero "
            "when the surroundings were replaced, washed in one color or lost."
        ),
    },
}


def read(path):
    return json.loads(Path(path).read_bytes())


def image(ref):
    return _Image.read(FileRecord.from_path(Path(ref["path"])))


def ask(off, on):
    return judge._decide((image(off), image(on)), STATE, QUESTIONS)


panels = read(PARENT_STUDY / "cohort-retest/panels.json")
cases = {c["id"]: c for c in panels["cases"]}
ranking = panels["panels"]["ranking"]
floors = {r["off"]["sha256"]: r["structural_floor"]
          for r in read(ROOT / "damage-definition/comparison.json")["parent"]}
bases = {"parent-0": PARENT_STUDY / POLICY["parent"]["calibration"] / D}
for base in sorted((ROOT / "arms").glob(f"*/calibration/*/{D}")):
    bases[f"{base.parts[-4]}-{base.parts[-2]}"] = base
jobs = {}
with ThreadPoolExecutor(6) as pool:
    for key, base in bases.items():
        result = read(base / "result.json")
        if result.get("alpha") is None:
            continue
        doc = read(base / "ranking" / (result["alpha"].hex() + ".json"))
        if doc["score"] is None:
            continue
        jobs[key] = {
            c: (doc["rows"][c], result["checks"][c], pool.submit(ask, cases[c]["off"], doc["rows"][c]["on"]))
            for c in ranking
        }
    readings = {}
    for key, row in jobs.items():
        readings[key] = {}
        for c, (measured, check, job) in row.items():
            answer = job.result()
            boss = measured["boss_distinction"]
            intact = 1 - max(0.0, max(check["checks"].values()) - floors[measured["off"]["sha256"]])
            parts = dict(boss=boss, intact=intact, **(answer or {}))
            # An edit works only as far as its weakest requirement.
            works = None if answer is None else min(boss, intact, answer["same_person"], answer["one_figure"])
            readings[key][c] = dict(on=measured["on"]["sha256"], works=works, **parts)
atomic_json(root / "readings.json", readings)
print(f'{"row":<22}{"working":>8}{"mean":>7}{"worst":>7}{"boss":>7}{"person":>8}{"one":>6}{"intact":>8}{"scene":>7}')
order = sorted(readings, key=lambda k: -fmean(v["works"] or 0 for v in readings[k].values()))
for key in order:
    v = list(readings[key].values())
    w = [x["works"] or 0 for x in v]
    m = lambda name: 100 * fmean(x.get(name) or 0 for x in v)
    print(f'{key:<22}{sum(x >= 0.5 for x in w):>6}/8{100 * fmean(w):>7.1f}{100 * min(w):>7.1f}{m("boss"):>7.1f}'
          f'{m("same_person"):>8.1f}{m("one_figure"):>6.1f}{m("intact"):>8.1f}{m("same_scene"):>7.1f}')
