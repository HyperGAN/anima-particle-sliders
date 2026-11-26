"""Compare candidate damage definitions on saved images. No image is rendered."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
import json

from particle_sliders.alpha_tune_jev import _Image
from particle_sliders.candidate_failure import FileRecord

from lumen_studio.contracts import atomic_json
from lumen_studio.study.design_score import QUESTIONS, STATE, damage
from lumen_studio.study.judge import StudyJudge

from campaign import PARENT_STUDY, POLICY, ROOT

D = "distinct_boss_score"
root = ROOT / "damage-definition"
root.mkdir(exist_ok=True)
judge = StudyJudge(root / "judgments", "Final Boss")
PAIR = (
    "Image 1 is the original illustration. Image 2 is an edit that was meant to redesign "
    "only the central character as a final boss."
)
TAKEOVER = {
    "takeover": {
        "type": "score",
        "instructions": (
            "How much of the original picture did the edit overwrite beyond the character's own design? "
            "Consider whether the same place is still recognizable, whether the lighting and colors of the "
            "surroundings survive, and whether there is still one central person. A dramatic new costume, "
            "armor, crown, mantle or pose is the intended edit and is not takeover."
        ),
        "criteria": [
            "The surroundings and their lighting are intact; only the character changed",
            "The surroundings are intact with a slight tint or a few altered details",
            "The place is recognizable but clearly recolored, darkened or partly replaced",
            "Most of the scene was replaced or recolored; little of the original picture remains",
            "The original picture is gone: a different scene, no clear person, or several figures in place of one",
        ],
    }
}


def read(path):
    return json.loads(Path(path).read_bytes())


def image(ref):
    return _Image.read(FileRecord.from_path(Path(ref["path"])))


def accident(ref):
    answer = judge._decide((image(ref),), STATE, QUESTIONS)
    return None if answer is None else damage(answer)


def takeover(off, on):
    answer = judge._decide((image(off), image(on)), PAIR, TAKEOVER)
    return None if answer is None else answer["takeover"]


panels = read(PARENT_STUDY / "cohort-retest/panels.json")
cases = {c["id"]: c for c in panels["cases"]}
ranking = panels["panels"]["ranking"]
sets = {}


def board(label, base):
    result = read(base / "result.json")
    rows = read(base / "ranking" / (result["alpha"].hex() + ".json"))["rows"]
    sets[label] = [
        dict(off=cases[c]["off"], on=rows[c]["on"], accident=rows[c]["image_breakage"],
             gain=rows[c]["boss_distinction"] - rows[c]["original_boss_distinction"],
             structural=max(result["checks"][c]["checks"].values()))
        for c in ranking
    ]


board("parent", PARENT_STUDY / POLICY["parent"]["calibration"] / D)
for arm, extra in (("cleanest", 12), ("cleanest-seed2", 12), ("control", 12), ("control-warm", 48),
                   ("continued", 12), ("control-low-g", 48)):
    board(f"{arm} +{extra}", ROOT / "arms" / arm / "calibration" / str(extra) / D)
current = read(ROOT / "teachers/current.json")
selection = read(ROOT / "teachers/selection-cleaner.json")
sets["current teachers"] = [
    dict(off=s["neutral_image"], on=s["current"]["image"], accident=s["current"]["damage"],
         gain=s["current"]["gain"], structural=None) for s in current["scenes"]
]
neutral = {s["scene"]: s["neutral_image"] for s in current["scenes"]}
sets["teacher proposals"] = [
    dict(off=neutral[scene], on=o["image"], accident=o["damage"], gain=o["gain"], structural=None)
    for scene, d in selection["decisions"].items() for o in d["proposals"]
]
with ThreadPoolExecutor(6) as pool:
    floors = {}
    for rows in sets.values():
        for row in rows:
            key = row["off"]["sha256"]
            if key not in floors:
                floors[key] = pool.submit(accident, row["off"])
    jobs = [(row, pool.submit(takeover, row["off"], row["on"])) for rows in sets.values() for row in rows]
    for row, job in jobs:
        row["takeover"] = job.result()
        row["floor"] = floors[row["off"]["sha256"]].result()
        row["added"] = None if row["floor"] is None else max(0.0, row["accident"] - row["floor"])
atomic_json(root / "comparison.json", sets)
print(f'{"set":<22}{"n":>3}{"gain":>7}{"now":>7}{"OFF":>7}{"added":>7}{"struct":>8}{"takeover":>10}')
for label, rows in sets.items():
    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return f"{100 * fmean(values):.1f}" if values else "—"
    print(f'{label:<22}{len(rows):>3}{mean("gain"):>7}{mean("accident"):>7}{mean("floor"):>7}{mean("added"):>7}'
          f'{mean("structural"):>8}{mean("takeover"):>10}')

# The structural check is already read on every ranked image. Read it on the matched OFF
# images and the teachers too, so each definition can be compared above its own floor.
from lumen_studio.vibe_integrity import FrameIntegrity  # noqa: E402

integrity = FrameIntegrity(root / "judgments")
assert integrity.calibrate()["passed"]


def structure(ref):
    return max(integrity.check(FileRecord.from_path(Path(ref["path"])))["checks"].values())


with ThreadPoolExecutor(6) as pool:
    cache, jobs = {}, []
    for rows in sets.values():
        for row in rows:
            for side in ("off", "on"):
                key = row[side]["sha256"]
                if side == "on" and row["structural"] is not None:
                    continue
                if key not in cache:
                    cache[key] = pool.submit(structure, row[side])
    for rows in sets.values():
        for row in rows:
            row["structural_floor"] = cache[row["off"]["sha256"]].result()
            if row["structural"] is None:
                row["structural"] = cache[row["on"]["sha256"]].result()
            row["structural_added"] = max(0.0, row["structural"] - row["structural_floor"])
atomic_json(root / "comparison.json", sets)
print(f'\n{"set":<22}{"struct":>8}{"OFF":>7}{"added":>7}{"worst":>7}')
for label, rows in sets.items():
    print(f'{label:<22}{100 * fmean(r["structural"] for r in rows):>8.1f}'
          f'{100 * fmean(r["structural_floor"] for r in rows):>7.1f}'
          f'{100 * fmean(r["structural_added"] for r in rows):>7.1f}'
          f'{100 * max(r["structural_added"] for r in rows):>7.1f}')


# Rescore each ranked row at its kept alpha with damage read as structure added over OFF.
from lumen_studio.study.design_score import score_case, summarize  # noqa: E402

floor={r["off"]["sha256"]: r["structural_floor"] for r in sets["parent"]}
def rows_for(base):
    res=read(base / "result.json")
    if res.get("alpha") is None: return None
    doc=read(base / "ranking" / (res["alpha"].hex() + ".json"))
    if doc["score"] is None: return None
    cases={}; dmg=[]
    for c,r in doc["rows"].items():
        s=max(res["checks"][c]["checks"].values()); added=max(0.0, s-floor[r["off"]["sha256"]]); dmg.append(added)
        cases[c]=score_case(dict(boss_distinction=r["boss_distinction"], original_boss_distinction=r["original_boss_distinction"], image_breakage=added))
    new=summarize(cases, doc["pairs"])["score"]
    return 100*doc["score"], 100*new, 100*sum(dmg)/len(dmg)
out=[("parent",0,*rows_for(PARENT_STUDY/POLICY["parent"]["calibration"]/D))]
for base in sorted((ROOT/"arms").glob("*/calibration/*/"+D)):
    v=rows_for(base)
    if v: out.append((base.parts[-4], int(base.parts[-2]), *v))
old=sorted(out,key=lambda r:-r[2]); new=sorted(out,key=lambda r:-r[3])
print(f'{"new":>3} {"old":>3}  {"arm":<18}{"+":>3}{"score now":>10}{"rescored":>10}{"added structural":>18}')
for i,r in enumerate(new,1):
    print(f'{i:>3} {old.index(r)+1:>3}  {r[0]:<18}{r[1]:>3}{r[2]:>10.2f}{r[3]:>10.2f}{r[4]:>18.1f}')
atomic_json(root / "rescored.json", [dict(arm=r[0], extra=r[1], score=r[2], rescored=r[3], added_structural=r[4]) for r in new])
