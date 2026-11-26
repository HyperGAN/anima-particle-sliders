"""Choose each scene's Boss teacher by the working Boss score of the teacher images themselves.

The slider can only be as good as what it is taught. Every candidate teacher image is read
exactly as a slider's output is read on the board: is it a final boss, one whole finished
figure, in the scene it started in. Candidates from different scenes are compared for
repeated designs in both orders. The chosen set is the one with the highest working Boss
score over the twelve training scenes, so teachers that share one design cost each other.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from statistics import fmean
import sys

import torch
from particle_sliders.alpha_tune_jev import _Image
from particle_sliders.candidate_failure import FileRecord

from lumen_studio.cache import TargetCache
from lumen_studio.contracts import atomic_json, file_hash
from lumen_studio.study import working_score as ws
from lumen_studio.study.boss_targets import quiet_boss_background, replace_boss_teachers
from lumen_studio.study.design_score import summarize
from lumen_studio.study.judge import StudyJudge
from lumen_studio.study.objectives import protocol
from lumen_studio.study.outputs import Outputs, read

from campaign import CODE, PARENT_STUDY, POLICY, ROOT, open_runtime

root = ROOT / "teachers"
cache = TargetCache(PARENT_STUDY / POLICY["parent"]["cache"], verify=True)
assert cache.index["fingerprint"] == POLICY["parent"]["cache_fingerprint"]
# Each round adds proposals to the ones before it. ``working`` is the first round.
name = sys.argv[1] if len(sys.argv) > 1 else "working"
sources = POLICY["targets"][name + "-quiet"]["proposals"]
proposals = {}
for source in sources:
    for scene, options in read(CODE / source).items():
        proposals.setdefault(scene, []).extend(options)
judge = StudyJudge(ROOT / "working-metric" / "judgments", "Final Boss")
reader = ws.DrawingReader(ROOT / "working-search" / "drawing-receipts")
cfg = protocol()


def picture(record):
    return _Image.read(FileRecord.from_path(Path(record["path"])))


def value(answer, key):
    return None if answer is None else answer[key]


def measure(off, on):
    a, b = picture(off), picture(on)
    paired = judge._decide((a, b), ws.PAIR_STATE, ws.PAIR_QUESTIONS)
    # One neutral image never gets an answer for its own Boss reading. The score floors that
    # reading at BOSS_FLOOR anyway, and every answered scene reads far below it.
    original = value(judge._decide((a, a), **cfg["boss"]), "boss_distinction")
    row = dict(
        image=on,
        original=ws.BOSS_FLOOR if original is None else original,
        original_answered=original is not None,
        boss=value(judge._decide((a, b), **cfg["boss"]), "boss_distinction"),
        broken=reader.broken(on["path"]),
        off_broken=reader.broken(off["path"]),
        one_figure=value(paired, "one_figure"),
        same_person=value(paired, "same_person"),
        figure=value(judge._decide((b,), ws.FIGURE_STATE, ws.FIGURE_QUESTIONS), "figure"),
        recolored=value(judge._decide((a, b), ws.RECOLOR_STATE, ws.RECOLOR_QUESTIONS), "recolored"),
    )
    row["quality"] = ws.quality(row["broken"], row["off_broken"], row["one_figure"], row["figure"])
    row["case"] = ws.score_case(row["boss"], row["original"], row["quality"], row["recolored"])
    return row


def repeat(left, right):
    x, y = picture(left), picture(right)
    values = [value(judge._decide(order, **cfg["repeat"]), "repeat") for order in ((x, y), (y, x))]
    return None if None in values else fmean(values)


rt = open_runtime()
try:
    out = Outputs(root, rt, "Final Boss", {})
    scenes = []
    for shard in cache.index["shards"]:
        row = torch.load(cache.directory / shard["path"], map_location="cpu", weights_only=True)["row"]
        neutral = out.image(row["neutral"], shard["seed"])
        current = out.image(row["positive"], shard["seed"])
        assert current.sha256 == row["selection"]["positive"]["output"]["sha256"]
        options = [dict(name="current", positive=row["positive"], image=asdict(current))]
        for p in proposals.get(shard["row"], []):
            options.append(dict(p, image=asdict(out.image(p["positive"], shard["seed"]))))
        scenes.append(dict(scene=shard["row"], seed=shard["seed"], neutral=asdict(neutral), options=options))
    with ThreadPoolExecutor(24) as pool:
        jobs = [(o, pool.submit(measure, s["neutral"], o["image"]),
                 pool.submit(out.boss_teacher_pair, dict(output=s["neutral"]), dict(output=o["image"])))
                for s in scenes for o in s["options"]]
        for option, reading, separate in jobs:
            option.update(reading.result(), separate=separate.result()["status"])
        # A proposal must be one transformed person, fully measured.
        for s in scenes:
            s["eligible"] = [o for o in s["options"] if o["case"] is not None
                             and (o["name"] == "current" or o["separate"] == "pass")]
            assert any(o["name"] == "current" for o in s["eligible"]), "A current teacher is unmeasured; rerun"
        pairs = {}
        for a, b in combinations(scenes, 2):
            for x in a["eligible"]:
                for y in b["eligible"]:
                    pairs[x["image"]["sha256"], y["image"]["sha256"]] = pool.submit(repeat, x["image"], y["image"])
        pairs = {key: job.result() for key, job in pairs.items()}
    if any(v is None for v in pairs.values()):
        raise SystemExit("A repeat reading is unanswered; rerun to retry the judge")

    def score(choice):
        cases = {s["scene"]: o["case"] for s, o in zip(scenes, choice)}
        compared = [dict(cases=[a["scene"], b["scene"]], repeat=pairs[x["image"]["sha256"], y["image"]["sha256"]])
                    for (a, x), (b, y) in combinations(zip(scenes, choice), 2)]
        return summarize(cases, compared)

    current = [next(o for o in s["eligible"] if o["name"] == "current") for s in scenes]
    choice = [max(s["eligible"], key=lambda o: o["case"]) for s in scenes]
    improved = True
    while improved:
        improved = False
        for index, s in enumerate(scenes):
            for o in s["eligible"]:
                trial = [*choice[:index], o, *choice[index + 1:]]
                if score(trial)["score"] > score(choice)["score"] + 1e-12:
                    choice, improved = trial, True
    if all(o["name"] != "current" for o in choice):
        # A derived cache keeps at least one parent scene. Give back the cheapest one.
        index = max(range(len(scenes)),
                    key=lambda i: score([*choice[:i], current[i], *choice[i + 1:]])["score"])
        choice[index] = current[index]
    replacements = {
        s["scene"]: dict(positive=o["positive"], explicit=o["explicit"], selected=dict(
            output=dict(prompt=o["positive"], seed=s["seed"], path=o["image"]["path"], sha256=o["image"]["sha256"]),
            readings={k: o[k] for k in ("boss", "original", "quality", "recolored", "case")}))
        for s, o in zip(scenes, choice) if o["name"] != "current"
    }
    atomic_json(root / f"{name}-selection.json", dict(
        proposals={source: file_hash(CODE / source) for source in sources}, version=ws.VERSION,
        current=score(current), chosen=score(choice),
        scenes=[dict(scene=s["scene"], selected=o["name"], options=s["options"]) for s, o in zip(scenes, choice)]))
    for s, o in zip(scenes, choice):
        print(s["scene"][-8:], f"{o['name']:<20}", " ".join(
            f"{c['name']}={'—' if c['case'] is None else format(100 * c['case'], '.0f')}" for c in s["options"]), flush=True)
    print("current teachers", score(current), "chosen", score(choice), flush=True)
    working = replace_boss_teachers(rt, cache, ROOT / "targets" / name, replacements)
    quiet = quiet_boss_background(rt, working, ROOT / "targets" / f"{name}-quiet", 0.0)
    print(name, working.index["fingerprint"], f"{name}-quiet", quiet.index["fingerprint"], flush=True)
finally:
    rt.close()
