"""Teach further scenes beside the chosen teachers, each with its best-scoring Boss teacher.

None of the twelve parent scenes is a portrait, and the slider turns a portrait into a
full-length shot of a repainted room. The added scenes are portraits whose characters and
rooms differ from every search and ranking character. Each gets three proposed teachers,
read as the board reads a slider's output, and the set with the highest working Boss score
over all scenes is kept. The foreground matte is the study's U²-Net procedure.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import combinations
from statistics import fmean
import subprocess
import sys

import torch

from lumen_studio.cache import TargetCache
from lumen_studio.contracts import atomic_json, file_hash
from lumen_studio.study import working_score as ws
from lumen_studio.study.boss_targets import add_boss_scenes
from lumen_studio.study.design_score import summarize
from lumen_studio.study.outputs import Outputs, read

from campaign import CODE, POLICY, ROOT, open_runtime
from teacher_score import measure, repeat

CV_PYTHON = "/mnt/nvme/ai/conda/anaconda3/bin/python3"
name = sys.argv[1]
spec = POLICY["targets"][name]
parent = TargetCache(ROOT / "targets" / spec["parent"], verify=True)
added = read(CODE / spec["scenes"])
root = ROOT / "teachers"
rt = open_runtime()
try:
    out = Outputs(root, rt, "Final Boss", {})
    fixed = []
    for shard in parent.index["shards"]:
        row = torch.load(parent.directory / shard["path"], map_location="cpu", weights_only=True)["row"]
        sides = [{k: row["selection"][side]["output"][k] for k in ("path", "sha256")} for side in ("neutral", "positive")]
        fixed.append(dict(scene=shard["row"], neutral=sides[0], options=[dict(name="current", image=sides[1])]))
    fresh = []
    for scene, entry in added.items():
        neutral = asdict(out.image(entry["neutral"], entry["seed"]))
        options = [dict(p, image=asdict(out.image(p["positive"], entry["seed"]))) for p in entry["proposals"]]
        fresh.append(dict(scene=scene, seed=entry["seed"], prompt=entry["neutral"], neutral=neutral, options=options))
    scenes = fixed + fresh
    with ThreadPoolExecutor(24) as pool:
        jobs = [(s, o, pool.submit(measure, s["neutral"], o["image"]),
                 pool.submit(out.boss_teacher_pair, dict(output=s["neutral"]), dict(output=o["image"])))
                for s in scenes for o in s["options"]]
        for s, option, reading, separate in jobs:
            option.update(reading.result(), separate=separate.result()["status"])
        for s in scenes:
            s["eligible"] = [o for o in s["options"] if o["case"] is not None
                             and (s in fixed or o["separate"] == "pass")]
            if not s["eligible"]:
                raise SystemExit(f"No measured teacher for {s['scene']}; rerun or write better proposals")
        pairs = {}
        for a, b in combinations(scenes, 2):
            for x in a["eligible"]:
                for y in b["eligible"]:
                    pairs[x["image"]["sha256"], y["image"]["sha256"]] = pool.submit(repeat, x["image"], y["image"])
        pairs = {key: job.result() for key, job in pairs.items()}
    # The judge refuses a few image pairs every time. They take the mean of the answered pairs.
    unanswered = [key for key, v in pairs.items() if v is None]
    if len(unanswered) > len(pairs) // 50:
        raise SystemExit(f"{len(unanswered)} repeat readings are unanswered; rerun to retry the judge")
    typical = fmean(v for v in pairs.values() if v is not None)
    pairs = {key: typical if v is None else v for key, v in pairs.items()}
    print(len(unanswered), "of", len(pairs), "repeat readings unanswered", flush=True)

    def score(choice):
        cases = {s["scene"]: o["case"] for s, o in zip(scenes, choice)}
        compared = [dict(cases=[a["scene"], b["scene"]], repeat=pairs[x["image"]["sha256"], y["image"]["sha256"]])
                    for (a, x), (b, y) in combinations(zip(scenes, choice), 2)]
        return summarize(cases, compared)

    choice = [max(s["eligible"], key=lambda o: o["case"]) for s in scenes]
    improved = True
    while improved:
        improved = False
        for index, s in enumerate(scenes):
            for o in s["eligible"]:
                trial = [*choice[:index], o, *choice[index + 1:]]
                if score(trial)["score"] > score(choice)["score"] + 1e-12:
                    choice, improved = trial, True
    (root / "mattes").mkdir(exist_ok=True)
    built = {}
    for s, o in zip(scenes, choice):
        print(s["scene"][-20:], f"{o['name']:<20}", " ".join(
            f"{c['name']}={'—' if c['case'] is None else format(100 * c['case'], '.0f')}" for c in s["options"]), flush=True)
        if s in fixed:
            continue
        matte = root / "mattes" / f"{s['scene']}.png"
        coverage = subprocess.run([CV_PYTHON, str(CODE / "matte.py"), s["neutral"]["path"], str(matte)],
                                  check=True, capture_output=True, text=True).stdout.strip()
        built[s["scene"]] = dict(neutral=s["prompt"], positive=o["positive"], explicit=o["explicit"], seed=s["seed"],
                                 neutral_image={k: s["neutral"][k] for k in ("path", "sha256")},
                                 positive_image={k: o["image"][k] for k in ("path", "sha256")}, matte=str(matte))
        o["matte_coverage"] = float(coverage)
    atomic_json(root / f"{name}-selection.json", dict(
        scenes_file=file_hash(CODE / spec["scenes"]), parent=parent.index["fingerprint"], version=ws.VERSION,
        parent_only=score(choice[:len(fixed)]) if len(fixed) > 1 else None, chosen=score(choice),
        scenes=[dict(scene=s["scene"], selected=o["name"], options=s["options"]) for s, o in zip(scenes, choice)]))
    print("teaching set", score(choice), flush=True)
    cache = add_boss_scenes(rt, parent, ROOT / "targets" / name, built)
    print(name, cache.index["fingerprint"], len(cache), "fields", flush=True)
finally:
    rt.close()
