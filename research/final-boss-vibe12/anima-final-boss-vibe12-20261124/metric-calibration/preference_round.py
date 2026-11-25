"""Compare finished edits in both presentation orders using pinned, matched native renders."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
from math import isfinite
from pathlib import Path
from statistics import fmean
import json

from review import HERE, ROOT, assess, harness, judge
from finished_image import score_case

OUT = HERE / "preference-round"
SHOWN = ("ranger", "smith", "stationer")
STATE = ("Image_1 is the original. Image_2 and image_3 are alternative finished edits of it. "
         "Choose which edit better delivers the character concept 'Final Boss' as a usable finished illustration.")
QUESTIONS = {"preference": {"type": "score", "instructions": (
    "Which character would you more readily recognize as the game's final encounter? "
    "Look for an imposing, individual design you could recognize again: costume, body, silhouette, "
    "or a signature visual motif can carry it. A stern face, generic soldier's armor, or a darker scene alone "
    "gives little evidence. A masked figure, monster, or radical transformation can fully succeed. "
    "Favor the stronger character effect unless visible drawing damage materially obscures it. "
    "Use the middle answer when neither finished edit is meaningfully better."),
    "criteria": ["Image_2 clearly better", "Image_2 slightly better", "Equivalent finished edits",
                 "Image_3 slightly better", "Image_3 clearly better"]}}
PROTOCOL = ("Round robin on the five highest-ranked recipe winners at the frozen snapshot. Equal weight per case "
            "and opponent. One observation per image order. Positive margin favors the first named method. "
            "Check all six ordered pairs from the user's 4,1,3,2 panel and both 2>1 and 3>1 comparisons "
            "from the second panel. Record 2 vs 3 separately. Consider promotion only if the candidate improves "
            "strict agreement over v2 on these eight comparisons, has no losses vs slider-off controls, and "
            "agrees in both presentation orders on a majority of the current individual case comparisons. "
            "These exposed panels establish calibration, not independent generalization.")
OPT = ROOT.parent / "anima-final-boss-optimization-20261123"


def old_case(ref, edited, scores):
    damage = assess(edited)
    result = dict(scores, image_breakage=damage)
    return {"off": ref, "on": edited, "final_boss_score": score_case(result)}


def freeze():
    path = OUT / "panels.json"
    if path.is_file():
        return harness.read(path)
    snapshot = (ROOT / "storage/alpha-standings.json").read_bytes()
    board = json.loads(snapshot)
    top = [row for row in board["rows"] if row["best_for_recipe"]][:5]
    groups = {"current": {r["method"]: r["cases"] for r in top}}
    first = harness.read(OPT / "boss-pairwise-audit/report.json")
    second = harness.read(OPT / "image-breakage-audit/report.json")
    old = harness.read(OPT / "leaderboard.json")["rows"]
    groups["historical-effect"] = {}
    for ident, row_key in first["entries"].items():
        groups["historical-effect"][ident] = {
            c: old_case(first["cases"][c]["original"], first["cases"][c]["edits"][ident], old[row_key]["cases"][c])
            for c in SHOWN}
    groups["historical-damage"] = {}
    for ident, (method, profile, alpha) in second["rows"].items():
        row = old[f"{method}-{profile}-{alpha.hex()}"]
        groups["historical-damage"][ident] = {
            c: old_case(assets["inputs"]["off"], assets["inputs"]["on"], row["cases"][c])
            for c, assets in second["observations"][ident].items() if c in SHOWN}
    method = top[0]["method"]
    controls = {c:groups["current"][method][c] for c in SHOWN}
    groups["control"] = {"edited":controls,"off":{c:dict(r, on=r["off"], final_boss_score=0) for c,r in controls.items()}}
    panels = {"snapshot_sha256": sha256(snapshot).hexdigest(), "version": board["scoring_version"],
              "settings": {r["method"]: {"profile":r["profile"], "alpha":r["alpha"], "score":r["means"]["final_boss_score"]}
                           for r in top}, "groups": groups}
    harness.write(path, panels)
    return panels


def run(panels, destination, state, questions, protocol, receipts=None):
    judge()
    from particle_sliders.alpha_tune_jev import JevJudge, _Image, _digest, JEV_MODEL, JEV_URL
    from particle_sliders.candidate_failure import FileRecord
    client = JevJudge(receipts or destination / "receipts", "Finished Final Boss edit")
    @lru_cache(maxsize=96)
    def load_image(path, sha):
        return _Image.read(FileRecord(path, Path(path).stat().st_size, sha))

    def decide(ref, left, right):
        images = tuple(load_image(a["path"], a["sha256"]) for a in (ref, left, right))
        binding = dict(endpoint=JEV_URL, model=JEV_MODEL, state=state, questions=questions,
                       images=[image.identity() for image in images])
        values = client._decide(images, state, questions)
        if values is None or not isfinite(values["preference"]):
            raise RuntimeError(f"Unmeasured Jev comparison {_digest(binding)}")
        return {"value":values["preference"], "receipt":_digest(binding)}

    comparisons = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for group, entries in panels["groups"].items():
            for a,b in combinations(entries, 2):
                cases = {}
                for c,x in entries[a].items():
                    y = entries[b][c]
                    if x["off"]["sha256"] != y["off"]["sha256"]:
                        raise ValueError(f"Unmatched originals for {group} {a} {b} {c}")
                    off = x["off"]
                    cases[c] = (pool.submit(decide, off, x["on"], y["on"]),
                                pool.submit(decide, off, y["on"], x["on"]), x, y)
                comparisons.append((group,a,b,cases))
        output = []
        for group,a,b,jobs in comparisons:
            cases = {}
            for c,(forward,reverse,x,y) in jobs.items():
                ab,ba = forward.result(),reverse.result()
                cases[c] = {"forward":ab, "reverse":ba, "margin":ba["value"]-ab["value"],
                            "v2_margin":(x["final_boss_score"]-y["final_boss_score"] if None not in
                                         (x["final_boss_score"], y["final_boss_score"]) else None),
                            "both_orders_agree":(ab["value"]-.5)*(ba["value"]-.5)<0,
                            "images":[v["sha256"] for v in (x["off"],x["on"],y["on"])]}
            output.append({"group":group, "a":a, "b":b, "cases":cases,
                           "margin":fmean(v["margin"] for v in cases.values()),
                           "v2_margin":(fmean(v["v2_margin"] for v in cases.values())
                                        if all(v["v2_margin"] is not None for v in cases.values()) else None)})
            print(group, a, b, round(output[-1]["margin"],3), flush=True)
    harness.write(destination / "report.json", {"protocol":protocol, "model":JEV_MODEL, "state":state, "questions":questions,
                  "snapshot_sha256":panels["snapshot_sha256"], "settings":panels["settings"], "comparisons":output})


if __name__ == "__main__":
    judge()
    run(freeze(), OUT, STATE, QUESTIONS, PROTOCOL)
