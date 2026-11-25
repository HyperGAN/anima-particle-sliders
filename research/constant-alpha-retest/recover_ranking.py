"""Measure missing ranking judgments on the reserved images under a separate recorded observation."""

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from statistics import fmean
import sys
import evaluate as ev
from lumen_studio.study.outputs import read
from lumen_studio.contracts import atomic_json, file_hash
from lumen_studio.study.judge import StudyJudge
from release_tools.final_boss_visible_damage import score_case, STATE, QUESTIONS, damage
from release_tools.final_boss_design_scores import summarize
from locations import RESULTS as root

fresh = StudyJudge(root / "ranking-reobservations", "Final Boss")
original_ask = ev.ask


def observed(refs, state, questions, stores=()):
    ans, key = original_ask(refs, state, questions, stores)
    if ans is None:
        images = tuple(ev.image(r["path"], r["sha256"]) for r in refs)
        ans = fresh._decide(images, state, questions)
    return ans, key


ev.ask = observed


def recover(outpath):
    folder = outpath.parent
    out = read(outpath)
    if (
        out.get("score") is not None
        or out.get("alpha") is None
        or (folder / "result-observed.json").exists()
    ):
        return
    a = out["alpha"]
    p = folder / "ranking" / (a.hex() + ".json")
    initial = read(p)
    assert read(folder / "selected.json")["alpha"] == a and initial["score"] is None
    rows = initial["rows"]
    design = out["objective"] == ev.D
    for r in rows.values():
        for name in ("off", "on"):
            assert file_hash(r[name]["path"]) == r[name]["sha256"]
        for field, pair in (
            ("boss_distinction", [r["off"], r["on"]]),
            ("original_boss_distinction", [r["off"], r["off"]]),
        ):
            if r[field] is None:
                answer, _ = ev.ask(pair, **ev.cfg["boss"])
                if not answer:
                    return
                r[field] = answer["boss_distinction"]
        if design and r["image_breakage"] is None:
            answer, _ = ev.ask([r["on"]], STATE, QUESTIONS)
            if not answer:
                return
            r["image_breakage"] = damage(answer)
    pairs = []
    details = None
    if design:
        measured = {tuple(p["cases"]): p for p in initial["pairs"] if p is not None}

        def recoverpair(key):
            return measured.get(key) or ev.pair(*key, rows)

        with ThreadPoolExecutor(4) as pool:
            pairs = list(pool.map(recoverpair, combinations(rows, 2)))
        if any(p is None for p in pairs):
            return
        details = summarize({c: score_case(r) for c, r in rows.items()}, pairs)
    gain = fmean(
        r["boss_distinction"] - r["original_boss_distinction"] for r in rows.values()
    )
    score = details["score"] if design else gain
    original = dict(
        ranking_sha256=file_hash(p),
        result_sha256=file_hash(outpath),
        original_score=None,
        observation="Separate observation of missing judgments on the same reserved images",
    )
    atomic_json(
        folder / "ranking-observed.json",
        dict(
            initial,
            rows=rows,
            pairs=pairs,
            design=details,
            score=score,
            mean_gain=gain,
            original=original,
        ),
    )
    atomic_json(
        folder / "result-observed.json",
        dict(out, score=score, mean_gain=gain, decision="measured", original=original),
    )
    print(folder.parent.name, out["objective"], score, flush=True)


metric = sys.argv[1] if len(sys.argv) > 1 else ev.D
assert metric in (ev.D, ev.G)
for p in sorted((root / "methods").glob(f"*/{metric}/result.json")):
    recover(p)
