"""Re-pick each RAW's constant alpha by the working Boss score and rank it on the held-out characters."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
import shutil
import sys

from campaign import CODE, PARENT_STUDY, POLICY, ROOT, open_runtime, parent_arm

panel = ROOT / "cohort-retest"
panel.mkdir(exist_ok=True)
if not (panel / "panels.json").exists():
    shutil.copyfile(PARENT_STUDY / "cohort-retest/panels.json", panel / "panels.json")
sys.path.insert(0, str(CODE.parent / "constant-alpha-retest"))

from evaluate import D, cases, cfg, distinction, pair  # noqa: E402
from particle_sliders.alpha_tune_jev import _Image  # noqa: E402
from particle_sliders.candidate_failure import FileRecord  # noqa: E402

from lumen_studio.contracts import atomic_json  # noqa: E402
from lumen_studio.particles import ParticleAdapter  # noqa: E402
from lumen_studio.study import working_score as ws  # noqa: E402
from lumen_studio.study.judge import StudyJudge  # noqa: E402
from lumen_studio.study.outputs import Outputs, read  # noqa: E402
from lumen_studio.study.training import arm_path  # noqa: E402

import paired_questions  # noqa: E402

home = ROOT / "working-search"
reader = ws.DrawingReader(home / "drawing-receipts")
judge = StudyJudge(ROOT / "working-metric" / "judgments", "Final Boss")
STEP = 1 / 16


def image(ref):
    return _Image.read(FileRecord.from_path(Path(ref["path"])))


def figure(off, on):
    return judge._decide((image(off), image(on)), paired_questions.STATE, paired_questions.QUESTIONS)


def source(name):
    """The earlier calibration of this RAW and the checkpoint it measured."""
    arm, extra = name.rsplit("-", 1)
    if arm == "parent":
        return PARENT_STUDY / POLICY["parent"]["calibration"] / D, read(parent_arm() / "cursor.json")["raw"], POLICY["parent"]["step"]
    spec = POLICY["arms"][arm]
    step = int(extra) if spec["entry"] == "fork" else POLICY["parent"]["step"] + int(extra)
    trained = arm_path(ROOT / "arms" / arm, spec.get("rates", POLICY["rates"]))
    return ROOT / "arms" / arm / "calibration" / extra / D, read(trained / "fit" / f"{step}.json")["raw"], step


def search(rt, name, pool):
    base, raw, step = source(name)
    out = home / name
    if (out / "result.json").exists():
        return read(out / "result.json")
    out.mkdir(parents=True, exist_ok=True)
    outputs = Outputs(out, rt, "Final Boss", {})
    adapter = ParticleAdapter(rt.transformer).to(rt.device)
    meta = adapter.load_export(raw["path"], model_identity=rt.identity)
    assert meta["step"] == step and meta["weights"] == "live"
    adapter.requires_grad_(False).eval()
    key = f"working-{name}"
    rt.mixer.add(key, adapter)
    try:
        def measure(alpha, phase):
            path = out / phase / (alpha.hex() + ".json")
            if path.exists():
                return read(path)
            names = cfg["panels"][phase]
            earlier = base / phase / (alpha.hex() + ".json")
            rows = {}
            if earlier.exists() and read(earlier)["identity"]["raw"] == raw["sha256"]:
                old = read(earlier)
                rows = {c: dict(on=r["on"], boss=r["boss_distinction"], original=r["original_boss_distinction"])
                        for c, r in old["rows"].items()}
                pairs = old["pairs"] or None
            else:
                for c in names:
                    rows[c] = dict(on=asdict(outputs.image(**cases[c]["request"], raw=raw["sha256"],
                                                           variation=key, gains=[alpha] * 10)))
                jobs = {c: (pool.submit(distinction, cases[c]["off"], cases[c]["off"]),
                            pool.submit(distinction, cases[c]["off"], rows[c]["on"])) for c in names}
                for c, (before, after) in jobs.items():
                    rows[c].update(original=before.result(), boss=after.result())
                pairs = None
            if not pairs:
                from itertools import combinations

                evaluated = {c: dict(on=r["on"]) for c, r in rows.items()}
                pairs = [j.result() for j in [pool.submit(pair, a, b, evaluated) for a, b in combinations(names, 2)]]
            readings = {c: (pool.submit(reader.broken, rows[c]["on"]["path"]),
                            pool.submit(reader.broken, cases[c]["off"]["path"]),
                            pool.submit(figure, cases[c]["off"], rows[c]["on"])) for c in names}
            for c, (broken, floor, answer) in readings.items():
                a = answer.result()
                row = rows[c]
                row.update(broken=broken.result(), off_broken=floor.result(),
                           one_figure=a and a["one_figure"], same_person=a and a["same_person"])
                row["quality"] = ws.quality(row["broken"], row["off_broken"], row["one_figure"])
                row["case"] = ws.score_case(row["boss"], row["original"], row["quality"])
            complete = all(p is not None for p in pairs)
            score = ws.working_score({c: r["case"] for c, r in rows.items()}, pairs) if complete else None
            gains = [r["boss"] - r["original"] for r in rows.values() if None not in (r["boss"], r["original"])]
            doc = dict(alpha=alpha, phase=phase, raw=raw["sha256"], score=score, rows=rows, pairs=pairs,
                       mean_gain=fmean(gains) if len(gains) == len(rows) else None)
            # An unanswered reading is retried by the next run instead of being kept as a result.
            if score is not None:
                atomic_json(path, doc)
            print(name, phase, alpha, None if score is None else round(100 * score, 2), flush=True)
            return doc

        trials = {float.fromhex(p.stem): None for p in (base / "search").glob("*.json")}
        # Saved renders need no GPU, so their strengths are read side by side.
        with ThreadPoolExecutor(4) as side:
            for alpha, doc in zip(sorted(trials), side.map(lambda a: measure(a, "search"), sorted(trials))):
                trials[alpha] = doc["score"]
        for _ in range(2):
            measured = sorted(a for a, v in trials.items() if v is not None)
            if not measured:
                break
            best = max(measured, key=lambda a: (trials[a], -a))
            index = measured.index(best)
            fresh = [(best + other) / 2 for other in
                     (measured[index - 1] if index else None, measured[index + 1] if index + 1 < len(measured) else None)
                     if other is not None and abs(other - best) > STEP]
            if not fresh:
                break
            for alpha in fresh:
                trials[alpha] = measure(alpha, "search")["score"]
        measured = [a for a, v in trials.items() if v is not None]
        earlier = read(base / "result.json")
        result = dict(name=name, raw=raw, step=step, version=ws.VERSION, drawing_model=reader.model,
                      trials=sorted(trials.items()), previous_alpha=earlier.get("alpha"),
                      previous_score=earlier.get("score"))
        if measured:
            alpha = max(measured, key=lambda a: (trials[a], -a))
            ranking = measure(alpha, "ranking")
            result.update(alpha=alpha, search_score=trials[alpha], score=ranking["score"],
                          mean_gain=ranking["mean_gain"])
        else:
            result.update(alpha=None, score=None)
        atomic_json(out / "result.json", result)
        return result
    finally:
        rt.mixer.close()


rt = open_runtime()
try:
    with ThreadPoolExecutor(24) as pool:
        for name in sys.argv[1:]:
            result = search(rt, name, pool)
            print("Searched", name, result.get("alpha"), result.get("score"), flush=True)
finally:
    rt.close()
