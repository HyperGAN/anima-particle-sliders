"""Train and confirm a character-vibe slider from its phrase alone.

    regime.py "pirate captain"

Every choice below is fixed in advance or made by a measured score. No prompt is written
for the vibe by a person or by a language model.

1. Teach on the fixed scene grid, three portraits in every five scenes.
2. Render three fixed treatments of each scene and keep the one that scores highest the way
   a slider's output is scored: the vibe is there, the drawing is one whole finished figure,
   and the scene kept its colors. A scene with no treatment worth teaching is left out.
3. Build late-cues targets from the kept pairs and train stock EMPF from scratch.
4. For each late checkpoint that passed its structural checks, search one constant alpha on
   four characters. Keep the checkpoint and alpha with the highest search score.
5. Rank that setting on eight other characters, then confirm it on the sixteen grid scenes
   that were held back. Neither panel chose anything.

Rerun the same command to resume. Rendered images and judge answers are kept and reused.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from statistics import fmean
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(CODE), str(CODE.parent / "post-training")]

from particle_sliders.alpha_tune_jev import _Image  # noqa: E402
from particle_sliders.candidate_failure import FileRecord  # noqa: E402

from lumen_studio.contracts import atomic_json, digest  # noqa: E402
from lumen_studio.particles import ParticleAdapter  # noqa: E402
from lumen_studio.study import working_score as ws  # noqa: E402
from lumen_studio.study.boss_targets import add_boss_scenes  # noqa: E402
from lumen_studio.study.design_score import summarize  # noqa: E402
from lumen_studio.study.judge import StudyJudge  # noqa: E402
from lumen_studio.study.outputs import Outputs, read  # noqa: E402
from lumen_studio.study.teachers import validation_panels  # noqa: E402
from lumen_studio.study.training import Catastrophe, arm_path, train  # noqa: E402

from campaign import PARENT_STUDY, ROOT, open_runtime  # noqa: E402
import library  # noqa: E402
import questions  # noqa: E402

CV_PYTHON = "/mnt/nvme/ai/conda/anaconda3/bin/python3"
RATES = dict(g=0.00075, d=0.009, particle=0.03)
UPDATES, HORIZON, LATE = 512, 1024, (256, 320, 384, 448, 512)
CANDIDATES = 3  # The last passing checkpoints that get an alpha search.
TEACH_AT_LEAST, WORTH_TEACHING = 16, 0.35
GRID, STEP = [0.75 + k / 4 for k in range(6)], 1 / 16
vibe = " ".join(sys.argv[1:]).strip()
assert vibe, "Name the vibe"
asked = questions.build(vibe)
(ROOT / "readings").mkdir(parents=True, exist_ok=True)
judge = StudyJudge(ROOT / "readings", vibe)
reader = ws.DrawingReader(ROOT / "readings" / "drawing")


def picture(ref):
    return _Image.read(FileRecord.from_path(Path(ref["path"])))


def answer(images, question, key):
    values = judge._decide(tuple(images), question["state"], question["questions"])
    return None if values is None else values[key]


def measure(off, on):
    """One image, read the way the board reads a slider's output."""
    a, b = picture(off), picture(on)
    paired = judge._decide((a, b), asked["pair"]["state"], asked["pair"]["questions"])
    original = answer((a, a), asked["effect"], "effect")
    row = dict(
        on=on,
        # An unanswered OFF reading takes the floor the score applies to it anyway.
        original=ws.BOSS_FLOOR if original is None else original,
        boss=answer((a, b), asked["effect"], "effect"),
        broken=reader.broken(on["path"]),
        off_broken=reader.broken(off["path"]),
        one_figure=paired and paired["one_figure"],
        same_person=paired and paired["same_person"],
        same_scene=paired and paired["same_scene"],
        figure=answer((b,), asked["figure"], "figure"),
        recolored=answer((a, b), asked["recolor"], "recolored"),
    )
    row["quality"] = ws.quality(row["broken"], row["off_broken"], row["one_figure"], row["figure"])
    row["case"] = ws.score_case(row["boss"], row["original"], row["quality"], row["recolored"])
    return row


def repeat(left, right):
    x, y = picture(left), picture(right)
    values = [answer(order, asked["repeat"], "repeat") for order in ((x, y), (y, x))]
    return None if None in values else fmean(values)


def panel_score(pool, cases, on):
    """Score one setting on a panel. A few refused repeat pairs leave bounds instead of no score."""
    for _ in range(4):
        rows = {c: job.result() for c, job in {c: pool.submit(measure, cases[c], on[c]) for c in cases}.items()}
        pairs = {(a, b): pool.submit(repeat, on[a], on[b]) for a, b in combinations(cases, 2)}
        pairs = [dict(cases=[a, b], repeat=job.result()) for (a, b), job in pairs.items()]
        if all(r["case"] is not None for r in rows.values()):
            break
        time.sleep(30)
    values = {c: r["case"] for c, r in rows.items()}
    refused = [p["cases"] for p in pairs if p["repeat"] is None]
    score = bounds = None
    if None not in values.values():
        if not refused:
            score = summarize(values, pairs)["score"]
        elif len(refused) <= max(1, len(pairs) // 50):
            bounds = [summarize(values, [dict(p, repeat=fill if p["repeat"] is None else p["repeat"])
                                         for p in pairs])["score"] for fill in (1.0, 0.0)]
            score = fmean(bounds)
    working = sum(r["case"] is not None and r["quality"] >= 0.5 and r["recolored"] < 0.5
                  and ws.effect(r["boss"], r["original"]) >= 0.5 for r in rows.values())
    return dict(score=score, bounds=bounds, refused_pairs=refused, working=working, rows=rows, pairs=pairs)


def collect(out, spec, pool):
    """Keep, for each scene, the treatment that is most worth teaching."""
    path = ROOT / "teachers.json"
    if path.exists():
        return read(path)
    kept, record = {}, []
    for attempt in range(2):
        rendered = []
        for scene in spec["teaching"]:
            if scene["id"] in kept:
                continue
            seed = scene["seed"] + 104729 * attempt
            neutral = asdict(out.image(scene["neutral"], seed))
            options = [dict(t, image=asdict(out.image(t["positive"], seed))) for t in scene["treatments"]]
            rendered.append((scene, seed, neutral, options))
        jobs = [(scene, seed, neutral, o, pool.submit(measure, neutral, o["image"]),
                 pool.submit(answer, (picture(neutral), picture(o["image"])), asked["separate"], "separate"),
                 pool.submit(out.integrity.check, FileRecord.from_path(Path(o["image"]["path"]))))
                for scene, seed, neutral, options in rendered for o in options]
        for scene, seed, neutral, option, reading, separate, whole in jobs:
            row = reading.result()
            option.update({k: v for k, v in row.items() if k != "on"}, separate=separate.result(),
                          integrity=whole.result()["status"])
        for scene, seed, neutral, options in rendered:
            worth = [o for o in options if o["case"] is not None and o["case"] >= WORTH_TEACHING
                     and o["separate"] is not None and o["separate"] < 0.5 and o["integrity"] != "fail"]
            chosen = max(worth, key=lambda o: o["case"]) if worth else None
            record.append(dict(scene=scene["id"], framing=scene["framing"], attempt=attempt, seed=seed,
                               neutral=neutral, options=options, chosen=chosen and chosen["positive"]))
            if chosen is None:
                continue
            matte = ROOT / "mattes" / f"{scene['id']}.png"
            matte.parent.mkdir(exist_ok=True)
            cut = subprocess.run([CV_PYTHON, str(CODE.parent / "post-training" / "matte.py"), neutral["path"], str(matte)],
                                 capture_output=True, text=True)
            if cut.returncode:
                record[-1]["chosen"] = None  # No usable character region to weight the edit by.
                continue
            kept[scene["id"]] = dict(
                neutral=scene["neutral"], positive=chosen["positive"], explicit=chosen["explicit"], seed=seed,
                framing=scene["framing"], case=chosen["case"],
                neutral_image={k: neutral[k] for k in ("path", "sha256")},
                positive_image={k: chosen["image"][k] for k in ("path", "sha256")}, matte=str(matte))
    result = dict(vibe=vibe, kept=kept, record=record)
    if len(kept) < TEACH_AT_LEAST:
        atomic_json(ROOT / "teachers-incomplete.json", result)
        raise SystemExit(f"Only {len(kept)} of {len(spec['teaching'])} scenes have a treatment worth teaching")
    atomic_json(path, result)
    return result


@contextmanager
def loaded(rt, out, step, raw):
    """Install one saved RAW and hand back a function that renders a panel at a constant alpha."""
    adapter = ParticleAdapter(rt.transformer).to(rt.device)
    meta = adapter.load_export(raw["path"], model_identity=rt.identity)
    assert meta["step"] == step and meta["weights"] == "live"
    adapter.requires_grad_(False).eval()
    key = f"regime-{step}"
    rt.mixer.add(key, adapter)

    def render(cases, alpha):
        return {c: asdict(out.image(case["prompt"], case["seed"], raw=raw["sha256"], variation=key,
                                    gains=[alpha] * 10)) for c, case in cases.items()}

    try:
        yield render
    finally:
        rt.mixer.close()


def search(render, pool, step, raw, panel):
    """One constant alpha for this RAW, from a fixed grid and two refinements around the leader."""
    folder = ROOT / "search" / str(step)
    cases = {c: panel["cases"][c] for c in panel["search"]}

    def trial(alpha):
        path = folder / (float(alpha).hex() + ".json")
        if path.exists():
            return read(path)["score"]
        doc = panel_score(pool, {c: v["off"] for c, v in cases.items()}, render(cases, alpha))
        if doc["score"] is not None:
            atomic_json(path, dict(alpha=alpha, **doc))
        print(vibe, step, "search", alpha, None if doc["score"] is None else round(100 * doc["score"], 2), flush=True)
        return doc["score"]

    trials = {alpha: trial(alpha) for alpha in GRID}
    for _ in range(2):
        measured = sorted(a for a, v in trials.items() if v is not None)
        if not measured:
            break
        best = max(measured, key=lambda a: (trials[a], -a))
        at = measured.index(best)
        fresh = [(best + other) / 2 for other in
                 (measured[at - 1] if at else None, measured[at + 1] if at + 1 < len(measured) else None)
                 if other is not None and abs(other - best) > STEP]
        if not fresh:
            break
        trials.update({alpha: trial(alpha) for alpha in fresh})
    measured = [a for a, v in trials.items() if v is not None]
    alpha = max(measured, key=lambda a: (trials[a], -a)) if measured else None
    return dict(step=step, raw=raw, alpha=alpha, search=alpha and trials[alpha], trials=sorted(trials.items()))


rt = open_runtime(checkpointing=True)
try:
    spec = library.spec(vibe)
    if (ROOT / "spec.json").exists():
        assert read(ROOT / "spec.json") == spec, "The scene grid or the vibe changed under this run"
    else:
        atomic_json(ROOT / "spec.json", spec)
    out = Outputs(ROOT, rt, vibe, asked["rubric"])
    assert out.integrity.calibrate()["passed"]
    with ThreadPoolExecutor(24) as pool:
        teachers = collect(out, spec, pool)
        print(vibe, "teaches", len(teachers["kept"]), "of", len(spec["teaching"]), "scenes, mean worth",
              round(100 * fmean(t["case"] for t in teachers["kept"].values()), 1), flush=True)
        cache = add_boss_scenes(rt, None, ROOT / "targets", teachers["kept"],
                                variation="vibe-" + digest(dict(vibe=vibe, kept=sorted(teachers["kept"])))[:16])
        watched = [dict(prompt=s["neutral"], seed=s["seed"]) for s in spec["watched"]]
        panels = validation_panels(out, dict(steps=watched), 3)
        try:
            train(out, cache, panels, RATES, UPDATES, "fit", schedule_horizon=HORIZON,
                  observation_steps=[n for n in LATE if n % (UPDATES // 4)])
            stopped = None
        except Catastrophe:
            stopped = read(arm_path(ROOT, RATES) / "failure.json")["step"]
            print(vibe, "stopped on a structural failure at update", stopped, flush=True)
        board = read(PARENT_STUDY / "cohort-retest/panels.json")
        panel = dict(search=board["panels"]["search"], ranking=board["panels"]["ranking"],
                     cases={c["id"]: dict(c["request"], off=c["off"]) for c in board["cases"]})
        candidates = []
        for step in LATE:
            seen = arm_path(ROOT, RATES) / "fit" / f"{step}.json"
            # A checkpoint whose own check failed is never a candidate.
            if seen.exists() and read(seen)["status"] != "fail" and step != stopped:
                candidates.append((step, read(seen)["raw"]))
        searched = []
        for step, raw in candidates[-CANDIDATES:]:
            with loaded(rt, out, step, raw) as render:
                found = search(render, pool, step, raw, panel)
            if found["alpha"] is not None:
                searched.append(found)
        if not searched:
            raise SystemExit("No checkpoint could be searched")
        chosen = max(searched, key=lambda f: (f["search"], -f["alpha"]))
        alpha = chosen["alpha"]
        # The two panels below choose nothing.
        with loaded(rt, out, chosen["step"], chosen["raw"]) as render:
            ranking_cases = {c: panel["cases"][c] for c in panel["ranking"]}
            ranking = panel_score(pool, {c: v["off"] for c, v in ranking_cases.items()}, render(ranking_cases, alpha))
            held = {s["id"]: dict(prompt=s["neutral"], seed=s["seed"], framing=s["framing"]) for s in spec["confirmation"]}
            off = {c: asdict(out.image(v["prompt"], v["seed"])) for c, v in held.items()}
            confirmation = panel_score(pool, off, render(held, alpha))
        checks = {name: {c: out.integrity.check(FileRecord.from_path(Path(r["on"]["path"])))["status"]
                         for c, r in doc["rows"].items()}
                  for name, doc in (("ranking", ranking), ("confirmation", confirmation))}
        result = dict(
            vibe=vibe, version=ws.VERSION, teachers=len(teachers["kept"]), stopped=stopped,
            candidates=[dict(step=f["step"], alpha=f["alpha"], search=f["search"]) for f in searched],
            step=chosen["step"], raw=chosen["raw"], alpha=alpha, search=chosen["search"],
            ranking=dict(ranking, off={c: v["off"] for c, v in ranking_cases.items()}),
            confirmation=dict(confirmation, off=off, framing={c: v["framing"] for c, v in held.items()}),
            structure=checks)
        atomic_json(ROOT / "result.json", result)
        show = lambda doc: None if doc["score"] is None else round(100 * doc["score"], 2)  # noqa: E731
        print(vibe, "RAW", chosen["step"], "alpha", alpha, "search", round(100 * chosen["search"], 2),
              "ranking", show(ranking), f"working {ranking['working']}/8",
              "confirmation", show(confirmation), f"working {confirmation['working']}/16", flush=True)
finally:
    rt.close()
