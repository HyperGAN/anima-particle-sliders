"""Train a character-vibe slider from its phrase alone.

    regime.py "pirate captain"

A language model writes image prompts for Anima. The raw vibe judge scores the renders against
the submitted phrase. Resumes use the exact prompts saved at setup.

1. Teach on the fixed scene grid, three portraits in every five scenes.
2. Rank three treatments per scene by the raw vibe score and teach on the strongest
   16 distinct scenes with usable subject masks.
3. Build late-cues targets from the kept pairs and train stock EMPF from scratch for 448
   updates. Nothing is rendered or judged while it trains.
4. Search one constant alpha for the finished slider on four characters.
5. Save the slider and a sheet of samples at that alpha.

With ``ANIMA_REGIME_MEASURE=1`` the run also ranks the setting on eight other characters,
confirms it on the sixteen held-back grid scenes and runs the structural check on both.
Neither panel chooses anything.

Rerun the same command to resume. Rendered images and judge answers are kept and reused.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from statistics import fmean
import os
import shutil
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
from lumen_studio.study.training import arm_path, train  # noqa: E402

from campaign import PARENT_STUDY, ROOT, open_runtime  # noqa: E402
import library  # noqa: E402
from prompt_writer import load_spec  # noqa: E402
import questions  # noqa: E402
from teacher_selection import LIMIT, MIN_VIBE_SCORE, has_vibe_score, select_teachers, subject_matte  # noqa: E402

CV_PYTHON = "/mnt/nvme/ai/conda/anaconda3/bin/python3"
RATES = dict(g=0.00075, d=0.009, particle=0.03)
# 448 updates is the earliest point at which the first vibe's slider was good: all eight ranking
# characters worked, and it scored as well as any later checkpoint at its own alpha. Training on
# lowers the alpha the slider needs. A longer budget continues the same run from its last update.
UPDATES, HORIZON = int(os.environ.get("ANIMA_REGIME_UPDATES", 448)), 1024
MEASURE = os.environ.get("ANIMA_REGIME_MEASURE") == "1"
GRID, STEP = [0.75 + k / 4 for k in range(6)], 1 / 16
vibe = " ".join(sys.argv[1:]).strip()
assert vibe, "Name the vibe"
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
    """Collect scored rounds until the strongest distinct scenes fill the training set."""
    path = ROOT / "teachers.json"
    if path.exists():
        return read(path)
    progress = ROOT / "teacher-rounds.json"
    legacy = ROOT / "teachers-incomplete.json"
    earlier = read(progress) if progress.exists() else read(legacy) if legacy.exists() else None
    if earlier and earlier["vibe"] != vibe:
        raise ValueError("Saved teacher rounds belong to another phrase")
    scenes = {scene["id"]: scene for scene in spec["teaching"]}
    rounds = earlier["record"] if earlier else []

    def matte_for(ref):
        return subject_matte(ROOT, ref, CV_PYTHON, CODE.parent / "post-training" / "matte.py")

    def receipt(kept):
        return dict(vibe=vibe, kept=kept, record=rounds,
                    selection=dict(metric="boss", minimum_score=MIN_VIBE_SCORE, limit=LIMIT))

    def integrity(ref):
        image = FileRecord.from_path(Path(ref["path"]))
        if image.sha256 != ref["sha256"]:
            raise ValueError("A teacher image changed after it was rendered")
        return out.integrity.check(image)["status"]

    kept = select_teachers(rounds, scenes, matte_for)
    if earlier:
        atomic_json(progress, receipt(kept))
    attempt = max((r["attempt"] for r in rounds), default=-1) + 1
    while len(kept) < LIMIT:
        rendered = []
        for scene in spec["teaching"]:
            if scene["id"] in kept:
                continue
            seed = scene["seed"] + 104729 * attempt
            neutral = asdict(out.image(scene["neutral"], seed))
            options = [dict(t, image=asdict(out.image(t["positive"], seed))) for t in scene["treatments"]]
            rendered.append((scene, seed, neutral, options))
        jobs = []
        for _, _, neutral, options in rendered:
            baseline_structure = pool.submit(integrity, neutral)
            for option in options:
                image = option["image"]
                jobs.append((option, pool.submit(measure, neutral, image),
                             baseline_structure, pool.submit(integrity, image)))
        for option, reading, baseline, candidate in jobs:
            row = reading.result()
            option.update({k: v for k, v in row.items() if k != "on"},
                          baseline_structure=baseline.result(), candidate_structure=candidate.result())
        for scene, seed, neutral, options in rendered:
            rounds.append(dict(scene=scene["id"], framing=scene["framing"], attempt=attempt, seed=seed,
                               neutral=neutral, options=options))
        kept = select_teachers(rounds, scenes, matte_for)
        atomic_json(progress, receipt(kept))
        if not any(has_vibe_score(o) for _, _, _, options in rendered for o in options):
            raise RuntimeError("No teacher image received a raw vibe score in this round")
        print(f"Teacher round {attempt+1}: {len(kept)}/{LIMIT} scenes selected", flush=True)
        attempt += 1
    result = receipt(kept)
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


class Unchecked:
    """Stands in for measured outputs while training, so no image is rendered or judged."""

    def __init__(self, root, runtime):
        self.root, self.runtime = root, runtime

    def compare(self, cases, variation, raw, alpha, profile, *, owner=None):
        return dict(raw=asdict(raw), status="pass", mean_gain=None, pairs=[], alpha=alpha, profile=profile)


def contact_sheet(pairs, path, width=256, columns=4):
    """Original above slider ON, for a quick look."""
    from PIL import Image

    rows = (len(pairs) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, 2 * width * rows), (24, 25, 28))
    for index, (off, on) in enumerate(pairs):
        x, y = (index % columns) * width, (index // columns) * 2 * width
        for image, dy in ((off, 0), (on, width)):
            with Image.open(image["path"]) as picture_:
                sheet.paste(picture_.convert("RGB").resize((width, width), Image.Resampling.LANCZOS), (x, y + dy))
    sheet.save(path, quality=88)


spec = load_spec(ROOT, vibe)
asked = questions.build(vibe, version=spec.get("phrase_version", library.WARDROBE_VERSION))
question_path = ROOT / "teacher-question.json"
if not question_path.exists():
    atomic_json(question_path, asked["effect"])
elif read(question_path) != asked["effect"]:
    raise ValueError("The raw vibe scoring question changed under this run")

rt = open_runtime(checkpointing=True)
try:
    out = Outputs(ROOT, rt, vibe, asked["rubric"])
    board = read(PARENT_STUDY / "cohort-retest/panels.json")
    panel = dict(search=board["panels"]["search"], ranking=board["panels"]["ranking"],
                 cases={c["id"]: dict(c["request"], off=c["off"]) for c in board["cases"]})
    ranking_cases = {c: panel["cases"][c] for c in panel["ranking"]}
    with ThreadPoolExecutor(24) as pool:
        if os.environ.get("ANIMA_REGIME_CURVE"):
            # Measure saved checkpoints of a finished run: one alpha search each, then the ranking panel.
            # This chooses nothing. It answers how few updates already score well.
            for step in map(int, os.environ["ANIMA_REGIME_CURVE"].split(",")):
                done = ROOT / "curve" / f"{step}.json"
                if done.exists():
                    continue
                raw = read(arm_path(ROOT, RATES) / "fit" / f"{step}.json")["raw"]
                with loaded(rt, out, step, raw) as render:
                    found = search(render, pool, step, raw, panel)
                    ranked = found["alpha"] and panel_score(
                        pool, {c: v["off"] for c, v in ranking_cases.items()}, render(ranking_cases, found["alpha"]))
                atomic_json(done, dict(step=step, alpha=found["alpha"], search=found["search"], trials=found["trials"],
                                       ranking=ranked and ranked["score"], working=ranked and ranked["working"],
                                       rows=ranked and ranked["rows"]))
                print(vibe, "update", step, "alpha", found["alpha"], "search", found["search"] and round(100 * found["search"], 2),
                      "ranking", ranked and ranked["score"] and round(100 * ranked["score"], 2),
                      "working", ranked and ranked["working"], flush=True)
            raise SystemExit(0)
        teachers = collect(out, spec, pool)
        metric = teachers.get("selection", {}).get("metric", "case")
        print(vibe, "teaches", len(teachers["kept"]), "of", len(spec["teaching"]), "scenes, mean",
              "vibe score" if metric == "boss" else "combined score",
              round(100 * fmean(t[metric] for t in teachers["kept"].values()), 1), flush=True)
        cache = add_boss_scenes(rt, None, ROOT / "targets", teachers["kept"],
                                variation="vibe-" + digest(dict(vibe=vibe, kept=sorted(teachers["kept"])))[:16])
        train(Unchecked(ROOT, rt), cache, dict(steps=[]), RATES, UPDATES, "fit", schedule_horizon=HORIZON)
        raw = read(arm_path(ROOT, RATES) / "fit" / f"{UPDATES}.json")["raw"]
        with loaded(rt, out, UPDATES, raw) as render:
            found = search(render, pool, UPDATES, raw, panel)
            alpha = found["alpha"]
            if alpha is None:
                raise SystemExit("The alpha search has no measured strength; rerun to retry the judge")
            # The slider and a look at it: the four search characters and the eight others, OFF and ON.
            shown = {c: panel["cases"][c] for c in (*panel["search"], *panel["ranking"])}
            on = render(shown, alpha)
            measured = {}
            if MEASURE:
                ranking = panel_score(pool, {c: v["off"] for c, v in ranking_cases.items()}, {c: on[c] for c in ranking_cases})
                held = {s["id"]: dict(prompt=s["neutral"], seed=s["seed"], framing=s["framing"]) for s in spec["confirmation"]}
                off = {c: asdict(out.image(v["prompt"], v["seed"])) for c, v in held.items()}
                confirmation = panel_score(pool, off, render(held, alpha))
                measured = dict(
                    ranking=dict(ranking, off={c: v["off"] for c, v in ranking_cases.items()}),
                    confirmation=dict(confirmation, off=off, framing={c: v["framing"] for c, v in held.items()}),
                    structure={name: {c: out.integrity.check(FileRecord.from_path(Path(r["on"]["path"])))["status"]
                                      for c, r in doc["rows"].items()}
                               for name, doc in (("ranking", ranking), ("confirmation", confirmation))})
        slider = ROOT / "slider"
        slider.mkdir(exist_ok=True)
        name = vibe.replace(" ", "-")
        for suffix in (".safetensors", ".engine.json", ".scoped.json"):
            # The exported RAW with the two sidecars that identify it.
            if not (slider / (name + suffix)).exists():
                shutil.copyfile(Path(raw["path"]).with_suffix(suffix), slider / (name + suffix))
        contact_sheet([(shown[c]["off"], on[c]) for c in shown], slider / "samples.jpg")
        result = dict(
            vibe=vibe, version=ws.VERSION, teachers=len(teachers["kept"]), updates=UPDATES, step=UPDATES, raw=raw,
            alpha=alpha, search=found["search"], trials=found["trials"], slider=str(slider / (name + ".safetensors")),
            samples=dict(sheet=str(slider / "samples.jpg"), off={c: v["off"] for c, v in shown.items()}, on=on), **measured)
        atomic_json(ROOT / ("result.json" if UPDATES == 448 else f"result-{UPDATES}.json"), result)
        atomic_json(slider / "use.json", dict(vibe=vibe, alpha=alpha, updates=UPDATES, sampler="ten Euler steps, CFG 1",
                                              strength="one constant alpha at every step", raw_sha256=raw["sha256"]))
        show = lambda doc: None if doc["score"] is None else round(100 * doc["score"], 2)  # noqa: E731
        print(vibe, "slider", slider / (name + ".safetensors"), "alpha", alpha, "search", round(100 * found["search"], 2),
              *(["ranking", show(measured["ranking"]), f"working {measured['ranking']['working']}/8", "confirmation",
                 show(measured["confirmation"]), f"working {measured['confirmation']['working']}/16"] if MEASURE else []),
              flush=True)
finally:
    rt.close()
