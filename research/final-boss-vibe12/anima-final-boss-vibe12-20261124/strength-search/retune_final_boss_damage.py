"""Tune a saved Anima adapter on character effect and graded image damage."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
from pathlib import Path
from statistics import fmean
import sys

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parent))

from control import CACHE, PREVIOUS, ROOT, SOURCE, STORAGE
import optimization_worker as harness
from publish import BOSS, DAMAGE, GAIN, ORIGINAL, SCORE, VERSION, score_case
import adaptive_alpha


BASELINE = ("constant", .6375)
PROFILES = ("constant", "early", "late")
FIXED_EARLY = ("early", 2 / 3)
METRICS = (BOSS, ORIGINAL, DAMAGE, GAIN, SCORE)
RUN = "leaderboard-retune-damage"


def summary(setting, rows):
    if list(rows) != setting["cases"] or not rows:
        raise ValueError("Incomplete or mismatched panel")
    for row in rows.values():
        boss, original = row[BOSS], row[ORIGINAL]
        gain = None if None in (boss, original) else max(0, boss - original)
        if row[SCORE] != score_case(row) or row[GAIN] != gain:
            raise ValueError("Saved Jev scores differ from the scoring rule")
    complete = all(row[key] is not None for row in rows.values() for key in METRICS)
    means = {key: fmean(row[key] for row in rows.values()) for key in METRICS} if complete else None
    return {"setting": setting, "rows": rows, "status": "measured" if complete else "unmeasured", "means": means}


def objective(trial, metric=SCORE):
    return trial["means"][metric] if trial["means"] is not None else None


def slider_definition(method, result):
    path = WORK / f"{method}.json"
    if path.is_file():
        return path
    template = harness.read(WORK / "shared-best-stock.json")
    checkpoint = result["raw"]
    template["checkpoint"] = checkpoint
    harness.write(path, template)
    return path


def run(method):
    harness.ROOT = ROOT
    import boss_review
    import image_breakage
    import tune

    boss_review.FOLDER = ROOT / "boss-character-review"
    image_breakage.CACHE = ROOT / "image-breakage-review/cases"
    tune.BREAKAGE_RULE = image_breakage.RULE = PREVIOUS / "image-breakage-rule.json"
    _, decide = tune.reviewers()
    from particle_sliders.alpha_tune import _load_definition, _render, _renderer
    from particle_sliders.strength_schedule import interval_strengths

    bridge = harness.PRIOR / "prompt-only-alpha-tune/renderer-installed"
    sys.path.insert(0, str(bridge))
    result = harness.read(ROOT / "cohorts" / method / "result.json")
    definition, body, raw, cases, _ = _load_definition(slider_definition(method, result))
    fixed_path = ROOT / "tuning" / method / "boss-effect-tuning.json"
    fixed = harness.read(fixed_path)
    history = harness.read(SOURCE / "docs/final-boss-raw192/results.json")
    search_cases = fixed["screen_cases"]
    board_cases = fixed["development_cases"]
    if (result["step"] != 192 or result["ema_decay"] is not None or result["raw"] != fixed["raw"]
            or result["raw"]["sha256"] != raw.sha256 or history["sigmas"] != fixed["sampler"]
            or set(cases) != {*search_cases, *board_cases}):
        raise ValueError("This search requires the exact RAW192 leaderboard checkpoint and cases")
    target = WORK / RUN / method
    config = {"definition": asdict(definition), "leaderboard": harness.pin(fixed_path),
              "implementation": harness.pin(Path(__file__).resolve()),
              "scoring_version": VERSION, "baseline": list(BASELINE),
              "start_alpha": adaptive_alpha.START, "initial_step": adaptive_alpha.INITIAL_STEP,
              "bounds": [adaptive_alpha.LOWER, adaptive_alpha.UPPER], "minimum_step": adaptive_alpha.MIN_STEP,
              "trial_budget_per_profile": adaptive_alpha.TRIAL_BUDGET, "profiles": list(PROFILES),
              "search_objective": SCORE, "quality_metric": DAMAGE, "fixed_early": list(FIXED_EARLY),
              "search_implementation": harness.pin(Path(adaptive_alpha.__file__)),
              "search_cases": search_cases, "board_cases": board_cases}
    request = target / "request.json"
    if request.is_file() and harness.read(request) != config:
        raise ValueError("The immutable search request changed")
    if (target / "result.json").is_file():
        return
    harness.write(request, config)

    priors = []
    for previous_run in ("leaderboard-retune-adaptive", "leaderboard-retune-centered", "leaderboard-retune"):
        prior = WORK / previous_run / method
        if (prior / "request.json").is_file():
            original = harness.read(prior / "request.json")
            if any(original[key] != config[key] for key in ("definition", "leaderboard", "scoring_version", "search_cases", "board_cases")):
                raise ValueError("Cached scores belong to a different RAW evaluation")
            priors.append(prior)

    archive_names = [method]
    if method == "shared-best-seed-split":
        archive_names.append(f"{method}-late")
    archives = []
    for name in archive_names:
        archive = WORK / f"{name}.alpha-tune"
        if not (archive / "request.json").is_file():
            continue
        binding = harness.read(archive / "request.json")
        current, _, checkpoint, _, _ = _load_definition(WORK / f"{name}.json")
        if binding != asdict(current) or checkpoint != raw:
            raise ValueError("Cached native renders belong to a different definition or adapter")
        archives.append(archive)

    with _renderer(body["renderer"])(raw, body["renderer_options"]) as renderer, ThreadPoolExecutor(5) as pool:
        sampler = renderer.sampler
        if (sampler["sigmas"] != fixed["sampler"]
                or sampler["model"] != harness.read(CACHE / "index.json")["identity"]["model"]
                or any(harness.read(path / "sampler.json") != sampler for path in archives)):
            raise ValueError("The native sampler changed")

        def measure(case, path, schedule, profile, alpha):
            row = cases[case]
            off = harness.pin(Path(row["off"].path))
            if off["sha256"] != row["off"].sha256:
                raise ValueError("Reference image changed")
            key = Path(profile) / alpha.hex() / case
            candidates = [parent / "renders" / key / "image.png" for parent in archives]
            candidates.extend(prior / "renders" / key / "image.png" for prior in priors)
            if (profile, alpha) in (BASELINE, FIXED_EARLY):
                groups = (*fixed["screen"][profile].values(), fixed["matched_constant"] if profile == "constant" else fixed["fixed_strength"])
                candidates.extend(Path(group["cases"][case]["on"]["path"]) for group in groups
                                  if group["alpha"] == alpha and case in group["cases"])
            on_path = next((item for item in candidates if item.with_name("render.json").is_file()), path)
            if on_path == path:
                _render(renderer, row, schedule, path)
            on = harness.pin(on_path)
            native = harness.read(on_path.with_name("render.json"))
            if (native["image"] != on or native["gains"] != schedule or native["model"] != sampler["model"]
                    or (native["prompt"], native["seed"]) != (row["request"]["prompt"], row["request"]["seed"])):
                raise ValueError(f"Render evidence differs for {method}/{case}")
            return off, on

        def judge(off, on):
            ratings = {BOSS: boss_review.assess(off, on, decide), ORIGINAL: boss_review.assess(off, off, decide),
                       DAMAGE: image_breakage.assess(off, on, decide)}
            boss, original = ratings[BOSS], ratings[ORIGINAL]
            return ratings | {SCORE: score_case(ratings), GAIN: None if None in (boss, original) else max(0, boss - original),
                              "off": off, "on": on}

        def evaluate(profile, alpha, panel, phase):
            schedule = list(interval_strengths(alpha, renderer.interval_widths, profile=profile))
            folder = target / "renders" / profile / alpha.hex()
            receipt = folder / f"{phase}.json"
            binding = {"profile": profile, "alpha": alpha, "effective_strengths": schedule, "cases": panel}
            if receipt.is_file():
                previous = harness.read(receipt)
                if previous["setting"] != binding:
                    raise ValueError("Cached scoring used different inputs")
                return previous
            for prior in priors:
                saved = prior / "renders" / profile / alpha.hex() / receipt.name
                if saved.is_file():
                    previous = harness.read(saved)
                    if previous["setting"] != binding:
                        raise ValueError("Previous scoring used different inputs")
                    rows = {case: {key: row[key] for key in (*METRICS, "off", "on")}
                            for case, row in previous["rows"].items()}
                    report = summary(binding, rows)
                    if report["means"] is not None:
                        for case, row in rows.items():
                            if row["off"]["sha256"] != cases[case]["off"].sha256:
                                raise ValueError("Cached judgment has a different original")
                            on = row["on"]
                            image_path = Path(on["path"])
                            native = harness.read(image_path.with_name("render.json"))
                            if (on != harness.pin(image_path) or native["image"] != on or native["gains"] != schedule
                                    or native["model"] != sampler["model"]
                                    or (native["prompt"], native["seed"]) != (cases[case]["request"]["prompt"], cases[case]["request"]["seed"])):
                                raise ValueError("Cached judgment has a different rendered image")
                        report["prior_scores"] = harness.pin(saved)
                        harness.write(receipt, report)
                        print(phase, profile, round(alpha, 5), "reused", flush=True)
                        return report
            pending = {case: pool.submit(judge, *measure(case, folder / case / "image.png", schedule, profile, alpha))
                       for case in panel}
            rows = {case: future.result() for case, future in pending.items()}
            report = summary(binding, rows)
            harness.write(receipt, report)
            print(phase, profile, round(alpha, 5), report["status"], report["means"], flush=True)
            return report

        def observations(profile, value):
            measured = evaluate(profile, value, search_cases, "search")
            return objective(measured)

        evaluate("constant", adaptive_alpha.START, search_cases, "search")
        baseline = evaluate(*BASELINE, board_cases, "validation")
        fixed_early = evaluate(*FIXED_EARLY, board_cases, "validation")
        ordinary = harness.read(STORAGE / "leaderboard.json")["rankings"]["matched_constant"]
        official = next((row for row in ordinary if row["method"] == method), None)
        if official and objective(baseline) != official["means"][SCORE]:
            raise ValueError("The alpha-search baseline did not replay the frozen leaderboard score")
        finalists, paths = {}, {}
        for profile in PROFILES:
            trials, steps = adaptive_alpha.search(lambda alpha: observations(profile, alpha))
            paths[profile] = {"trials": [{"alpha": alpha, "score": score} for alpha, score in trials.items()], "steps": steps}
            ranking = sorted((alpha for alpha, score in trials.items() if score is not None and (profile, alpha) != BASELINE),
                             key=trials.__getitem__, reverse=True)
            if ranking:
                most_effect = max(ranking, key=lambda alpha: objective(evaluate(profile, alpha, search_cases, "search"), GAIN))
                finalists[profile] = tuple(dict.fromkeys((*ranking[:2], most_effect)))
            harness.write(target / "search-path.json", paths)
        validations = []
        for profile, ranking in finalists.items():
            validations.extend(evaluate(profile, alpha, board_cases, "validation") for alpha in ranking)
        ranked = [trial for trial in (baseline, fixed_early, *validations) if objective(trial) is not None]
        best_score = max(ranked, key=objective, default=None)
        best_effect = max(ranked, key=lambda trial: objective(trial, GAIN), default=None)
        raw.verify()
        final = {"method": method, "scoring_version": VERSION, "quality_metric": DAMAGE,
                 "checkpoint": asdict(raw), "sampler": sampler, "baseline": baseline, "fixed_early": fixed_early,
                 "validation": validations, "highest_effect": best_effect, "highest_score": best_score,
                 "selected": best_score, "outcome": "selected" if best_score else "unmeasured"}
        harness.write(target / "result.json", final)


if __name__ == "__main__":
    _, _, method = sys.argv[1:]
    target = WORK / RUN / method
    target.mkdir(parents=True, exist_ok=True)
    with (target / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(method)
