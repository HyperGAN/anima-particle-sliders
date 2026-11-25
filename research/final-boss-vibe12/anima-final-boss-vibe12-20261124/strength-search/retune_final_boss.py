"""Search the Anima native alpha range using the published Final Boss score."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
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


BASELINE = ("constant", .6375)
GRIDS = {
    "constant": (.35, .50, .6375, .80, 1.00, 1.20, 1.45),
    "early": (.25, .40, .55, 2 / 3, .85, 1.05, 1.25),
    "late": (.35, .55, .75, .95, 1.15, 1.35, 1.55),
}
REFINEMENTS = 2
FINALISTS_PER_PROFILE = 2
MAX_VALIDATIONS_PER_PROFILE = 4
# The existing Jev anchor is a credible elite warrior or rival at .50.
MINIMUM_BOSS = .50


def summarize(rows):
    if any(row["integrity"] == "fail" for row in rows.values()):
        return {"status": "fail", "means": None}
    if any(row["integrity"] != "pass" or row[SCORE] is None for row in rows.values()):
        return {"status": "unmeasured", "means": None}
    return {"status": "pass", "means": {key: fmean(row[key] for row in rows.values())
                                        for key in (BOSS, GAIN, DAMAGE, SCORE)}}


def promote(trial, baseline):
    return (trial["status"] == "pass" and baseline["status"] == "pass"
            and trial["means"][BOSS] >= MINIMUM_BOSS
            and trial["means"][SCORE] > baseline["means"][SCORE])


def run(method):
    harness.ROOT = ROOT
    import boss_review
    import image_breakage
    import tune
    from review_inconclusive import reviewers

    boss_review.FOLDER = ROOT / "boss-character-review"
    image_breakage.CACHE = ROOT / "image-breakage-review/cases"
    tune.BREAKAGE_RULE = image_breakage.RULE = PREVIOUS / "image-breakage-rule.json"
    followup, decide, _, _, _ = reviewers()
    from final_boss_outcome import observer
    from particle_sliders.alpha_search import AlphaObservation, search_alpha
    from particle_sliders.alpha_tune import _load_definition, _render, _renderer
    from particle_sliders.strength_schedule import interval_strengths

    _, integrity = observer()
    bridge = harness.PRIOR / "prompt-only-alpha-tune/renderer-installed"
    sys.path.insert(0, str(bridge))
    base_path = WORK / f"{method}.json"
    definition, body, raw, cases, _ = _load_definition(base_path)
    result = harness.read(ROOT / "cohorts" / method / "result.json")
    fixed_path = ROOT / "tuning" / method / "boss-effect-tuning.json"
    fixed = harness.read(fixed_path)
    history = harness.read(SOURCE / "docs/final-boss-raw192/results.json")
    search_cases = fixed["screen_cases"]
    board_cases = fixed["development_cases"]
    if (result["step"] != 192 or result["ema_decay"] is not None or result["raw"] != fixed["raw"]
            or result["raw"]["sha256"] != raw.sha256 or history["sigmas"] != fixed["sampler"]
            or set(cases) != {*search_cases, *board_cases}):
        raise ValueError("This search requires the exact RAW192 leaderboard checkpoint and cases")
    target = WORK / "leaderboard-retune" / method
    config = {"definition": asdict(definition), "leaderboard": harness.pin(fixed_path),
              "implementation": harness.pin(Path(__file__).resolve()),
              "scoring_version": VERSION, "baseline": list(BASELINE),
              "minimum_mean_boss": MINIMUM_BOSS, "grids": {key: list(grid) for key, grid in GRIDS.items()},
              "refinements": REFINEMENTS, "finalists_per_profile": FINALISTS_PER_PROFILE,
              "max_validations_per_profile": MAX_VALIDATIONS_PER_PROFILE,
              "search_cases": search_cases, "board_cases": board_cases}
    request = target / "request.json"
    if request.is_file() and harness.read(request) != config:
        raise ValueError("The immutable search request changed")
    if (target / "result.json").is_file():
        return
    harness.write(request, config)

    archive_names = [method]
    if method == "shared-best-seed-split":
        archive_names.append(f"{method}-late")
    archives = []
    for name in archive_names:
        archive = WORK / f"{name}.alpha-tune"
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
            if (profile, alpha) == BASELINE:
                groups = (*fixed["screen"]["constant"].values(), fixed["matched_constant"])
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
            intact = integrity(on)
            if intact["status"] == "unmeasured":
                intact = followup(on)
            ratings = {BOSS: boss_review.assess(off, on, decide), ORIGINAL: boss_review.assess(off, off, decide),
                       DAMAGE: image_breakage.assess(off, on, decide)}
            boss, original = ratings[BOSS], ratings[ORIGINAL]
            return ratings | {SCORE: score_case(ratings), GAIN: None if None in (boss, original) else max(0, boss - original),
                              "integrity": intact["status"], "checks": intact, "off": off, "on": on}

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
            pending = {case: pool.submit(judge, *measure(case, folder / case / "image.png", schedule, profile, alpha))
                       for case in panel}
            rows = {case: future.result() for case, future in pending.items()}
            report = {"setting": binding, "rows": rows} | summarize(rows)
            harness.write(receipt, report)
            print(phase, profile, round(alpha, 5), report["status"], report["means"], flush=True)
            return report

        def observations(profile, value):
            measured = evaluate(profile, value, search_cases, "search")
            return {case: AlphaObservation(0, row[SCORE], row["integrity"]) for case, row in measured["rows"].items()}

        baseline = evaluate(*BASELINE, board_cases, "validation")
        official = next(row for row in harness.read(STORAGE / "leaderboard.json")["rankings"]["matched_constant"]
                        if row["method"] == method)
        if baseline["status"] != "pass" or baseline["means"][SCORE] != official["means"][SCORE]:
            raise ValueError("The alpha-search baseline did not replay the frozen leaderboard score")
        finalists = {}
        for profile, alphas in GRIDS.items():
            search = search_alpha(checkpoint=raw, case_ids=search_cases, alphas=alphas,
                                  evaluate=lambda alpha: observations(profile, alpha), refinement_rounds=REFINEMENTS)
            ranking = sorted((trial for trial in search.trials if trial.status == "pass"
                              and (profile, trial.alpha) != BASELINE), key=lambda trial: trial.mean_gain, reverse=True)
            finalists[profile] = ranking[:MAX_VALIDATIONS_PER_PROFILE]
        validations = []
        for profile, ranking in finalists.items():
            verified = []
            for trial in ranking:
                if len(verified) >= FINALISTS_PER_PROFILE and any(row["status"] == "pass" for row in verified):
                    break
                verified.append(evaluate(profile, trial.alpha, board_cases, "validation"))
            validations.extend(verified)
        passing = sorted((row for row in validations if row["status"] == "pass"),
                         key=lambda row: row["means"][SCORE], reverse=True)
        selected = next((row for row in passing if promote(row, baseline)), None)
        raw.verify()
        final = {"method": method, "scoring_version": VERSION, "checkpoint": asdict(raw), "sampler": sampler,
                 "baseline": baseline, "validation": validations, "best_measured": passing[0] if passing else None,
                 "selected": selected, "outcome": "improved" if selected else "keep-baseline"}
        harness.write(target / "result.json", final)


if __name__ == "__main__":
    _, _, method = sys.argv[1:]
    target = WORK / "leaderboard-retune" / method
    target.mkdir(parents=True, exist_ok=True)
    with (target / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(method)
