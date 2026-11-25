"""Optimize matched Final Boss effect on the exposed development cases."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from boss_review import RUBRIC, augment
from image_breakage import RULE as BREAKAGE_RULE, assess as assess_breakage, with_readability
from optimization_targets import ARCHIVE, METHODS
from optimization_worker import BUDGET, PRIOR, PRODUCT, QUESTION, ROOT, SCREEN, attach_judge, gains, panel, pin, read, render, write
from score import (DISTINCTION, EFFECT, METRIC, OBJECTIVE, ORIGINAL_DISTINCTION, PRESERVATION, PRIOR_REPORTS, REPORT,
                   aggregate_total, case_effect, preservation, ranking, weighted_total)


GRID = {
    "early": (.35, .475, .60, 2 / 3, .80, .95),
    "constant": (.55, 2 / 3, .80, .95, 1.10, 1.35),
    "late": (.45, 2 / 3, .85, 1.10),
}
STRENGTH_BOUNDS = (.20, 1.60)
REFINEMENT_ROUNDS = 2
DEVELOPMENT_FINALISTS_PER_PROFILE = 2
BASELINE = ("early", 2 / 3)
MATCHED_CONSTANT = ("constant", 0.6375)
PROMPT_ONLY_CONTROL = ("early", 1.2)
REFERENCE_QUESTIONS = ARCHIVE / "jev-review/report.json"


def raw_for(method):
    if method == "prompt-only":
        result = read(PRODUCT / "result.json")
        if result["actual_updates"] != BUDGET or result["ema_decay"] is not None:
            raise ValueError("The prompt-only control must be the completed RAW192 adapter")
        return result["equal_best"][0]["raw"]
    if method == "incumbent":
        path = ARCHIVE / "cohorts/soft-subject/result.json"
    elif method in METHODS:
        path = ROOT / "cohorts" / method / "result.json"
    else:
        raise ValueError(f"Unknown optimization arm {method}")
    result = read(path)
    if result["step"] != BUDGET or result["ema_decay"] is not None:
        raise ValueError(f"{method} does not have its completed RAW{BUDGET}")
    return result["raw"]


def reviewers():
    concept, integrity = attach_judge()
    from particle_sliders.alpha_tune_jev import JEV_MODEL, JEV_URL, JevJudge, _Image, _digest, _response_values
    from particle_sliders.candidate_failure import FileRecord

    protocol = read(ROOT / "search-protocol.json")
    if pin(REFERENCE_QUESTIONS) != protocol["review_rubric"]:
        raise ValueError("The ranking rubric changed after the optimization was declared")
    rubric = read(REFERENCE_QUESTIONS)
    if rubric["effect_question"] != QUESTION or rubric["model"] != JEV_MODEL or not all(rubric["calibration"]["checks"].values()):
        raise ValueError("The archived review uses a different or unqualified Jev rubric")
    questions = {key: {"type": "noul", "instructions": item[1]} for key, item in rubric["metrics"].items()}
    state = rubric["state"]
    reads = (ARCHIVE / "jev-review/receipts", ROOT / "incumbent-cache/receipts", ROOT / "image-breakage-audit/receipts")
    judge = JevJudge(ROOT / "jev/overall", QUESTION)

    def image(asset):
        path = Path(asset["path"])
        return _Image.read(FileRecord(str(path), path.stat().st_size, asset["sha256"]))

    def decide(assets, prompt, prompts):
        images = tuple(image(asset) for asset in assets)
        binding = {"endpoint": JEV_URL, "model": JEV_MODEL, "state": prompt, "questions": prompts,
                   "images": [item.identity() for item in images]}
        name = f"{_digest(binding)}.json"
        for cache in reads:
            receipt = cache / name
            if not receipt.is_file():
                continue
            saved = read(receipt)
            if saved["request"] != binding or saved["response_sha256"] != _digest(saved["response"]):
                raise ValueError("Archived Jev review receipt changed")
            if saved["status"] == "ok":
                return _response_values(saved["response"], prompts)
        return judge._decide(images, prompt, prompts)

    def observe(off, on):
        overall = decide((off, on), state, questions)
        before = concept("direct-final-boss", off, off)
        after = concept("direct-final-boss", off, on)
        return {"off": off, "on": on, "scores": overall, "reference": before, "score": after,
                "gain": None if before is None or after is None else after - before, "integrity": integrity(on)}

    return observe, decide


def summarize(cases, metric=METRIC):
    checks = [row["integrity"]["status"] for row in cases.values()]
    complete = all(row["scores"] is not None and row["gain"] is not None for row in cases.values())
    required = {METRIC: (DISTINCTION, ORIGINAL_DISTINCTION), "boss_distinction_total": (DISTINCTION,)}
    complete = complete and all(row["scores"].get(key) is not None
                                for row in cases.values() for key in required.get(metric, ()))
    status = "fail" if "fail" in checks else "pass" if complete and set(checks) == {"pass"} else "unmeasured"
    means = None
    if complete:
        keys = [key for key in next(iter(cases.values()))["scores"]
                if all(row["scores"].get(key) is not None for row in cases.values())]
        means = {key: fmean(row["scores"][key] for row in cases.values()) for key in keys}
        means["gain"] = fmean(row["gain"] for row in cases.values())
        if metric == METRIC:
            means[EFFECT] = round(fmean(case_effect(row["scores"]) for row in cases.values()), 12)
            means[PRESERVATION] = preservation(means)
        elif metric == "boss_distinction_total":
            means[metric] = aggregate_total(row["scores"] for row in cases.values())
        elif metric == "boss_adjusted_total":
            means[metric] = aggregate_total((row["scores"] for row in cases.values()), "character")
        elif metric == "weighted_total":
            means[metric] = weighted_total(means)
    return {"cases": cases, "status": status, "means": means}


def render_setting(runtime, raw, key, row, folder, schedule):
    proof = folder / "adapter.json"
    binding = {"raw": raw, "prompt": row["prompt"], "seed": row["seed"], "gains": schedule}
    if proof.exists():
        if read(proof) != binding:
            raise ValueError("Render folder is bound to a different adapter or sampler")
    else:
        if (folder / "render.json").exists():
            raise ValueError("Existing render has no RAW provenance")
        write(proof, binding)
    return render(runtime, row["prompt"], row["seed"], folder, gains=schedule, variation=key)


def archived_frame(path, raw, case, row, schedule, model):
    setting = read(path)
    if setting["raw"] != raw or setting["effective_gains"] != schedule:
        raise ValueError("Archived render has different RAW or tuning strengths")
    saved = setting["cases"][case]["on"]
    path = Path(saved["path"])
    history = read(path.with_name("render.json"))
    current = pin(path)
    if (history["gains"] != schedule or history["model"] != model
            or (history["prompt"], history["seed"]) != (row["prompt"], row["seed"])
            or history["image"] != current or saved["sha256"] != current["sha256"]
            or ("bytes" in saved and path.stat().st_size != saved["bytes"])):
        raise ValueError("Archived image came from a different native render")
    return current


def midpoints(grid, winners):
    values = sorted(grid)
    candidates = set()
    for alpha in winners:
        pos = values.index(alpha)
        if pos:
            candidates.add((values[pos - 1] + alpha) / 2)
        elif len(values) > 1:
            candidates.add(max(STRENGTH_BOUNDS[0], alpha - (values[1] - alpha) / 2))
        if pos + 1 < len(values):
            candidates.add((values[pos + 1] + alpha) / 2)
        elif len(values) > 1:
            candidates.add(min(STRENGTH_BOUNDS[1], alpha + (alpha - values[-2]) / 2))
    return sorted(candidates - set(grid))


def qualified(settings):
    return sorted((row for row in settings if row["status"] == "pass"),
                  key=lambda row: ranking(row["means"]), reverse=True)


def alpha_tune(runtime, method):
    from replay import loaded

    protocol = read(ROOT / "search-protocol.json")
    if protocol["grid"] != {key: list(values) for key, values in GRID.items()} or protocol["strength_bounds"] != list(STRENGTH_BOUNDS):
        raise ValueError("Alpha tuning differs from the declared search")
    if protocol["active_score"] != OBJECTIVE or protocol["boss_distinction_rubric"] != RUBRIC:
        raise ValueError("Alpha tuning differs from the declared leaderboard objective")
    gate = pin(BREAKAGE_RULE)
    if protocol["image_breakage_rule"] != gate:
        raise ValueError("Alpha tuning requires the calibrated image check")
    formulation = protocol.get("formulation_comparison", {})
    fixed_only = method in formulation.get("methods", ())
    if fixed_only and (formulation["shared_strength"] != list(BASELINE)
                       or formulation["secondary_control"] != list(MATCHED_CONSTANT)):
        raise ValueError("Formulation comparison changed its shared slider strength")
    raw = raw_for(method)
    key = loaded(runtime, raw, BUDGET)
    board = ROOT / "tuning" / method
    sampler = runtime.scheduler(10).sigmas.tolist()
    if sampler != read(PRODUCT / "schedules.json")["sigmas"]:
        raise ValueError("The comparison needs the pinned ten-step Anima sampler")
    original = {"incumbent": ARCHIVE / "tuning/soft-subject",
                "prompt-only": ARCHIVE / "tuning/prompt-only"}.get(method)
    examples = {row["label"]: row for row in panel()}
    references = {case: read(PRIOR / "shared-off" / case / "render.json")["image"] for case in examples}
    development = tuple(case for case in examples if case not in SCREEN)
    observe, decide = reviewers()
    archived_conditions = ([(file.parent.name, float.fromhex(file.stem))
                            for file in sorted((original / "heldout").glob("*/*.json"))
                            if STRENGTH_BOUNDS[0] <= float.fromhex(file.stem) <= STRENGTH_BOUNDS[1]]
                           if method == "prompt-only" else [])
    earlier_reports = [read(board / name) for name in PRIOR_REPORTS if (board / name).exists()]
    if any(earlier["raw"] != raw or earlier["sampler"] != sampler
           or earlier["screen_cases"] != list(SCREEN) or earlier["development_cases"] != list(development)
           for earlier in earlier_reports):
        raise ValueError("The earlier alpha search is bound to a different experiment")

    def frame(profile, alpha, case, schedule):
        folder = board / "renders" / profile / alpha.hex() / case
        if original is not None:
            stage = "screen" if case in SCREEN else "heldout"
            previous = original / stage / profile / f"{alpha.hex()}.json"
            if previous.exists():
                return archived_frame(previous, raw, case, examples[case], schedule, runtime.identity)
        return render_setting(runtime, raw, key, examples[case], folder, schedule)

    def evaluate(pool, profile, alpha, case_ids, stage):
        schedule = gains(runtime, alpha, profile)
        path = board / stage / profile / f"{alpha.hex()}.json"
        binding = {"raw": raw, "profile": profile, "alpha": alpha, "gains": schedule,
                   "question": QUESTION, "review_sha256": pin(REFERENCE_QUESTIONS)["sha256"]}
        if path.exists():
            result = read(path)
            if any(result[field] != value for field, value in binding.items()) or set(result["cases"]) != set(case_ids):
                raise ValueError("Search receipt has a different input binding")
            observed = result["cases"]
        else:
            pending = {case: pool.submit(observe, references[case], frame(profile, alpha, case, schedule))
                       for case in case_ids}
            observed = {case: future.result() for case, future in pending.items()}
            write(path, binding | summarize(observed, "overall"))
        if summarize(observed, "overall")["status"] != "pass":
            result = binding | summarize(observed)
        else:
            concepts = {case: pool.submit(augment, row, decide) for case, row in observed.items()}
            quality = {case: pool.submit(assess_breakage, row["off"], row["on"], decide)
                       for case, row in observed.items()}
            observed = {case: future.result() for case, future in concepts.items()}
            result = binding | with_readability(summarize(observed), {case: future.result() for case, future in quality.items()})
        print(method, stage, profile, f"{alpha:.6g}", result["status"], result["means"], flush=True)
        return result

    screen, evaluated, reviewed = {}, {}, {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        for profile, grid in ({} if fixed_only else GRID).items():
            earlier = (float(value) for report in earlier_reports for value in report["screen"][profile])
            archived = (alpha for kind, alpha in archived_conditions if kind == profile)
            strengths = sorted(set(grid).union(earlier, archived))
            if method == "prompt-only" and profile == PROMPT_ONLY_CONTROL[0]:
                strengths = sorted({*strengths, PROMPT_ONLY_CONTROL[1]})
            initial = {alpha: evaluate(pool, profile, alpha, SCREEN, "screen") for alpha in strengths}
            for _ in range(REFINEMENT_ROUNDS):
                passing = qualified(initial.values())
                if not passing:
                    break
                tied = [row["alpha"] for row in passing if ranking(row["means"]) == ranking(passing[0]["means"])]
                for alpha in midpoints(initial, tied):
                    initial[alpha] = evaluate(pool, profile, alpha, SCREEN, "screen")
            screen[profile] = initial
            passing = qualified(initial.values())
            kept = []
            selections = []
            for contender in passing:
                if (len(kept) >= DEVELOPMENT_FINALISTS_PER_PROFILE
                        and ranking(contender["means"]) < tuple(kept[-1]["screen_rank"])):
                    break
                observed = evaluate(pool, profile, contender["alpha"], development, "development")
                result = observed | {"screen_score": contender["means"][METRIC], "screen_rank": list(ranking(contender["means"]))}
                selections.append(result)
                reviewed[profile, contender["alpha"]] = result
                if observed["status"] == "pass":
                    kept.append(result)
            evaluated[profile] = selections
        fixed = {}
        comparable = {}
        fixed_conditions = (BASELINE, MATCHED_CONSTANT, PROMPT_ONLY_CONTROL) if method == "prompt-only" else (BASELINE, MATCHED_CONSTANT)
        for profile, alpha in fixed_conditions:
            condition = (profile, alpha)
            settings = screen.setdefault(profile, {})
            if alpha not in settings:
                settings[alpha] = evaluate(pool, profile, alpha, SCREEN, "screen")
            checked = settings[alpha]
            measured = evaluate(pool, profile, alpha, development, "development")
            fixed[condition] = measured
            reviewed[condition] = measured
            if checked["status"] == "pass":
                comparable[condition] = measured | {"screen_score": checked["means"][METRIC]}

        replayed = {}
        earlier_conditions = [(setting["profile"], setting["alpha"])
                              for report in earlier_reports for setting in report["qualifying_settings"]]
        for condition in dict.fromkeys((*earlier_conditions, *archived_conditions)):
            checked = screen[condition[0]][condition[1]]
            measured = evaluate(pool, *condition, development, "development")
            reviewed[condition] = measured
            replayed[condition] = {"profile": condition[0], "alpha": condition[1],
                                   "screen_status": checked["status"], "development_status": measured["status"]}
            if checked["status"] == "pass" and measured["status"] == "pass":
                comparable[condition] = measured | {"screen_score": checked["means"][METRIC]}

    candidates = [row for group in evaluated.values() for row in group]
    measured_conditions = {(row["profile"], row["alpha"]) for row in candidates}
    candidates.extend(row for condition, row in comparable.items() if condition not in measured_conditions)
    candidates = qualified(candidates)
    selected = [row for row in candidates if ranking(row["means"]) == ranking(candidates[0]["means"])]
    report = {"method": method, "step": BUDGET, "raw": raw, "selection_metric": METRIC, "objective": OBJECTIVE,
              "image_breakage_rule": gate,
              "search_mode": "shared strengths" if fixed_only else "alpha search",
              "scope": "previously exposed Anima development cases", "screen_cases": list(SCREEN),
              "development_cases": list(development), "sampler": sampler, "qualifying_settings": candidates,
              "selected": selected, "fixed_strength": fixed[BASELINE], "matched_constant": fixed[MATCHED_CONSTANT],
              "screen": screen, "evaluated": evaluated, "replayed": list(replayed.values()),
              "reviewed_settings": list(reviewed.values())}
    write(board / REPORT, report)


def run(runtime, method):
    if method in ("confirm0", "confirm1"):
        from confirm import confirm
        confirm(runtime, int(method[-1]))
    else:
        alpha_tune(runtime, method)
