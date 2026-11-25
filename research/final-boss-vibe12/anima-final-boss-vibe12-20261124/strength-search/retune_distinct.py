"""Tune a Final Boss RAW adapter using cross-character design, effect and damage together."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from statistics import fmean
from time import monotonic
import fcntl
import sys

import retune_final_boss_visible as native
from retune_final_boss_visible import ROOT, WORK, BOSS, ORIGINAL, DAMAGE, SCORE, GAIN, harness, adaptive_alpha
from design_range import pair_observer, STATE, QUESTIONS
from design_scores import summarize

RUN = "leaderboard-retune-design"
METRIC = "distinct_boss_score"


def select(evaluate, historical, record):
    """Choose every search step and finalist by the distinct-design objective."""
    paths, finalists = {}, []
    for profile in native.PROFILES:
        trials, steps = adaptive_alpha.search(lambda a: evaluate(profile,a,"search")["design"]["score"])
        paths[profile] = {"trials":[{"alpha":a,"score":s} for a,s in trials.items()], "steps":steps}
        record(paths)
        ranked = sorted(trials, key=lambda a:-trials[a])
        finalists.extend((profile,a) for a in ranked[:2])
    settings = list(dict.fromkeys([*historical, *finalists, native.BASELINE, native.FIXED_EARLY]))
    best, measured, skipped = None, [], []
    for profile, alpha in settings:
        # The repeat deduction is nonnegative. Even zero repetition cannot lift this upper bound.
        bound = historical.get((profile,alpha))
        if best is not None and bound is not None and bound <= best["design"]["score"]:
            skipped.append({"profile":profile,"alpha":alpha,"upper_bound":bound})
            continue
        trial = evaluate(profile,alpha,"validation")
        measured.append(trial)
        if best is None or trial["design"]["score"] > best["design"]["score"]:
            best = trial
    return paths,best,measured,skipped


class NativeRenderNeeded(Exception):
    pass


def run(method, cached_only=False):
    import boss_review
    import tune
    harness.ROOT = ROOT
    boss_review.FOLDER = ROOT / "boss-character-review"
    _, decide = tune.reviewers()
    from particle_sliders.alpha_tune import _load_definition, _render, _renderer
    from particle_sliders.alpha_tune_jev import JEV_MODEL
    from particle_sliders.strength_schedule import interval_strengths
    sys.path.insert(0,str(harness.PRIOR/"prompt-only-alpha-tune/renderer-installed"))
    trained = harness.read(ROOT / "cohorts" / method / "result.json")
    definition, body, raw, cases, _ = _load_definition(native.slider_definition(method,trained))
    fixed_path = ROOT/"tuning"/method/"boss-effect-tuning.json"
    fixed = harness.read(fixed_path)
    screen, develop = fixed["screen_cases"], fixed["development_cases"]
    if (trained["step"] != 192 or trained["ema_decay"] is not None or raw.sha256 != fixed["raw"]["sha256"]
            or trained["raw"] != fixed["raw"] or set(screen)&set(develop) or set(cases)!=set(screen+develop)
            or len(screen)<2 or len(develop)<2):
        raise ValueError("Distinct-design tuning requires the same RAW192 adapter and disjoint matched panels")
    target = WORK/RUN/method
    config = {"definition":asdict(definition),"leaderboard":harness.pin(fixed_path),"implementation":harness.pin(Path(__file__)),
              "search_objective":METRIC,"quality_metric":DAMAGE,"scoring_version":native.VERSION,"jev_binding":native.BINDING,
              "model":JEV_MODEL,"state":STATE,"questions":QUESTIONS,"search_cases":screen,"board_cases":develop,
              "sources":[harness.pin(p) for p in (WORK/"adaptive_alpha.py",ROOT/"metric-calibration/design_scores.py",
                         ROOT/"metric-calibration/design_range.py",ROOT/"metric-calibration/finished_image.py")],
              "start_alpha":adaptive_alpha.START,"initial_step":adaptive_alpha.INITIAL_STEP,
              "bounds":[adaptive_alpha.LOWER,adaptive_alpha.UPPER],"trial_budget_per_profile":adaptive_alpha.TRIAL_BUDGET,
              "profiles":list(native.PROFILES)}
    request = target/"request.json"
    if request.is_file() and harness.read(request)!=config:
        raise ValueError("Immutable distinct-design search inputs changed")
    harness.write(request,config)
    if (target/"result.json").is_file():
        return
    board = harness.read(ROOT/"storage/alpha-standings.json")
    if board["scoring_version"]!=native.VERSION or board["cases"]!=develop:
        raise ValueError("Unexpected score or case panel")
    # Freeze the incumbent and all earlier eight-case strengths even if the live board changes during a resume.
    history_path = target/"incumbents.json"
    if not history_path.is_file():
        rows = [r for r in board["rows"] if r["method"]==method and RUN not in r["runs"]]
        harness.write(history_path,rows)
    archived = {(r["profile"],r["alpha"]):r for r in harness.read(history_path)}
    for r in archived.values():
        if (set(r["cases"]) != set(develop) or
                fmean(native.score_case(v) for v in r["cases"].values()) != r["means"][SCORE] or
                any(v["off"]["sha256"] != cases[c]["off"].sha256 for c,v in r["cases"].items())):
            raise ValueError("Historical strength has an invalid upper bound")
    priors = []
    for name in (native.RUN,"leaderboard-retune-damage","leaderboard-retune-adaptive","leaderboard-retune-centered","leaderboard-retune"):
        prior = WORK/name/method
        if (prior/"request.json").is_file():
            old = harness.read(prior/"request.json")
            if any(old[k]!=config[k] for k in ("definition","leaderboard","search_cases","board_cases")):
                raise ValueError("Historical render belongs to a different adapter or panel")
            priors.append(prior)
    model = harness.read(native.CACHE/"index.json")["identity"]["model"]
    sigmas = fixed["sampler"]
    widths = [a-b for a,b in zip(sigmas,sigmas[1:])]
    observer = pair_observer()
    started = monotonic()
    with ExitStack() as resources:
        pool = resources.enter_context(ThreadPoolExecutor(12))
        @lru_cache(maxsize=1)
        def renderer():
            if cached_only:
                raise NativeRenderNeeded("Next probe needs a native GPU render")
            gpu = resources.enter_context(_renderer(body["renderer"])(raw,body["renderer_options"]))
            if gpu.sampler["model"]!=model or gpu.sampler["sigmas"]!=sigmas:
                raise ValueError("Native model or scheduler differs from the trained benchmark")
            return gpu

        def rate(off,on):
            b,orig = boss_review.assess(off,on,decide),boss_review.assess(off,off,decide)
            ratings={BOSS:b, ORIGINAL:orig, DAMAGE:native.assess(on)}
            return dict(ratings, **{GAIN:max(0,b-orig),SCORE:native.score_case(ratings)},off=off,on=on)

        def evaluate(profile,alpha,phase):
            names = screen if phase=="search" else develop
            gains = list(interval_strengths(alpha,widths,profile=profile))
            key = Path(profile)/alpha.hex()
            dest = target/"renders"/key
            receipt = dest/(phase+".json")
            setting = {"profile":profile,"alpha":alpha,"effective_strengths":gains,"cases":names}
            if receipt.is_file():
                previous = harness.read(receipt)
                if (previous["setting"] != setting or previous["design"] != summarize(
                        {c:r[SCORE] for c,r in previous["rows"].items()},previous["pairs"])):
                    raise ValueError("Cached distinct-design measurement changed")
                return previous
            begin = monotonic()
            old_rows = {}
            if phase=="validation" and (profile,alpha) in archived:
                old_rows=archived[profile,alpha]["cases"]
            for prior in priors:
                prior_trial=prior/"renders"/key/receipt.name
                if not old_rows and prior_trial.is_file():
                    old=harness.read(prior_trial)
                    if old["setting"]!=setting:
                        raise ValueError("Historical schedule or panel changed")
                    old_rows=old["rows"]
            jobs={}
            rendered=0
            for c in names:
                case=cases[c]
                off=harness.pin(Path(case["off"].path))
                if c in old_rows:
                    on=old_rows[c]["on"]
                else:
                    path=next((p/"renders"/key/c/"image.png" for p in priors
                               if (p/"renders"/key/c/"render.json").is_file()),dest/c/"image.png")
                    if not path.with_name("render.json").is_file():
                        _render(renderer(),case,gains,path)
                        rendered+=1
                    on=harness.pin(path)
                evidence=harness.read(Path(on["path"]).with_name("render.json"))
                if (off["sha256"]!=case["off"].sha256 or (c in old_rows and off["sha256"]!=old_rows[c]["off"]["sha256"])
                        or evidence["image"]!=on or harness.pin(Path(on["path"]))!=on or evidence["gains"]!=gains
                        or evidence["model"]!=model or evidence["prompt"]!=case["request"]["prompt"]
                        or evidence["seed"]!=case["request"]["seed"]):
                    raise ValueError(f"Unmatched native render for {method}/{c}")
                jobs[c]=pool.submit(rate,off,on)
            rows={c:j.result() for c,j in jobs.items()}
            point=native.summary(setting,rows)
            if point["status"]!="measured":
                raise ValueError("Incomplete effect or damage measurement")
            if phase=="validation" and (profile,alpha) in archived and point["means"][SCORE]!=archived[profile,alpha]["means"][SCORE]:
                raise ValueError("The historical upper bound differs from its source ratings")
            pair_start=monotonic()
            work=[(a,b,pool.submit(observer,rows[a]["on"],rows[b]["on"])) for a,b in combinations(names,2)]
            pairs=[dict(f.result(),cases=[a,b]) for a,b,f in work]
            design=summarize({c:r[SCORE] for c,r in rows.items()},pairs)
            report=dict(point,design=design,pairs=pairs,rendered=rendered,seconds=monotonic()-begin,
                        pair_seconds=monotonic()-pair_start)
            harness.write(receipt,report)
            print(phase,profile,alpha,round(100*design["score"],2),flush=True)
            return report

        historical={k:r["means"][SCORE] for k,r in archived.items()}
        paths,best,validations,skipped=select(evaluate,historical,lambda p:harness.write(target/"search-path.json",p))
    raw.verify()
    selected=best["setting"]
    tuning={"method":method,"objective":METRIC,"checkpoint":asdict(raw),"definition":asdict(definition),
            "model":model,"sigmas":sigmas,"profile":selected["profile"],"alpha":selected["alpha"],
            "effective_strengths":selected["effective_strengths"],"design":best["design"],
            "measurement":harness.pin(target/"renders"/selected["profile"]/selected["alpha"].hex()/"validation.json")}
    harness.write(target/"tuning.json",tuning)
    harness.write(target/"result.json",{"selected":tuning,"search":paths,"seconds_this_invocation":monotonic()-started,
                                        "measured":[v["setting"] for v in validations],"pruned":skipped})


if __name__=="__main__":
    method=sys.argv[-1]
    target=WORK/RUN/method
    target.mkdir(parents=True,exist_ok=True)
    with (target/"lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(method, "--cached-only" in sys.argv[1:-1])
        except NativeRenderNeeded as e:
            print(e,flush=True)
            raise SystemExit(75)
