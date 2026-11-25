"""Run a bounded Anima slider study, from matched teachers through confirmed RAW replay."""

import argparse
import fcntl
import json
import math
from pathlib import Path
import shlex
import sys

from ..contracts import atomic_json, digest
from .outputs import Outputs, Unmeasured, read
from .leaderboard import gallery
from .objectives import DISTINCT_BOSS, MATCHED_GAIN, default_objective, protocol_id, selection_score
from .reference import REFERENCES, compare_reference
from .teachers import collect, validation_panels
from .rate_search import DEFAULT_ARM_BUDGET, explore
from .training import (
    Catastrophe,
    confirm,
    rate_candidates,
    recheck_screened_cases,
    select_checkpoint,
    train,
    tune,
    validate_calibration,
)

REFERENCE = {"g": 0.00075, "d": 0.009, "particle": 0.03}


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vibe", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/anima/adult-study.json",
    )
    parser.add_argument("--boss-targets", type=Path,
                        metavar="late-cues|FILE",
                        help="use the bundled late-cues recipe or a matching teacher-pair recipe file")
    parser.add_argument("--require-reference-gain", action=argparse.BooleanOptionalAction, default=None,
                        help="require a higher score on this slider’s registered objective than its matched reference for a successful full run (default for bundled sliders)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-cap-gib", type=float, default=19.0)
    parser.add_argument(
        "--steps",
        type=int,
        help="maximum updates; defaults to the confirmed budget or 256 for a new search",
    )
    parser.add_argument("--min-teachers", type=int, default=6)
    parser.add_argument(
        "--collection-attempts",
        type=int,
        default=8,
        help="maximum matched seeds per scene",
    )
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument(
        "--prepare-only",
        action="store_true",
        help="collect and verify resumable frozen teachers and validation panels",
    )
    stop.add_argument(
        "--select-only",
        action="store_true",
        help="reserve a qualified RAW and strength before rendering held-out ON images",
    )
    parser.add_argument(
        "--search-rates",
        action="store_true",
        help="screen G/D/particle rates and continue qualified candidates",
    )
    parser.add_argument("--rate-arm-budget", type=int, help="maximum distinct LR screens, including the seven initial probes (default 31)")
    for role in REFERENCE:
        parser.add_argument(
            "--" + role + "-lr",
            type=float,
            help="override the absolute " + role + " rate",
        )
    args = parser.parse_args(argv)
    if args.boss_targets == Path("late-cues"):
        args.boss_targets = Path(__file__).resolve().parents[2] / "configs/anima/boss-teachers/late-cues.json"
    if args.require_reference_gain is None:
        args.require_reference_gain = args.vibe.casefold() in REFERENCES and not (args.prepare_only or args.select_only)
    if args.require_reference_gain and (args.vibe.casefold() not in REFERENCES or args.prepare_only or args.select_only):
        parser.error("--require-reference-gain requires a full confirmation for a slider with a pinned reference")
    if args.rate_arm_budget is not None and args.rate_arm_budget < 7:
        parser.error("--rate-arm-budget must allow at least seven distinct rate arms")
    if args.steps is not None and (args.steps < 16 or args.steps % 16):
        parser.error("--steps must be a positive multiple of 16")
    if args.collection_attempts < 1 or args.min_teachers < 1:
        parser.error("teacher count and collection attempts must be positive")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.g_lr, args.d_lr, args.particle_lr, args.gpu_cap_gib)
        if value is not None
    ):
        parser.error("rates and GPU cap must be finite and positive")
    return args


def training_plan(args, root, concept, config):
    bound = root / "training-config.json"
    saved = read(bound) if bound.exists() else None
    recipes = read(
        Path(__file__).resolve().parents[2] / "configs/anima/study-recipes.json"
    )
    measured = recipes["concepts"].get(concept)
    if args.search_rates or (
        measured and measured["applies_to_plan_sha256"] != digest(config)
    ):
        measured = None
    # A retained failure is replay evidence. Fresh bundled Boss runs start
    # at the published training recipe and must qualify on the current data.
    if measured and not saved:
        measured = measured.get("fresh_training", measured)
    baseline = saved["rates"] if saved else measured["rates"] if measured else REFERENCE
    rates = dict(baseline)
    for role in rates:
        explicit = getattr(args, f"{role}_lr")
        if explicit is not None:
            rates[role] = explicit
    if saved:
        default_budget = saved.get("max_updates", saved["horizon"])
    else:
        default_budget = (
            measured["update"]
            if measured and rates == measured["rates"]
            else recipes["horizon"]
        )
    budget = args.steps or default_budget
    # Selecting a shorter RAW must preserve the optimizer schedule that produced it.
    starting_horizon = (
        measured.get("horizon", recipes["horizon"])
        if measured and rates == measured["rates"]
        else recipes["horizon"]
    )
    horizon = saved["horizon"] if saved else max(budget, starting_horizon)
    plan = dict(rates=rates, budget=budget, horizon=horizon)
    arm_budget = getattr(args,"rate_arm_budget",None)
    if arm_budget is None and saved:
        arm_budget = saved.get("rate_arm_budget")
    if arm_budget is not None:
        plan["rate_arm_budget"] = arm_budget
    return plan


def screened_rates(outputs, cache, panels, reference, steps, schedule_horizon=None, arm_budget=DEFAULT_ARM_BUDGET):
    """Compare absolute training rates at the same update, objective and unit strength."""
    ledger = outputs.root / "rate-search.json"
    prior = read(ledger) if ledger.is_file() else []
    if prior and not (outputs.root / "rate-search-coverage.json").exists():
        archive = outputs.root / "rate-search-history" / (digest(prior) + ".json")
        if not archive.exists():
            atomic_json(archive, prior)
    ranking = {digest(row["rates"]): row for row in prior}
    objective = getattr(outputs, "objective", MATCHED_GAIN)
    def screen(candidate):
        try:
            result = train(outputs, cache, panels, candidate, steps, "screen", schedule_horizon=schedule_horizon)
            if objective == DISTINCT_BOSS:
                result = outputs.with_objective(result)
            row = dict(rates=candidate, status=result["status"], mean_gain=result["mean_gain"],
                       objective=objective, objective_score=selection_score(result, objective))
        except (Catastrophe, Unmeasured) as error:
            row = dict(rates=candidate, status="fail" if isinstance(error, Catastrophe) else "unmeasured", reason=str(error))
        ranking[digest(candidate)] = row
        atomic_json(ledger, list(ranking.values()))
        return row

    def coverage(status, best, factor, missing, unavailable, count):
        atomic_json(outputs.root / "rate-search-coverage.json", dict(
            algorithm="coordinate-log-v1", status=status, reference=reference, objective=objective,
            screen_updates=steps//4, schedule_horizon=schedule_horizon or steps, alpha=1.0,
            panels=digest(panels["rates"]) if panels else None, arm_budget=arm_budget,
            measured_or_attempted=count, best=best, factor=factor, untested=missing, unmeasured=unavailable))

    yield from explore(reference, rate_candidates(reference), screen, coverage, arm_budget)


def fit_candidates(outputs, cache, panels, plan, args):
    rates = plan["rates"]
    if args.search_rates:
        yield from screened_rates(
            outputs, cache, panels, rates, plan["budget"], plan["horizon"], plan.get("rate_arm_budget", DEFAULT_ARM_BUDGET)
        )
        return
    yield rates
    # The default command first uses the measured recipe. If it fails before
    # confirmation, expand the search in the same run and reuse its checkpoints.
    if all(getattr(args, f"{role}_lr") is None for role in REFERENCE):
        for candidate in screened_rates(
            outputs, cache, panels, rates, plan["budget"], plan["horizon"], plan.get("rate_arm_budget", DEFAULT_ARM_BUDGET)
        ):
            if candidate != rates:
                yield candidate


def choose_for_confirmation(outputs, cache, panels, plan, args):
    """Commit one candidate before the first held-out image is rendered or scored."""
    root = outputs.root
    reservation = root / "confirmation-selection.json"
    if reservation.exists():
        return read(reservation)
    # Completed runs from earlier versions already bound their held-out selection.
    if (root / "outcome.json").exists():
        previous = read(root / "outcome.json")
        selection = {
            key: previous[key] for key in ("rates", "strength", "selected_step")
        }
        atomic_json(reservation, selection)
        return selection
    # Commit the training decision before looking at any alpha trial. A failed
    # calibration must never send the search back to choose different weights.
    training_selection = root / "training-selection.json"
    if training_selection.exists():
        frozen = read(training_selection)
    else:
        steps = plan["budget"]
        failures = []
        for rates in fit_candidates(outputs, cache, panels, plan, args):
            try:
                fit = train(
                    outputs,
                    cache,
                    panels,
                    rates,
                    steps,
                    "fit",
                    schedule_horizon=plan["horizon"],
                    always_check_rates=plan.get("always_check_rates", False),
                )
                screen_check = recheck_screened_cases(
                    outputs,
                    cache,
                    panels,
                    rates,
                    steps,
                    fit,
                    required=plan.get("always_check_rates", False),
                )
                observer = {"outputs": outputs} if getattr(outputs, "objective", None) == DISTINCT_BOSS else {}
                selected = select_checkpoint(root, rates, steps, **observer)
            except (Unmeasured, Catastrophe) as error:
                failures.append(dict(rates=rates, reason=str(error)))
                atomic_json(root / "fit-failures.json", failures)
                continue
            frozen = dict(rates=rates, raw=selected["raw"], screened=screen_check is not None)
            atomic_json(training_selection, frozen)
            break
        else:
            raise Unmeasured("No rate candidate completed with usable measurements; see fit-failures.json")
    best = tune(outputs, cache, panels, frozen, screened=frozen["screened"])
    if best["raw"] != frozen["raw"] or best["profile"] != "constant":
        raise ValueError("Alpha selection must retain the training checkpoint and constant profile")
    selection = dict(
        rates=frozen["rates"], strength=best,
        selected_step=read(Path(best["raw"]["path"]).with_suffix(".scoped.json"))["completed_steps"],
    )
    atomic_json(reservation, selection)
    return selection


def study_panels(spec, concept, root):
    base = spec["validation"]
    extra = spec.get("screen_cases", {}).get(concept, [])
    holdout = spec.get("confirmation_cases", {}).get(concept)
    if not extra and holdout is None:
        return base, False
    rates = list(base["rates"])
    for case in extra:
        if case not in rates:
            rates.append(case)
    screened = base | {"rates": rates}
    current = screened if holdout is None else screened | {"confirm": holdout}
    bound = root / "data-config.json"
    if bound.exists():
        original = read(bound)["spec"]
        # Keep both the cases and the screening behavior of earlier runs.
        for groups, required in [
            (current, bool(extra)),
            (screened, bool(extra)),
            (base, False),
        ]:
            if (
                digest(dict(teachers=spec["concepts"][concept], validation=groups))
                == original
            ):
                return groups, required
        legacy = {
            k: v
            for k, v in spec.items()
            if k not in ("screen_cases", "confirmation_cases")
        }
        if digest(legacy) == original:
            return base, False
    return current, bool(extra)


def run(args):
    import torch
    from ..backends.anima import TurboRuntime
    from ..cache import TargetCache, prepare_targets
    from ..core_source import verify_core
    from ..vibe import validate_manifest
    from particle_sliders.candidate_failure import FileRecord

    root = args.out.resolve()
    spec = read(args.spec)
    names = {k.casefold(): k for k in spec["concepts"]}
    if args.vibe.casefold() not in names:
        raise ValueError("--vibe must name a concept in --spec")
    concept = names[args.vibe.casefold()]
    config = spec["concepts"][concept]
    groups, always_check_rates = study_panels(spec, concept, root)
    if not 1 <= args.min_teachers <= len(config["teachers"]):
        raise ValueError("--min-teachers exceeds the scene pool")
    plan = training_plan(args, root, concept, config)
    rates, steps = plan["rates"], plan["budget"]
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Native training requires a CUDA device")
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, args.gpu_cap_gib * 2**30 / total), device
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    rt = TurboRuntime(
        args.model_dir, device=args.device, verify=True, checkpointing=True
    )
    try:
        opts = dict(
            schema="anima-stock-empf-study-v2",
            model=rt.identity,
            core=verify_core(),
            spec=digest(dict(teachers=config, validation=groups)),
            concept=concept,
            minimum=args.min_teachers,
            attempts=args.collection_attempts,
            ema_decay=None,
        )
        if args.boss_targets:
            if concept.casefold() != "final boss":
                raise ValueError("--boss-targets requires the Final Boss concept")
            opts["field_recipe_sha256"] = digest(read(args.boss_targets))
        bound = root / "data-config.json"
        if bound.exists():
            original = read(bound)
            if original["schema"] == "anima-stock-empf-study-v1" and original[
                "spec"
            ] in (
                digest(spec),
                digest(
                    {
                        k: v
                        for k, v in spec.items()
                        if k not in ("screen_cases", "confirmation_cases")
                    }
                ),
            ):
                atomic_json(root / "data-config-v1.json", original)
                original = original | {"schema": opts["schema"], "spec": opts["spec"]}
            if original | {"attempts": args.collection_attempts} != opts:
                raise ValueError("The existing run has a different study configuration")
            if (
                args.collection_attempts < original["attempts"]
                and not (root / "manifest.json").exists()
            ):
                raise ValueError("The collection attempt budget can only increase")
        atomic_json(bound, opts)
        objective = default_objective(concept)
        policy = root / "selection-objective.json"
        if policy.exists():
            expected = dict(objective=objective, protocol=protocol_id() if objective == DISTINCT_BOSS else None)
            if read(policy) != expected:
                raise ValueError("Saved selection objective differs from the registered study protocol")
        elif (root / "confirmation-selection.json").exists() or (root / "outcome.json").exists():
            objective = MATCHED_GAIN  # A previously reserved scalar belongs to its original measurement.
        else:
            atomic_json(policy, dict(objective=objective, protocol=protocol_id() if objective == DISTINCT_BOSS else None))
        outputs = Outputs(root, rt, concept, config["effect"], objective=objective)
        calibration = outputs.integrity.calibrate()
        atomic_json(root / "detector.json", calibration)
        if not calibration["passed"]:
            raise Unmeasured(
                "Structural detector did not recognize its calibration images"
            )
        manifest_path = root / "manifest.json"
        manifest = (
            read(manifest_path)
            if manifest_path.exists()
            else collect(
                outputs, config, concept, args.min_teachers, args.collection_attempts
            )
        )
        validate_manifest(manifest)
        if (
            manifest["collection_identity"]["teacher_config"] != digest(config)
            or manifest["collection_identity"]["model"] != rt.identity
            or len(manifest["rows"]) < args.min_teachers
        ):
            raise ValueError("Retained teacher manifest is from another study")
        for row in manifest["rows"]:
            for side in ("neutral", "positive"):
                decision = row["selection"][side]
                expected = FileRecord(**decision["output"])
                if (
                    decision["integrity"]["status"] != "pass"
                    or FileRecord.from_path(expected.path) != expected
                ):
                    raise ValueError(
                        "Selected teacher image or integrity decision differs"
                    )
        panels = validation_panels(outputs, groups, args.collection_attempts)
        if always_check_rates:
            present = {(case["prompt"], case["seed"]) for case in panels["rates"]}
            if any(
                (case["prompt"], case["seed"]) not in present
                for case in spec["screen_cases"][concept]
            ):
                raise Unmeasured(
                    "The registered regression prompt and seed lack a qualified OFF image"
                )
        target = root / "frozen" / "targets"
        if not (target / "index.json").is_file():
            prepare_targets(rt, manifest, manifest["variation"], target, resolution=512)
        cache = TargetCache(target, verify=True)
        if (
            cache.index["identity"]["model"] != rt.identity
            or cache.index["identity"]["manifest_sha256"] != manifest["sha256"]
        ):
            raise ValueError(
                "Frozen targets do not match the verified manifest and model"
            )
        if args.boss_targets:
            from .boss_targets import prepare_boss_targets
            cache = prepare_boss_targets(rt, cache, root / "frozen/boss-targets", args.boss_targets)
        if args.prepare_only:
            return dict(
                status="prepared",
                teacher_count=len(manifest["rows"]),
                fields=len(cache),
                cache=cache.index["fingerprint"],
                panel_sizes={k: len(v) for k, v in panels.items()},
            )
        settings = dict(
            data=manifest["sha256"],
            validation_sha256=digest(panels),
            rates=rates,
            horizon=plan["horizon"],
            search=args.search_rates,
        )
        if "rate_arm_budget" in plan:
            settings["rate_arm_budget"] = plan["rate_arm_budget"]
        if steps != plan["horizon"]:
            settings["max_updates"] = steps
        training_config = root / "training-config.json"
        if always_check_rates and (
            not training_config.exists()
            or read(training_config).get("always_check_rates")
        ):
            settings["always_check_rates"] = True
            plan["always_check_rates"] = True
        if training_config.exists():
            original = read(training_config)
            # An explicitly requested rate search can expand a failed single-rate
            # fit on the same data, horizon and reference rates. Existing arms
            # keep their exact cursors and structural stop receipts.
            if original != settings and not (
                args.search_rates
                and not original["search"]
                and original | {"search": True} == settings
                and (root / "fit-failures.json").is_file()
                and not (root / "outcome.json").exists()
                and not (root / "confirmation-selection.json").exists()
            ):
                raise ValueError("Training settings differ from the committed run")
        atomic_json(training_config, settings)
        selection = choose_for_confirmation(outputs, cache, panels, plan, args)
        if policy.exists():
            validate_calibration(root, selection["strength"], panels["steps"], outputs)
        outcome = root / "outcome.json"
        if args.select_only:
            return dict(
                status="selected",
                rates=selection["rates"],
                steps=selection["selected_step"],
                raw=selection["strength"]["raw"],
                profile=selection["strength"]["profile"],
                objective=selection["strength"].get("objective", MATCHED_GAIN),
                score=selection_score(selection["strength"]),
                integrity=selection["strength"]["status"],
                alpha=selection["strength"]["alpha"],
            )
        if outcome.exists():
            measured = read(outcome)
            if (
                any(measured[key] != value for key, value in selection.items())
                or FileRecord.from_path(selection["strength"]["raw"]["path"]).sha256
                != selection["strength"]["raw"]["sha256"]
            ):
                raise ValueError(
                    "Saved confirmation differs from its reserved selection"
                )
        else:
            confirmation = confirm(outputs, cache, panels, selection["strength"])
            measured = dict(
                vibe=concept,
                particlegan="0.12.0",
                selection="raw",
                ema_decay=None,
                teacher_cache=cache.index["fingerprint"],
                confirmation=confirmation,
                **selection,
            )
            atomic_json(outcome, measured)
        reference = None
        if concept.casefold() in REFERENCES:
            reference = compare_reference(concept, measured["confirmation"], config["effect"])
            if reference["decision"] == "exceeds_reference" and measured["strength"]["status"] != "pass":
                reference = dict(reference, decision="retain_reference", selection_integrity=measured["strength"]["status"])
            atomic_json(root / "reference-comparison.json", reference)
        gallery(root, measured, reference=reference)
        result = measured["confirmation"]
        value = selection_score(result, objective)
        if (measured["strength"]["status"] != "pass" or result["status"] != "pass"
                or value is None or value <= 0):
            raise Unmeasured(
                "The selected setting did not confirm a positive intact effect on held-out cases"
            )
        return measured
    finally:
        rt.close()


def main(argv=None):
    args = arguments(argv)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another process owns this run", file=sys.stderr)
            return 75
        try:
            result = run(args)
        except (RuntimeError, ValueError, OSError) as error:
            status = dict(
                status="incomplete",
                reason=str(error),
                resume=shlex.join(
                    ["anima-vibe-train", *(sys.argv[1:] if argv is None else argv)]
                ),
            )
            atomic_json(root / "status.json", status)
            print(json.dumps(status))
            return 2
        if args.select_only:
            atomic_json(root / "selection-summary.json", result)
            if not (root / "outcome.json").exists():
                atomic_json(root / "status.json", result)
            print(json.dumps(result))
            return 0
        if args.prepare_only:
            atomic_json(root / "preparation.json", result)
            if not (root / "training-config.json").exists():
                atomic_json(root / "status.json", result)
            print(json.dumps(result))
            return 0
        summary = dict(
            status="confirmed",
            raw=result["strength"]["raw"],
            steps=result["selected_step"],
            gain=result["confirmation"]["mean_gain"],
            objective=result["confirmation"].get("objective", MATCHED_GAIN),
            score=selection_score(result["confirmation"]),
            outcome=str(root / "outcome.json"),
        )
        if result.get("vibe", "").casefold() in REFERENCES:
            summary["role"] = "study_candidate"
            reference_path = root / "reference-comparison.json"
            summary["reference_comparison"] = str(reference_path)
            summary["release_decision"] = read(reference_path)["decision"]
        publishable = not args.require_reference_gain or summary.get("release_decision") == "exceeds_reference"
        if not publishable:
            summary["status"] = summary["release_decision"]
            summary["measurement"] = "confirmed_against_off"
        atomic_json(root / "status.json", summary)
        print(json.dumps(summary))
        return 0 if publishable else 3


if __name__ == "__main__":
    raise SystemExit(main())
