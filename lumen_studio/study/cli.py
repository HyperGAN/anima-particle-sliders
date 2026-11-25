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
from .teachers import collect, validation_panels
from .training import (
    Catastrophe,
    confirm,
    rate_candidates,
    recheck_screened_cases,
    select_checkpoint,
    train,
    tune,
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
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="collect and verify resumable frozen teachers and validation panels",
    )
    parser.add_argument(
        "--search-rates",
        action="store_true",
        help="screen G/D/particle rates and continue qualified candidates",
    )
    for role in REFERENCE:
        parser.add_argument(
            "--" + role + "-lr",
            type=float,
            help="override the absolute " + role + " rate",
        )
    args = parser.parse_args(argv)
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
    horizon = saved["horizon"] if saved else max(budget, recipes["horizon"])
    return dict(rates=rates, budget=budget, horizon=horizon)


def screened_rates(outputs, cache, panels, reference, steps, schedule_horizon=None):
    """Rank matched-budget settings and refine D and particles at a changed G."""
    ranking = {}

    def screen(candidate):
        key = digest(candidate)
        if key not in ranking:
            try:
                result = train(
                    outputs,
                    cache,
                    panels,
                    candidate,
                    steps,
                    "screen",
                    schedule_horizon=schedule_horizon,
                )
                row = dict(
                    rates=candidate,
                    status=result["status"],
                    mean_gain=result["mean_gain"],
                )
            except (Catastrophe, Unmeasured) as error:
                row = dict(
                    rates=candidate,
                    status="fail" if isinstance(error, Catastrophe) else "unmeasured",
                    reason=str(error),
                )
            ranking[key] = row
            atomic_json(outputs.root / "rate-search.json", list(ranking.values()))
        return ranking[key]

    tried_fits = set()
    brackets = [rate_candidates(reference)] + [
        [reference | {"g": reference["g"] / n} for n in divisors]
        for divisors in ((4, 8), (16, 32))
    ]
    for bracket in brackets:
        observed = [screen(candidate) for candidate in bracket]
        measured = [r for r in observed if r["status"] == "pass"]
        if not measured:
            continue
        best = max(measured, key=lambda r: r["mean_gain"])["rates"]
        if best["g"] != reference["g"]:
            # Vary D and bank again where G is viable. A failed comparison
            # at the original G gives no measurement at the new G.
            for role in ("d", "particle"):
                for factor in (0.5, 2.0):
                    screen(best | {role: best[role] * factor})
            if len(observed) == 7 and len(measured) == 1 and best["g"] < reference["g"]:
                for factor in (0.5, 0.25):
                    screen(best | {"g": best["g"] * factor})
        eligible = [
            r
            for key, r in ranking.items()
            if r["status"] == "pass" and key not in tried_fits
        ]
        for row in sorted(eligible, key=lambda r: -r["mean_gain"]):
            tried_fits.add(digest(row["rates"]))
            yield row["rates"]


def fit_candidates(outputs, cache, panels, plan, args):
    rates = plan["rates"]
    if args.search_rates:
        yield from screened_rates(
            outputs, cache, panels, rates, plan["budget"], plan["horizon"]
        )
        return
    yield rates
    # The default command first uses the measured recipe. If it fails before
    # confirmation, expand the search in the same run and reuse its checkpoints.
    if all(getattr(args, f"{role}_lr") is None for role in REFERENCE):
        for candidate in screened_rates(
            outputs, cache, panels, rates, plan["budget"], plan["horizon"]
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
            selected = select_checkpoint(root, rates, steps)
            best = tune(
                outputs, cache, panels, selected, screened=screen_check is not None
            )
        except (Unmeasured, Catastrophe) as error:
            failures.append(dict(rates=rates, reason=str(error)))
            atomic_json(root / "fit-failures.json", failures)
            continue
        selection = dict(
            rates=rates,
            strength=best,
            selected_step=read(Path(best["raw"]["path"]).with_suffix(".scoped.json"))[
                "completed_steps"
            ],
        )
        atomic_json(reservation, selection)
        return selection
    raise Unmeasured(
        "No rate candidate completed with usable measurements; see fit-failures.json"
    )


def study_panels(spec, concept, root):
    groups = spec["validation"]
    extra = spec.get("screen_cases", {}).get(concept, [])
    if not extra:
        return groups, False
    bound = root / "data-config.json"
    if bound.exists():
        original = read(bound)
        old_spec = {k: v for k, v in spec.items() if k != "screen_cases"}
        if original["spec"] in (
            digest(dict(teachers=spec["concepts"][concept], validation=groups)),
            digest(old_spec),
        ) and any(case not in groups["rates"] for case in extra):
            return groups, False
    rates = list(groups["rates"])
    for case in extra:
        if case not in rates:
            rates.append(case)
    return groups | {"rates": rates}, True


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
        bound = root / "data-config.json"
        if bound.exists():
            original = read(bound)
            if original["schema"] == "anima-stock-empf-study-v1" and original[
                "spec"
            ] in (
                digest(spec),
                digest({k: v for k, v in spec.items() if k != "screen_cases"}),
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
        outputs = Outputs(root, rt, concept, config["effect"])
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
        outcome = root / "outcome.json"
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
        gallery(root, measured)
        result = measured["confirmation"]
        if result["status"] != "pass" or result["mean_gain"] <= 0:
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
            outcome=str(root / "outcome.json"),
        )
        atomic_json(root / "status.json", summary)
        print(json.dumps(summary))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
