"""Checkpointed stock EMPF training with effect-only selection and a separate failure gate."""

from dataclasses import asdict
import fcntl
import json
import math
from pathlib import Path
from uuid import uuid4

import torch
from lumen_studio_scoped import open_training, open_inference
from lumen_studio_scoped.training import Checkpoint, file_pin
from ..contracts import atomic_json, digest
from .outputs import Unmeasured, read
from .objectives import DISTINCT_BOSS, MATCHED_GAIN, protocol_id, selection_score
from .scalar_search import search_constant, verify_search


class Catastrophe(RuntimeError):
    pass


@torch.no_grad()
def check_native_cache(runtime, cache, request):
    expected = dict(
        model_identity=runtime.identity,
        cache_fingerprint=cache.index["fingerprint"],
        response_shape=list(cache[0]["positive"].shape),
    )
    if request != expected:
        return False
    offset = 0
    with runtime.mixer.scales({}):
        for index, shard in enumerate(cache.index["shards"]):
            row = cache[offset + (index * 7 + 3) % shard["count"]]
            latent = row["latent"].to(runtime.device)
            neutral = None
            for side in ("neutral", "positive"):
                embedding = runtime.encode(row["row"][side])
                prediction = runtime.predict(latent, row["timestep"], embedding)
                if side == "neutral":
                    neutral = prediction
                elif cache.index["identity"].get("layout") == "final-boss/late-cues-v1":
                    from .boss_targets import field_context, target_field
                    context = field_context(runtime, cache.index["identity"]["field_recipe"], cache.directory,
                                            row["row"], latent.shape[-2:])
                    prediction = target_field(runtime, row, neutral, prediction, context)
                if not torch.equal(prediction.cpu(), row[side]):
                    return False
            offset += shard["count"]
    return True


def resolve_rates(game, expected):
    roles = {}
    for group in game.opt_g.param_groups:
        bank = any(parameter is game.prior.z for parameter in group["params"])
        if bank and len(group["params"]) != 1:
            raise ValueError(
                "Particle table shares an optimizer group with generator weights"
            )
        roles.setdefault("particle" if bank else "g", set()).add(group["lr"])
    roles["d"] = {group["lr"] for group in game.opt_d.param_groups}
    for role, rate in expected.items():
        if len(roles.get(role, ())) != 1 or not math.isclose(
            next(iter(roles[role])), rate, rel_tol=1e-12
        ):
            raise ValueError(f"Actual optimizer {role} rate differs from {rate}")
    return {name: next(iter(values)) for name, values in roles.items()}


def arm_path(root, rates):
    return root / "rates" / digest(rates)[:20]


def recover_pending(arm, previous):
    """Reconcile the one pending disk transaction while holding the arm's writer lock."""
    intent = arm / "pending-update.json"
    if not intent.exists():
        return previous, False
    pending = read(intent)
    step = pending["step"]
    output = Path(pending["output"])
    if (not output.resolve().is_relative_to((arm / "attempts").resolve())
            or output.name != f"FULL{step:06d}.pt"):
        raise ValueError("Invalid pending checkpoint path")
    if previous and previous["step"] == step:
        if asdict(file_pin(output)) != previous["full"]:
            raise ValueError("Pending transaction differs from the committed FULL")
        intent.unlink()
        return previous, False
    if (not previous or previous["step"] + 1 != step
            or previous["full"] != pending["base"]):
        raise ValueError("Pending update differs from committed cursor")
    receipt = arm / "updates" / f"{step:06d}.json"
    if output.is_file():
        pin = asdict(file_pin(output))
        if receipt.is_file():
            row = read(receipt)
            if row["step"] != step or row["full"] != pin or row["rates"] != previous["rates"]:
                raise ValueError("Pending FULL disagrees with its update receipt")
            if "raw" in row and asdict(file_pin(row["raw"]["path"])) != row["raw"]:
                raise ValueError("Recovered RAW differs from its update receipt")
            return row, True
        return dict(step=step, full=pin, rates=previous["rates"]), True
    if receipt.exists() or any(output.parent.glob(f"RAW{step:06d}.*")):
        raise Unmeasured("An update receipt or RAW exists but its FULL is missing")
    if asdict(file_pin(previous["full"]["path"])) != previous["full"]:
        raise ValueError("Last committed FULL differs before recovery")
    # Record the abandoned in-memory attempt with its original destination. A
    # fresh attempt starts at the exact committed model, optimizer, bank and RNG.
    # No completed checkpoint is replayed or counted twice.
    archive = output.parent / f"interrupted-{step:06d}.json"
    note = dict(status="no-published-FULL", in_memory_completion="unknown", intent=pending)
    if archive.exists() and read(archive) != note:
        raise ValueError("Conflicting interrupted transaction receipt")
    atomic_json(archive, note)
    intent.unlink()
    return previous, False


def train(
    outputs,
    cache,
    panels,
    rates,
    budget,
    phase,
    *,
    schedule_horizon=None,
    always_check_rates=False,
    observation_steps=(),
    parent=None,
    seed=1,
):
    """A 1/4-horizon LR screen can be continued from exactly the committed FULL.

    With ``parent``, the first entry forks that trained RAW into a separate cohort
    and the budget counts the additional updates. Later entries resume the child.
    """
    arm = arm_path(outputs.root, rates)
    arm.mkdir(parents=True, exist_ok=True)
    with (arm / "writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(outputs, cache, panels, rates, budget, phase, schedule_horizon,
                      always_check_rates, observation_steps, parent, seed)


def _train(outputs, cache, panels, rates, budget, phase, schedule_horizon,
           always_check_rates, observation_steps, parent=None, seed=1):
    rt = outputs.runtime
    dest = arm_path(outputs.root, rates)
    cursor = dest / "cursor.json"
    if (dest / "failure.json").exists():
        raise Catastrophe(str(dest / "failure.json"))
    previous = read(cursor) if cursor.exists() else None
    variation = cache.index["identity"]["variation"]
    cut = budget // 4
    milestones = (
        range(cut // 4, cut + 1, cut // 4)
        if phase == "screen"
        else range(cut, budget + 1, cut)
    )
    end = cut if phase == "screen" else budget
    if any(type(n) is not int or not 0 < n <= end for n in observation_steps):
        raise ValueError("Extra observation steps must be within the training budget")
    milestones = set(milestones).union(observation_steps)

    intent = dest / "pending-update.json"
    previous, recovered = recover_pending(dest, previous)

    def observe(step, raw, inference=None):
        result_path = dest / phase / f"{step}.json"
        if result_path.exists():
            result = read(result_path)
            if result["raw"] != asdict(raw) or file_pin(raw.path) != raw:
                raise ValueError("Saved observation differs from RAW")
        else:
            cases = panels["rates" if phase == "screen" else "steps"]
            result = outputs.compare(
                cases, variation, raw, 1.0, "constant", owner=inference
            )
            atomic_json(result_path, result)
            print(
                json.dumps(
                    dict(
                        phase=phase,
                        update=step,
                        budget=end,
                        rates=rates,
                        mean_gain=result["mean_gain"],
                        integrity=result["status"],
                    )
                ),
                flush=True,
            )
        if result["status"] == "fail":
            atomic_json(
                dest / "failure.json", dict(step=step, phase=phase, result=result)
            )
            raise Catastrophe(f"Catastrophic output at update {step}")
        if (
            phase == "fit"
            and step < end
            and (
                always_check_rates
                or (cut < step and (dest / "screen" / f"{cut}.json").is_file())
            )
        ):
            # A screened scene can break earlier than the separate step panel.
            # Check while the current weights are still installed, before the next update.
            score_rate_panel(
                dest,
                dest / "fit-screen" / f"{step}.json",
                step,
                raw,
                lambda: outputs.compare(
                    panels["rates"], variation, raw, 1.0, "constant", owner=inference
                ),
            )
        return result

    last_result = dest / phase / f"{end}.json"
    if previous and previous["step"] >= end and last_result.exists():
        result = read(last_result)
        if file_pin(result["raw"]["path"]) != Checkpoint(**result["raw"]):
            raise ValueError(f"Retained RAW differs at {last_result}")
        return result
    if previous and not recovered and previous["step"] >= end:
        # A completed single-rate fit can later enter the optional rate screen.
        # Render its committed quarter RAW on the rate panel without replaying an update.
        row = read(dest / "updates" / f"{end:06d}.json")
        raw = Checkpoint(**row["raw"])
        inference = open_inference(
            rt,
            variation,
            raw=raw,
            owner_admission=lambda req: (
                req["RAW"] == asdict(raw) and req["model_identity"] == rt.identity
            ),
        )
        try:
            return observe(end, raw, inference)
        finally:
            inference.close()
    directory = dest / "attempts" / uuid4().hex
    owner = open_training(
        rt,
        cache,
        directory,
        variation,
        mode="resume" if previous else "fork" if parent else "fresh",
        cohort_id="anima-study-"
        + digest(dict(rates=rates, cache=cache.index["fingerprint"],
                      **(dict(parent=parent.sha256) if parent else {}),
                      **(dict(seed=seed) if seed != 1 else {}))),
        resume=Checkpoint(**previous["full"]) if previous else None,
        fork=None if previous else parent,
        preset="empf",
        seed=seed,
        batch_size=2,
        total_steps=schedule_horizon or budget,
        critic_width=32,
        ema_decay=None,
        lr=rates["g"],
        d_lr_mult=rates["d"] / rates["g"],
        prior_lr_mult=rates["particle"] / rates["g"],
        native_admission=lambda req: check_native_cache(rt, cache, req),
    )
    try:
        if (
            owner.game.recipe.effective_critic_formulation != "empf"
            or owner.game.recipe.reg_every != 1
        ):
            raise ValueError("Unexpected stock EMPF recipe")
        actual = resolve_rates(owner.game, rates)
        if previous is None:
            previous = dict(
                step=0,
                full=asdict(owner.entry_FULL),
                raw=asdict(owner.entry_RAW),
                rates=actual,
            )
            atomic_json(cursor, previous)
        if owner.step != previous["step"]:
            raise ValueError("Restored cursor differs from committed FULL")
        if recovered:
            if owner.step in milestones and "raw" not in previous:
                previous["raw"] = asdict(owner.entry_RAW)
            atomic_json(dest / "updates" / f"{owner.step:06d}.json", previous)
            atomic_json(cursor, previous)
            intent.unlink()
        if owner.step in milestones:
            observe(owner.step, Checkpoint(**previous["raw"]))
        while owner.step < end:
            step = owner.step + 1
            atomic_json(
                intent,
                dict(
                    step=step,
                    base=previous["full"],
                    output=str(directory / f"FULL{step:06d}.pt"),
                ),
            )
            metrics = owner.update()
            if metrics["step"] != previous["step"] + 1 or owner.step != metrics["step"]:
                raise ValueError("Native optimizer update count differs")
            current = dict(
                step=owner.step,
                full=asdict(owner.save()),
                rates=actual,
                metrics=metrics,
            )
            if owner.step in milestones:
                raw = owner.export(directory / f"RAW{owner.step:06d}.safetensors")
                current["raw"] = asdict(raw)
            atomic_json(dest / "updates" / f"{owner.step:06d}.json", current)
            atomic_json(cursor, current)
            intent.unlink()
            # A committed FULL with an observed RAW is durable. Keep ordinary
            # recovery storage bounded by retiring the preceding transient FULL.
            old = Path(previous["full"]["path"])
            if "raw" not in previous and old.is_relative_to(dest / "attempts"):
                old.unlink(missing_ok=True)
            previous = current
            if owner.step in milestones:
                observe(owner.step, raw)
    finally:
        owner.close()
    return read(last_result)


def rate_candidates(reference):
    yield reference
    for role in ("g", "d", "particle"):
        for factor in (0.5, 2.0):
            yield reference | {role: reference[role] * factor}


def score_rate_panel(arm, receipt, step, raw, measure):
    if receipt.is_file():
        result = read(receipt)
        if result["raw"] != asdict(raw) or file_pin(raw.path) != raw:
            raise ValueError("Screening scene check differs from its saved RAW")
    else:
        result = measure()
        atomic_json(receipt, result)
    if result["status"] == "fail":
        atomic_json(
            arm / "failure.json", dict(step=step, phase="fit-rate-panel", result=result)
        )
        raise Catastrophe(
            f"A screened case developed a catastrophic defect at update {step}"
        )
    return result


def recheck_screened_cases(
    outputs, cache, panels, rates, horizon, fit, *, required=False
):
    """Check the original screening scenes at the full training budget."""
    arm = arm_path(outputs.root, rates)
    if not required and not (arm / "screen" / f"{horizon // 4}.json").is_file():
        return
    raw = Checkpoint(**fit["raw"])

    def measure():
        rt = outputs.runtime
        variation = cache.index["identity"]["variation"]
        owner = open_inference(
            rt,
            variation,
            raw=raw,
            owner_admission=lambda request: (
                request["model_identity"] == rt.identity
                and request["RAW"] == asdict(raw)
            ),
        )
        try:
            return outputs.compare(
                panels["rates"], variation, raw, 1.0, "constant", owner=owner
            )
        finally:
            owner.close()

    result = score_rate_panel(arm, arm / "fit-rate-panel.json", horizon, raw, measure)
    if result["status"] != "pass":
        raise Unmeasured(
            "Full-budget integrity on the original rate panel was unmeasured"
        )

    return result


def eligible_checkpoints(root, rates, horizon):
    directory = arm_path(root, rates) / "fit"
    eligible = []
    for step in range(horizon // 4, horizon + 1, horizon // 4):
        row = read(directory / f"{step}.json")
        screened = directory.parent / "fit-screen" / f"{step}.json"
        if row["status"] == "pass" and (
            not screened.exists() or read(screened)["status"] == "pass"
        ):
            eligible.append(row)
    if not eligible:
        raise Unmeasured("No checkpoint has measured intact outputs")
    return eligible


def select_checkpoint(root, rates, horizon, *, outputs=None):
    rows = eligible_checkpoints(root, rates, horizon)
    if outputs is not None:
        rows = [outputs.with_objective(row) for row in rows]
    scored = [r for r in rows if selection_score(r) is not None]
    if not scored:
        raise Unmeasured("No qualified checkpoint has a complete selection objective")
    return max(scored, key=selection_score)


def select_strength(results, receipt, recheck=None):
    """Rank by effect, excluding settings that regress on earlier screening cases."""
    eligible = sorted(
        (r for r in results if r["status"] == "pass"), key=lambda r: -r["mean_gain"]
    )
    for candidate in eligible:
        if recheck is not None:
            evidence = recheck(candidate)
            candidate["rate_panel"] = evidence
            candidate["status"] = evidence["status"]
            atomic_json(receipt, results)
        if candidate["status"] == "pass":
            return candidate
    raise Unmeasured("No measured usable strength")


def refine_strength(raw, cases, evaluate, strongest, *, anchor=None):
    """Refine both sides of an intact leader and expand a rising upper edge."""
    from particle_sliders.alpha_search import AlphaObservation, search_alpha
    from particle_sliders.candidate_failure import FileRecord

    ids = tuple(digest([case["prompt"], case["seed"]]) for case in cases)
    grid = [0.5, 0.75, 1.0, 1.25, 1.5]
    if anchor is not None:
        grid = sorted(set(grid + [anchor]))

    def observe(alpha):
        row = evaluate(alpha)
        return {
            digest([pair["case"]["prompt"], pair["case"]["seed"]]): AlphaObservation(
                pair["case"]["off"]["judgment"]["score"],
                pair["on"]["judgment"]["score"],
                row["status"],
            )
            for pair in row["pairs"]
        }

    # A measured plateau needs no extension. An unmeasured upper edge is a
    # bracket boundary; its interval can still contain an intact, stronger edit.
    while grid[0] > 0.03125:
        first = evaluate(grid[0])
        others = [
            row["mean_gain"]
            for alpha in grid[1:]
            if (row := evaluate(alpha))["status"] == "pass"
        ]
        if others and (first["status"] != "pass" or first["mean_gain"] <= max(others)):
            break
        grid.insert(0, grid[0] / 2)

    while grid[-1] < strongest:
        edge = evaluate(grid[-1])
        lower = [
            row["mean_gain"]
            for alpha in grid[:-1]
            if (row := evaluate(alpha))["status"] == "pass"
        ]
        if edge["status"] != "pass" or (lower and edge["mean_gain"] <= max(lower)):
            break
        grid.append(min(strongest, grid[-1] * 1.5))
    search_alpha(
        checkpoint=FileRecord(**asdict(raw)),
        case_ids=ids,
        alphas=grid,
        evaluate=observe,
        refinement_rounds=2,
    )


def tune(outputs, cache, panels, selected, *, screened=False, receipt=None):
    """Calibrate one constant alpha on a single frozen RAW and fixed selection cases."""
    rt = outputs.runtime
    raw = Checkpoint(**selected["raw"])
    variation = cache.index["identity"]["variation"]
    owner = open_inference(
        rt, variation, raw=raw,
        owner_admission=lambda request: (
            request["model_identity"] == rt.identity and request["RAW"] == asdict(raw)
        ),
    )
    try:
        return calibrate_constant(outputs, panels, variation, raw, screened=screened, owner=owner, receipt=receipt)
    finally:
        owner.close()


def calibrate_constant(outputs, panels, variation, raw, *, screened=False, owner=None, receipt=None):
    """Select one constant strength for an already loaded, verified RAW."""
    receipt = receipt or outputs.root / "strengths.json"
    if receipt.exists() and any(r["profile"] != "constant" for r in read(receipt)):
        archived = receipt.with_name(receipt.stem + "-before-constant.json")
        if not archived.exists():
            atomic_json(archived, read(receipt))
    measured = {}
    objective = getattr(outputs, "objective", MATCHED_GAIN)

    def evaluate(alpha):
        if alpha not in measured:
            row = outputs.compare(panels["steps"], variation, raw, alpha, "constant", owner=owner)
            if (row["raw"] != asdict(raw) or row["profile"] != "constant"
                    or row["alpha"] != alpha or row["gains"] != [alpha] * 10):
                raise ValueError("Alpha selection requires the pinned RAW and ten identical gains")
            measured[alpha] = row
            atomic_json(receipt, list(measured.values()))
        return measured[alpha]

    if {c["prompt"] for c in panels["steps"]} & {c["prompt"] for c in panels.get("confirm", [])}:
        raise ValueError("Alpha search and confirmation prompts overlap")
    trials = search_constant(lambda a: selection_score(evaluate(a), objective))
    available = [a for a in trials if trials[a] is not None]
    if not available:
        raise Unmeasured(f"No complete alpha measurements for {objective}")
    alpha = max(available, key=lambda a: (trials[a], -a))
    verify_search(list(trials.items()), alpha)
    selected = measured[alpha]
    if screened and selected["status"] == "pass":
        panel = outputs.compare(panels["rates"], variation, raw, alpha, "constant", owner=owner)
        selected = dict(selected, selection_status=selected["status"], rate_panel=panel, status=panel["status"])
        measured[alpha] = selected
        atomic_json(receipt, list(measured.values()))
    reservation = dict(objective=objective, protocol=protocol_id() if objective == DISTINCT_BOSS else None,
                       raw_sha256=raw.sha256, cases_sha256=digest(panels["steps"]),
                       trials=[[a, value] for a, value in trials.items()], alpha=alpha)
    bound = receipt.with_name(receipt.stem + "-reservation.json")
    if bound.exists() and read(bound) != reservation:
        raise ValueError("Scalar search differs from its reserved objective, RAW or case panel")
    atomic_json(bound, reservation)
    return selected


def validate_calibration(root, strength, cases, outputs):
    """Refuse a held-out render when its scalar was selected with another metric or RAW."""
    objective = outputs.objective
    proof = read(root / "strengths-reservation.json")
    policy = dict(objective=objective, protocol=protocol_id() if objective == DISTINCT_BOSS else None,
                  raw_sha256=strength["raw"]["sha256"], cases_sha256=digest(cases))
    if any(proof.get(key) != value for key, value in policy.items()):
        raise ValueError("Reserved alpha does not belong to this study objective, RAW and panel")
    record = []
    for trial in read(root / "strengths.json"):
        alpha = trial["alpha"]
        if (trial["raw"] != strength["raw"] or trial["profile"] != "constant"
                or trial["gains"] != [alpha] * 10 or [p["case"] for p in trial["pairs"]] != cases):
            raise ValueError("Scalar search contains different weights, intervals or cases")
        value = selection_score(trial, objective)
        if objective == DISTINCT_BOSS and selection_score(outputs.with_objective(trial), objective) != value:
            raise ValueError("Stored trial score differs from its bound design observations")
        record.append([alpha, value])
    if record != proof["trials"] or proof["alpha"] != strength["alpha"]:
        raise ValueError("Held-out selection differs from the scalar search reservation")
    verify_search(record, strength["alpha"])
    if (strength["profile"] != "constant" or strength["gains"] != [strength["alpha"]] * 10
            or dict(record)[strength["alpha"]] != selection_score(strength, objective)):
        raise ValueError("Reserved strength or objective differs from the winning search trial")


def measure_strength(outputs, cache, cases, strength):
    rt = outputs.runtime
    variation = cache.index["identity"]["variation"]
    raw = Checkpoint(**strength["raw"])
    owner = open_inference(
        rt,
        variation,
        raw=raw,
        owner_admission=lambda request: (
            request["model_identity"] == rt.identity and request["RAW"] == asdict(raw)
        ),
    )
    try:
        return outputs.compare(
            cases,
            variation,
            raw,
            strength["alpha"],
            strength["profile"],
            owner=owner,
        )
    finally:
        owner.close()


def confirm(outputs, cache, panels, strength):
    return measure_strength(outputs, cache, panels["confirm"], strength)
