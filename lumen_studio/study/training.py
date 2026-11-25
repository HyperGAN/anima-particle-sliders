"""Checkpointed stock EMPF training with effect-only selection and a separate failure gate."""

from dataclasses import asdict
import json
import math
from pathlib import Path
from uuid import uuid4

import torch
from lumen_studio_scoped import open_training, open_inference
from lumen_studio_scoped.training import Checkpoint, file_pin
from ..contracts import atomic_json, digest
from .outputs import Unmeasured, read


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
            for side in ("neutral", "positive"):
                embedding = runtime.encode(row["row"][side])
                prediction = runtime.predict(latent, row["timestep"], embedding)
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


def train(outputs, cache, panels, rates, horizon, phase):
    """A 1/4-horizon LR screen can be continued from exactly the committed FULL."""
    rt = outputs.runtime
    dest = arm_path(outputs.root, rates)
    cursor = dest / "cursor.json"
    if (dest / "failure.json").exists():
        raise Catastrophe(str(dest / "failure.json"))
    previous = read(cursor) if cursor.exists() else None
    variation = cache.index["identity"]["variation"]
    cut = horizon // 4
    milestones = (
        range(cut // 4, cut + 1, cut // 4)
        if phase == "screen"
        else range(cut, horizon + 1, cut)
    )
    end = cut if phase == "screen" else horizon

    # Record the to-be-published FULL path before entering the update. A crash
    # after its atomic publication can recover the actual completed update.
    intent = dest / "pending-update.json"
    recovered = False
    if intent.exists():
        pending = read(intent)
        if previous and previous["step"] == pending["step"]:
            intent.unlink()
        elif (
            previous
            and previous["step"] + 1 == pending["step"]
            and previous["full"] == pending["base"]
        ):
            full_path = Path(pending["output"])
            if (
                not full_path.resolve().is_relative_to((dest / "attempts").resolve())
                or full_path.name != f"FULL{pending['step']:06d}.pt"
            ):
                raise ValueError("Invalid pending checkpoint path")
            if not full_path.is_file():
                raise Unmeasured(
                    "Interrupted update has no atomically published FULL; its completion is unknown"
                )
            previous = dict(
                step=pending["step"], full=asdict(file_pin(full_path)), rates=rates
            )
            recovered = True
        else:
            raise ValueError("Pending update differs from committed cursor")

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
        mode="resume" if previous else "fresh",
        cohort_id="anima-study-"
        + digest(dict(rates=rates, cache=cache.index["fingerprint"])),
        resume=Checkpoint(**previous["full"]) if previous else None,
        preset="empf",
        seed=1,
        batch_size=2,
        total_steps=horizon,
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
            if owner.step in milestones:
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


def recheck_screened_cases(outputs, cache, panels, rates, horizon, fit):
    """Catch late regressions on the cases that originally qualified the rate."""
    arm = arm_path(outputs.root, rates)
    if not (arm / "screen" / f"{horizon // 4}.json").is_file():
        return
    raw = Checkpoint(**fit["raw"])
    receipt = arm / "fit-rate-panel.json"
    if receipt.is_file():
        result = read(receipt)
        if result["raw"] != asdict(raw) or file_pin(raw.path) != raw:
            raise ValueError("Full-budget rate check differs from its saved RAW")
    else:
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
            result = outputs.compare(
                panels["rates"], variation, raw, 1.0, "constant", owner=owner
            )
        finally:
            owner.close()
        atomic_json(receipt, result)
    if result["status"] == "fail":
        atomic_json(
            arm / "failure.json",
            dict(step=horizon, phase="fit-rate-panel", result=result),
        )
        raise Catastrophe(
            "A screened case developed a catastrophic defect at the full budget"
        )
    if result["status"] != "pass":
        raise Unmeasured(
            "Full-budget integrity on the original rate panel was unmeasured"
        )


def select_checkpoint(root, rates, horizon):
    directory = arm_path(root, rates) / "fit"
    rows = [
        read(directory / f"{step}.json")
        for step in range(horizon // 4, horizon + 1, horizon // 4)
    ]
    eligible = [r for r in rows if r["status"] == "pass"]
    if not eligible:
        raise Unmeasured("No checkpoint has measured intact outputs")
    return max(eligible, key=lambda row: row["mean_gain"])


def tune(outputs, cache, panels, selected):
    rt = outputs.runtime
    raw = Checkpoint(**selected["raw"])
    variation = cache.index["identity"]["variation"]
    owner = open_inference(
        rt,
        variation,
        raw=raw,
        owner_admission=lambda request: (
            request["model_identity"] == rt.identity and request["RAW"] == asdict(raw)
        ),
    )
    try:
        results = []
        for profile in ("constant", "early", "late"):
            for alpha in (0.5, 1.0, 1.5):
                result = outputs.compare(
                    panels["steps"], variation, raw, alpha, profile, owner=owner
                )
                results.append(result)
                atomic_json(outputs.root / "strengths.json", results)
        eligible = [r for r in results if r["status"] == "pass"]
        if not eligible:
            raise Unmeasured("No measured usable strength")
        best = max(eligible, key=lambda r: r["mean_gain"])
        confirm = outputs.compare(
            panels["confirm"],
            variation,
            raw,
            best["alpha"],
            best["profile"],
            owner=owner,
        )
        return best, confirm
    finally:
        owner.close()
