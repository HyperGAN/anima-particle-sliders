"""Fit the admitted stock EMPF game on frozen, Jev-qualified Anima teachers."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
import shutil

from design import ROOT, save
from judge import compare
from native import guarded, pin, read, render

VARIATION = "character-final-boss"
COHORT = "anima-research-raw192-foreground-v1"
BUDGET = 192
MILESTONES = (48, 96, 144, 192)


def holdouts():
    return read(ROOT / "benchmark-root/panel.json")


def render_holdouts(runtime):
    for row in holdouts():
        render(runtime, row["prompt"], row["seed"], ROOT / "heldout" / row["label"] / "off")
        print(f"Saved held-out OFF {row['label']}", flush=True)


def freeze_cache(runtime):
    import torch
    from lumen_studio.cache import TargetCache, save_tensor_file
    from lumen_studio.contracts import digest

    folder = ROOT / "target-cache"
    if (folder / "index.json").exists():
        cache = TargetCache(folder, verify=True, max_shards=4)
        if cache.index["identity"]["model"] != runtime.identity:
            raise ValueError("Frozen cache belongs to another runtime")
        return cache
    entries = read(ROOT / "localized-judgments.json")["selected"]
    if not 4 <= len(entries) <= 8:
        raise ValueError("Need at least four independently qualified scene pairs")
    teachers = []
    for entry in sorted(entries, key=lambda x: x["row"]["id"]):
        manifest = read(ROOT / "localized" / entry["row"]["id"] / "target.json")
        if (manifest["model"] != runtime.identity or manifest["row"] != entry["row"] or
                [manifest["neutral"], manifest["image"]] != entry["images"] or
                pin(Path(manifest["fields"]["path"])) != manifest["fields"]):
            raise ValueError("Selected teacher differs from the frozen render or judge evidence")
        teachers.append(manifest)
    identity = dict(model=runtime.identity, split="train", variation=VARIATION,
                    resolution=512, steps=10, cfg=1,
                    layout="both-trajectories/masked-positive-velocity/shared-state-v1",
                    manifest_sha256=digest([m["row"] for m in teachers]),
                    selected_manifest=pin(ROOT / "localized-judgments.json"), teachers=teachers)
    fingerprint = digest(identity)
    folder.mkdir(parents=True, exist_ok=True)
    shards = []
    for teacher in teachers:
        data = torch.load(teacher["fields"]["path"], map_location="cpu", weights_only=True)
        if (data["row"]["variation"] != VARIATION or data["seed"] != teacher["row"]["seed"] or
                {(x["trajectory"], x["position"]) for x in data["records"]} !=
                {(track, position) for track in ("neutral", "positive") for position in range(10)}):
            raise ValueError("Teacher tensor cache has unexpected bindings")
        for record in data["records"]:
            outside = (data["mask"] == 0).expand_as(record["positive"])
            if not torch.equal(record["positive"][outside], record["neutral"][outside]):
                raise ValueError("Teacher changes response outside its allowed foreground")
        data["fingerprint"] = fingerprint
        path = folder / f"{data['row']['id']}-{data['seed']}.pt"
        if path.exists():
            previous = torch.load(path, map_location="cpu", weights_only=True)
            if previous["fingerprint"] != fingerprint or previous["row"] != data["row"]:
                raise ValueError("Existing qualified shard comes from different targets")
            for old, new in zip(previous["records"], data["records"], strict=True):
                if any(not torch.equal(old[key], new[key]) for key in ("latent", "neutral", "positive")):
                    raise ValueError("Existing qualified shard differs from selected source")
        else:
            save_tensor_file(path, data)
        shards.append(dict(path=path.name, sha256=pin(path)["sha256"], count=20,
                           row=data["row"]["id"], seed=data["seed"]))
    save(folder / "index.json", dict(identity=identity, input_fingerprint=fingerprint,
                                   fingerprint=digest(dict(input=fingerprint, shards=shards)), shards=shards))
    return TargetCache(folder, verify=True, max_shards=4)


def parity(runtime, cache, request):
    import torch
    from fit import mask_for

    expected = dict(model_identity=runtime.identity, cache_fingerprint=cache.index["fingerprint"],
                    response_shape=list(cache[0]["positive"].shape))
    if request != expected:
        return False
    # Verify both paths on separated native steps using frozen owned teachers.
    teacher = cache.index["identity"]["teachers"][0]
    errors = []
    with torch.no_grad(), runtime.mixer.scales({}):
        neutral = runtime.encode(teacher["row"]["neutral"])
        positive = runtime.encode(teacher["row"]["positive"])
        for index in (3, 17):
            row = cache[index]
            latent = row["latent"].to(runtime.device)
            baseline = runtime.predict(latent, row["timestep"], neutral)
            edited = runtime.predict(latent, row["timestep"], positive)
            mask = mask_for(Path(teacher["mask"]["path"]), latent.shape[-2:], runtime.device)
            target = baseline + mask * (edited - baseline)
            errors.append(dict(index=index,
                neutral=float((baseline.cpu() - row["neutral"]).abs().max()),
                localized=float((target.cpu() - row["positive"]).abs().max())))
    passed = all(item["neutral"] == item["localized"] == 0. for item in errors)
    evidence = ROOT / "teacher-native-parity.json"
    report = dict(passed=passed, errors=errors, request=request)
    if not evidence.exists():
        save(evidence, report)
    elif read(evidence) != report:
        raise RuntimeError("Native parity no longer matches its first result")
    return passed


def schedules(runtime):
    from particle_sliders.strength_schedule import interval_strengths
    sigmas = [float(x) for x in runtime.scheduler(10).sigmas.tolist()]
    widths = [abs(b - a) for a, b in zip(sigmas, sigmas[1:])]
    age = 2 / 3
    arms = dict(constant_dose=[age] * 10,
                early=list(interval_strengths(age, widths, profile="early")),
                late=list(interval_strengths(age, widths, profile="late")))
    report = dict(sigmas=sigmas, strengths=arms, ema_decay=None, weights="raw")
    path = ROOT / "schedules.json"
    if path.exists() and read(path) != report:
        raise ValueError("Schedule changed")
    if not path.exists():
        save(path, report)
    return arms


def check_checkpoint(runtime, step, arms, variation):
    evidence = ROOT / "milestones" / f"RAW{step:06d}.json"
    if evidence.exists():
        return read(evidence)
    candidates = {}
    for profile, gains in arms.items():
        candidates[profile] = []
        for row in (holdouts() if step == BUDGET else holdouts()[:4]):
            folder = ROOT / "heldout" / row["label"]
            off = read(folder / "off" / "render.json")["image"]
            on = render(runtime, row["prompt"], row["seed"], folder / f"raw{step:06d}-{profile}",
                        gains=gains, variation=variation)
            candidates[profile].append((row, off, on))

    def score(task):
        tag, row, off, on = task
        images = [Path(off["path"]), Path(on["path"])]
        if [pin(path) for path in images] != [off, on]:
            raise ValueError("Matched checkpoint outputs changed before curation")
        result = compare(tag, images, row["background"], row["lighting"],
                         character=row["person"].removeprefix("adult "))
        return dict(row=row, result=result, off=off, on=on)

    table = {}
    with ThreadPoolExecutor(2) as pool:
        for profile, pairs in candidates.items():
            tasks = [(f"raw{step:06d}-{profile}-{row['label']}-v1", row, off, on) for row, off, on in pairs]
            outcomes = list(pool.map(score, tasks))
            fail = any(max(entry["result"]["catastrophic_probability"]) >= .8 for entry in outcomes)
            unknown = any(.2 < probability < .8 for entry in outcomes
                          for probability in entry["result"]["catastrophic_probability"])
            table[profile] = dict(cases=outcomes, mean_gain=fmean(row["result"]["gain"] for row in outcomes),
                mean_background=fmean(row["result"]["background_same"] for row in outcomes),
                mean_lighting=fmean(row["result"]["lighting_same"] for row in outcomes),
                status="fail" if fail else "unmeasured" if unknown else "pass")
    report = dict(step=step, progress_percent=100 * step // BUDGET, arms=table)
    save(evidence, report)
    print(f"{report['progress_percent']}% RAW{step}: " + ", ".join(
        f"{key} {value['mean_gain']:+.5f} ({value['status']})" for key, value in table.items()), flush=True)
    return report


def train(runtime, attempt):
    from lumen_studio_scoped import open_training
    from lumen_studio_scoped.training import Checkpoint

    guarded()
    calibration = read(ROOT / "jev-integrity-calibration.json")
    if not calibration["qualified"]:
        raise RuntimeError("Jev did not qualify on the frozen catastrophic fixtures")
    cache = freeze_cache(runtime)
    root = ROOT / "training"
    pending = root / "pending-update.json"
    if pending.exists():
        raise RuntimeError("Unreconciled in-flight update. Await exact artifact reconciliation.")
    records = sorted((root / "commits").glob("*.json"))
    states = [read(record) for record in records]
    if [s["step"] for s in states] != list(range(len(states))):
        raise ValueError("Training journal has a gap")
    resume = None if not states else Checkpoint(**states[-1]["full"])
    for milestone in (step for step in MILESTONES if states and step <= states[-1]["step"]):
        p = ROOT / "milestones" / f"RAW{milestone:06d}.json"
        if not p.exists():
            break
        if any(arm["status"] == "fail" for arm in read(p)["arms"].values()):
            raise RuntimeError("Checkpoint has a previously verified catastrophic failure")
    trainer = open_training(runtime, cache, root / f"run-{attempt}", VARIATION,
        mode="fresh" if resume is None else "resume", cohort_id=COHORT, resume=resume, seed=1,
        batch_size=2, total_steps=BUDGET, ema_decay=None, lr=.00075, d_lr_mult=12., prior_lr_mult=40.,
        native_admission=lambda request: parity(runtime, cache, request))
    try:
        recipe = trainer.game.recipe
        if recipe.effective_critic_formulation != "empf" or recipe.reg_every != 1:
            raise RuntimeError("Stock EMPF with lazy_k=1 required")
        if not states:
            states.append(dict(step=0, full=asdict(trainer.entry_FULL), raw=asdict(trainer.entry_RAW)))
            save(root / "commits/000000.json", states[0])
        elif trainer.step != states[-1]["step"]:
            raise RuntimeError("Training did not restore the last committed update")
        # Run the product's native admission before opening the first update journal.
        trainer._admit_native_update()
        if trainer.step in MILESTONES:
            check = check_checkpoint(runtime, trainer.step, {"early": schedules(runtime)["early"]}, VARIATION)
            if any(arm["status"] == "fail" for arm in check["arms"].values()):
                raise RuntimeError("Catastrophic output failure")
        while trainer.step < BUDGET:
            guarded()
            if shutil.disk_usage(root).free < 8 * 2**30:
                raise ValueError("Experiment volume reached its free-space floor")
            step = trainer.step + 1
            save(pending, dict(step=step, base=states[-1]["full"], attempt=attempt))
            metrics = trainer.update()
            if metrics["step"] != step or trainer.step != step:
                raise RuntimeError("Training clock differs from actual update count")
            full = trainer.save()
            record = dict(step=step, full=asdict(full), metrics=metrics)
            if step in MILESTONES:
                raw = trainer.export(trainer.directory / f"RAW{step:06d}.safetensors")
                record["raw"] = asdict(raw)
            save(root / "commits" / f"{step:06d}.json", record)
            states.append(record)
            pending.unlink()
            retired = step - 2
            if retired > 0 and retired not in MILESTONES:
                previous = Path(states[retired]["full"]["path"])
                if not previous.is_relative_to(root) or previous.name != f"FULL{retired:06d}.pt":
                    raise ValueError("Rolling checkpoint is outside the owned training directory")
                previous.unlink(missing_ok=True)
            if step in MILESTONES:
                profiles = schedules(runtime)
                arms = profiles if step == BUDGET else dict(early=profiles["early"])
                report = check_checkpoint(runtime, step, arms, VARIATION)
                if any(arm["status"] == "fail" for arm in report["arms"].values()):
                    raise RuntimeError("Catastrophic output failure")
        usable = []
        for step in (BUDGET,):
            report = read(ROOT / "milestones" / f"RAW{step:06d}.json")
            usable.extend((arm["mean_gain"], step, name, arm) for name, arm in report["arms"].items()
                          if arm["status"] == "pass" and name == "early")
        if not usable:
            raise RuntimeError("Completed fits require a qualified held-out RAW result")
        maximum = max(x[0] for x in usable)
        winners = [dict(step=step, profile=profile, result=result, raw=states[step]["raw"])
                   for gain, step, profile, result in usable if gain == maximum]
        save(ROOT / "result.json", dict(cohort=COHORT, actual_updates=trainer.step, rows=len(cache) // 20,
                                     ema_decay=None, weights="raw", best_gain=maximum, equal_best=winners))
        print(f"Completed {trainer.step} updates; best held-out effect {maximum:+.5f}", flush=True)
    finally:
        trainer.close()


def run(runtime, stage, attempt):
    if stage == "holdout":
        render_holdouts(runtime)
    elif stage == "train":
        train(runtime, attempt)
    else:
        raise ValueError(f"Unknown stage {stage}")
