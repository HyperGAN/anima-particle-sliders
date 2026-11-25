"""Captured ParticleGAN harness for the Final Boss target search."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "anima-final-boss-research-raw192-20261123"
PRODUCT = PRIOR / "cohorts/prompt-only"
STORAGE = Path("/mnt/nvme/sliders/experiments/anima-final-boss-optimization-20261123")
BASE_CACHE = PRIOR / "caches/prompt-only"
from optimization_targets import METHODS, apply_target, donor_cache, encode_context, plan, sources
QUESTION = "How well does the candidate convey ‘Final Boss’ without breaking the image?"
BUDGET = 192
MILESTONES = (48, 96, 144, 192)
VARIATION = "character-final-boss"
SCREEN = ("archivist", "bookkeeper", "botanist", "seamstress")
sys.path.append(str(PRODUCT))
from native import guarded, pin, render


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".partial")
    with pending.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    pending.replace(path)


def panel():
    return read(PRIOR / "panel.json")


def options():
    raw = read(PRODUCT / "result.json")["equal_best"][0]["raw"]
    sidecar = read(Path(raw["path"]).with_suffix(".scoped.json"))
    settings = sidecar["options"]
    if settings["ema_decay"] is not None or sidecar["selection"] != "raw":
        raise ValueError("Control training did not use RAW weights")
    return settings


def attach_judge():
    import particle_sliders

    # The original training closure remains in use. Stage only the checked
    # alpha-tune observer that is already used by the Prompt-only leaderboard.
    staged = PRIOR / "prompt-only-alpha-tune/pr-wheel-visible"
    for name, expected in read(staged / "module-hashes.json").items():
        if pin(staged / "particle_sliders" / name)["sha256"] != expected:
            raise ValueError("The Jev observer differs from the verified module")
    particle_sliders.__path__.append(str(staged / "particle_sliders"))
    sys.path.append(str(PRIOR))
    from final_boss_outcome import PROMPTS, observer

    if PROMPTS["direct-final-boss"][0] != QUESTION:
        raise ValueError("The evaluation question differs from the agreed prompt")
    return observer()


def paired_source(method):
    return donor_cache(method)


def prepare_cache(runtime, method):
    import torch
    from lumen_studio.cache import TargetCache, save_tensor_file
    from lumen_studio.contracts import digest

    directory = ROOT / "cohorts" / method / "target-cache"
    original = TargetCache(BASE_CACHE, verify=True)
    if original.index["identity"]["model"] != runtime.identity or len(original) != 120:
        raise ValueError("The full-frame cache does not match the native runtime")
    base_rows = {item["row"]["id"]: item["row"] for item in original.index["identity"]["teachers"]}
    donor = paired_source(method)
    augmented = {
        "parent": original.index["fingerprint"], "layout": f"blended-full-frame/{method}/v1",
        "implementation": [pin(Path(__file__).resolve()), pin(ROOT / "optimization_targets.py"), pin(ROOT / "plans" / f"{method}.json")],
        "recipe": plan(method),
        "donor_source": None if donor is None else donor.index["fingerprint"],
        "sources": sources(base_rows),
    }
    # Round-trip tuples to JSON before checking exact identity on resume.
    identity = json.loads(json.dumps(original.index["identity"] | augmented))
    input_sha = digest(identity)
    if (directory / "index.json").is_file():
        cache = TargetCache(directory, verify=True)
        if cache.index["identity"] != identity or len(cache) != len(original):
            raise ValueError("A different transformed cache is already pinned")
        return cache
    directory.mkdir(parents=True, exist_ok=True)
    shards, diagnostics = [], []
    offset = 0
    with torch.no_grad(), runtime.mixer.scales({}):
        for shard in original.index["shards"]:
            guarded()
            parent = torch.load(BASE_CACHE / shard["path"], map_location="cpu", weights_only=True)
            if len(parent["records"]) != 20 or parent["fingerprint"] != original.index["input_fingerprint"]:
                raise ValueError("Expect ten steps on both full-frame trajectories")
            context = encode_context(runtime, base_rows[parent["row"]["id"]], method, parent["records"][0]["latent"].shape[-2:])
            fields = []
            for record in parent["records"]:
                guarded()
                latent, neutral, positive = (record[name].to(runtime.device)
                                             for name in ("latent", "neutral", "positive"))
                target_direction = None
                if donor is not None:
                    direction = donor[offset]
                    if (direction["row"] != parent["row"] or direction["timestep"] != record["timestep"]
                            or not torch.equal(direction["latent"], record["latent"])
                            or not torch.equal(direction["neutral"], record["neutral"])):
                        raise ValueError("Donor contrast is not bound to this same native state")
                    target_direction = direction["positive"].to(runtime.device)
                target, details = apply_target(context, neutral, positive, target_direction)
                if not torch.isfinite(target).all():
                    raise ValueError("Nonfinite teacher response")
                fields.append(record | {"positive": target.cpu()})
                diagnostics.append({"scene": parent["row"]["id"], "position": record["position"],
                                    "trajectory": record["trajectory"], "source_l2": (positive-neutral).norm().item(),
                                    "target_l2": (target-neutral).norm().item(), **details})
                offset += 1
            path = directory / shard["path"]
            data = parent | {"records": fields, "fingerprint": input_sha}
            if path.exists():
                previous = torch.load(path, map_location="cpu", weights_only=True)
                if (previous["fingerprint"] != input_sha or any(
                    not torch.equal(old["positive"], new["positive"])
                    for old, new in zip(previous["records"], fields, strict=True)
                )):
                    raise ValueError(f"Incomplete cache has different fields: {path}")
            else:
                save_tensor_file(path, data)
            shards.append(shard | {"sha256": pin(path)["sha256"]})
            print(f"{method} teacher cache: {parent['row']['id']} verified", flush=True)
    if offset != len(original):
        raise ValueError("Derived cache did not consume all shared states")
    write(directory / "diagnostics.json", diagnostics)
    write(directory / "index.json", {"identity": identity, "input_fingerprint": input_sha,
                                     "shards": shards, "fingerprint": digest({"input": input_sha, "shards": shards})})
    return TargetCache(directory, verify=True)


def parity(runtime, method, cache, request):
    import torch
    from lumen_studio.cache import TargetCache

    expected = {"model_identity": runtime.identity, "cache_fingerprint": cache.index["fingerprint"],
                "response_shape": list(cache[0]["positive"].shape)}
    if request != expected:
        return False
    original = TargetCache(BASE_CACHE, verify=True)
    base_rows = {item["row"]["id"]: item["row"] for item in original.index["identity"]["teachers"]}
    donor = paired_source(method)
    evidence = []
    with torch.no_grad(), runtime.mixer.scales({}):
        for index in (3, 19, 46, 117):
            source, current = original[index], cache[index]
            latent = source["latent"].to(runtime.device)
            ctx = encode_context(runtime, base_rows[source["row"]["id"]], method, latent.shape[-2:])
            neutral = runtime.predict(latent, source["timestep"], ctx["neutral"])
            positive = runtime.predict(latent, source["timestep"], ctx["positive"])
            inherited_target = neutral + (positive - neutral)
            target_direction = None if donor is None else donor[index]["positive"].to(runtime.device)
            target, detail = apply_target(ctx, neutral, inherited_target, target_direction)
            checks = [torch.equal(neutral.cpu(), source["neutral"]),
                      torch.equal(inherited_target.cpu(), source["positive"]),
                      torch.equal(latent.cpu(), current["latent"]),
                      torch.equal(target.cpu(), current["positive"])]
            evidence.append({"index": index, "checks": checks,
                             "max_target_difference": (target.cpu() - current["positive"]).abs().max().item()})
    success = all(all(item["checks"]) for item in evidence)
    write(ROOT / "cohorts" / method / "native-parity.json", {"passed": success, "request": request, "fields": evidence})
    return success


def gains(runtime, alpha, profile):
    from particle_sliders.strength_schedule import interval_strengths

    sigmas = runtime.scheduler(10).sigmas.tolist()
    result = list(interval_strengths(alpha, [b - a for a, b in zip(sigmas, sigmas[1:])], profile=profile))
    if max(result) > 5:
        raise ValueError("Requested strength exceeds native slider limits")
    return result


def check(runtime, method, step, observe):
    path = ROOT / "cohorts" / method / "milestones" / f"RAW{step:06d}.json"
    if path.exists():
        return read(path)
    concept, integrity = observe
    strength = gains(runtime, 2/3, "early")
    rows = [row for row in panel() if row["label"] in SCREEN]

    def measure(label, off, on):
        before = concept("direct-final-boss", off, off)
        after = concept("direct-final-boss", off, on)
        return {"case": label, "off": off, "on": on, "integrity": integrity(on),
                "gain": None if before is None or after is None else after - before}

    with ThreadPoolExecutor(4) as pool:
        pending = []
        for row in rows:
            label = row["label"]
            reference = read(PRIOR / "shared-off" / label / "render.json")["image"]
            target = ROOT / "cohorts" / method / "milestones" / f"RAW{step:06d}" / label
            output = render(runtime, row["prompt"], row["seed"], target, gains=strength, variation=VARIATION)
            pending.append(pool.submit(measure, label, reference, output))
        cases = [job.result() for job in pending]
    result = {"step": step, "gain": fmean(row["gain"] for row in cases)
              if all(row["gain"] is not None for row in cases) else None,
              "gains": strength, "cases": cases, "question": QUESTION}
    write(path, result)
    print(f"{method} {step*100//BUDGET}% RAW{step}: {result['gain']}; "
          f"image checks {[row['integrity']['status'] for row in cases]}", flush=True)
    return result


def train(runtime, method, attempt):
    from lumen_studio_scoped import open_training
    from lumen_studio_scoped.training import Checkpoint

    settings = options()
    cache = prepare_cache(runtime, method)
    observe = attach_judge()
    fixtures = read(PRIOR / "final-boss-outcome-jev/direct-controls-screen-first.json")["integrity"]
    if any(fixtures[name]["status"] != "fail" for name in ("broken/fixture", "split/fixture")):
        raise ValueError("The structural observer did not catch its failure fixtures")
    root = STORAGE / method / "training"
    root.mkdir(parents=True, exist_ok=True)
    pending = root / "pending-update.json"
    if pending.exists():
        raise RuntimeError("Preserve in-flight update until its exact artifact is reconciled")
    records = [read(file) for file in sorted((root / "commits").glob("*.json"))]
    if [row["step"] for row in records] != list(range(len(records))):
        raise ValueError("Committed training cursor is noncontiguous")
    for path in (ROOT / "cohorts" / method / "milestones").glob("RAW*.json"):
        if any(item["integrity"]["status"] == "fail" for item in read(path)["cases"]):
            raise RuntimeError("The cohort has a failed finished-image check")
    resume = None if not records else Checkpoint(**records[-1]["full"])
    trainer = open_training(runtime, cache, root / f"run-{attempt}", VARIATION,
        mode="fresh" if resume is None else "resume", cohort_id=f"anima-raw192-optimization-{method}-v1",
        resume=resume, seed=settings["seed"], batch_size=settings["batch_size"], total_steps=BUDGET,
        response_scale=settings["response_scale"], sigma=settings["sigma"], critic_width=settings["critic_width"],
        ema_decay=None, **settings["learning_rates"],
        native_admission=lambda request: parity(runtime, method, cache, request))
    try:
        recipe = trainer.game.recipe
        if recipe.effective_critic_formulation != "empf" or recipe.reg_every != 1:
            raise RuntimeError("Use the original stock EMPF game")
        if not records:
            entry = {"step": 0, "full": asdict(trainer.entry_FULL), "raw": asdict(trainer.entry_RAW)}
            records.append(entry)
            write(root / "commits/000000.json", entry)
        if trainer.step != records[-1]["step"]:
            raise ValueError("The trainer did not restore the exact committed cursor")
        trainer._admit_native_update()

        def milestone():
            checked = check(runtime, method, trainer.step, observe)
            if any(row["integrity"]["status"] == "fail" for row in checked["cases"]):
                raise RuntimeError("A finished-image check flagged catastrophic failure")

        if trainer.step in MILESTONES:
            milestone()
        while trainer.step < BUDGET:
            guarded()
            if shutil.disk_usage(root).free < 8 * 2**30:
                raise RuntimeError("Training volume crossed its space floor")
            step = trainer.step + 1
            write(pending, {"step": step, "base": records[-1]["full"], "attempt": attempt})
            metrics = trainer.update()
            if metrics["step"] != step or trainer.step != step:
                raise ValueError("Training clock differs from actual update count")
            entry = {"step": step, "full": asdict(trainer.save()), "metrics": metrics}
            if step in MILESTONES:
                entry["raw"] = asdict(trainer.export(trainer.directory / f"RAW{step:06d}.safetensors"))
            write(root / "commits" / f"{step:06d}.json", entry)
            records.append(entry)
            pending.unlink()
            obsolete = step - 2
            if obsolete > 0 and obsolete not in MILESTONES:
                previous = Path(records[obsolete]["full"]["path"])
                if not previous.is_relative_to(root) or previous.name != f"FULL{obsolete:06d}.pt":
                    raise ValueError("Rolling checkpoint pin points outside its owned cohort")
                previous.unlink(missing_ok=True)
            if step in MILESTONES:
                milestone()
        write(ROOT / "cohorts" / method / "result.json",
              {"step": trainer.step, "raw": records[-1]["raw"], "ema_decay": None,
               "options": settings, "cache": cache.index["fingerprint"]})
    finally:
        trainer.close()


def main():
    import torch
    from lumen_studio.backends.anima import TurboRuntime
    from lumen_studio.core_source import verify_core

    guarded()
    stage, attempt, method, *args = sys.argv[1:]
    verify_core()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(19 * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Path(read(PRIOR / "nvme-execution.json")["mirror_model"])
    runtime = TurboRuntime(model, device="cuda:0", verify=True, checkpointing=stage == "train")
    try:
        if stage == "train" and method in METHODS:
            train(runtime, method, attempt)
        elif stage == "prepare" and method in METHODS:
            prepare_cache(runtime, method)
        elif stage == "tune":
            from tune import run
            run(runtime, method)
        else:
            raise ValueError(f"Unknown {stage} or {method}")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
