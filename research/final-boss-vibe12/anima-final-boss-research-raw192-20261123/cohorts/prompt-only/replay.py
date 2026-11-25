"""Qualify the native profile and verify the packaged RAW from fresh inference."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
import sys

from design import ROOT, save
from judge import compare
from native import MODEL_DIR, guarded, pin, read, render
from train import BUDGET, MILESTONES, holdouts, schedules


def loaded(runtime, asset, step):
    from lumen_studio.particles import ParticleAdapter

    path = Path(asset["path"])
    if pin(path)["sha256"] != asset["sha256"] or path.stat().st_size != asset["bytes"]:
        raise ValueError("Saved RAW changed")
    sidecar = read(path.with_suffix(".scoped.json"))
    if (sidecar["native_RAW_sha256"] != asset["sha256"] or sidecar["ema_decay"] is not None
            or sidecar["completed_steps"] != step):
        raise ValueError("RAW provenance or update count mismatch")
    adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
    metadata = adapter.load_export(path, model_identity=runtime.identity)
    if metadata["weights"] != "live" or metadata["step"] != step:
        raise ValueError("Inference requires the matching native RAW adapter")
    name = f"character-final-boss-{step}"
    runtime.mixer.add(name, adapter)
    return name


def new_profile(runtime, step, name, profile, gains, *, fresh=False):
    path = ROOT / "replay-results" / f"RAW{step:06d}-{profile}.json"
    if path.exists():
        return read(path)
    tasks = []
    for row in holdouts():
        folder = ROOT / "heldout" / row["label"]
        off = read(folder / "off" / "render.json")["image"]
        output = f"{'fresh-' if fresh else ''}raw{step:06d}-{profile}"
        on = render(runtime, row["prompt"], row["seed"], folder / output,
                    gains=gains, variation=name)
        tasks.append((row, off, on))

    def score(task):
        row, off, on = task
        observed = compare(f"qualification-raw{step:06d}-{profile}-{row['label']}-v1",
                           [Path(off["path"]), Path(on["path"])], row["background"], row["lighting"],
                           character=row["person"].removeprefix("adult "))
        return dict(row=row, result=observed, off=off, on=on)

    with ThreadPoolExecutor(2) as pool:
        cases = list(pool.map(score, tasks))
    peaks = [probability for row in cases for probability in row["result"]["catastrophic_probability"]]
    status = "fail" if max(peaks) >= .8 else "unmeasured" if max(peaks) > .2 else "pass"
    result = dict(cases=cases, status=status, mean_gain=fmean(row["result"]["gain"] for row in cases),
        mean_background=fmean(row["result"]["background_same"] for row in cases),
        mean_lighting=fmean(row["result"]["lighting_same"] for row in cases))
    save(path, result)
    return result


def main():
    import torch
    from lumen_studio.backends.anima import TurboRuntime
    from lumen_studio.core_source import verify_core

    guarded()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(19 * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    verify_core()
    if sys.argv[1] == "probe":
        step = 64
        raw = read(ROOT / "training/commits" / f"{step:06d}.json")["raw"]
        runtime = TurboRuntime(MODEL_DIR, device="cuda:0", verify=True)
        try:
            name = loaded(runtime, raw, step)
            print(new_profile(runtime, step, name, "constant1", [1.] * 10)["mean_gain"], flush=True)
        finally:
            runtime.close()
        return
    result = read(ROOT / "result.json")
    shortlist = sorted({item["step"] for item in result["equal_best"]} | {BUDGET})
    tested = {step: read(ROOT / "milestones" / f"RAW{step:06d}.json")["arms"] for step in MILESTONES}
    force_fresh = sys.argv[1] == "replay-fresh"
    runtime = TurboRuntime(MODEL_DIR, device="cuda:0", verify=True)
    try:
        profiles = schedules(runtime) | {"constant1": [1.] * 10}
        raw = {step: read(ROOT / "training/commits" / f"{step:06d}.json")["raw"] for step in MILESTONES}
        attached = {}
        for step in shortlist:
            attached[step] = name = loaded(runtime, raw[step], step)
            for profile, gains in profiles.items():
                if force_fresh or profile not in tested[step]:
                    tested[step][profile] = new_profile(runtime, step, name, profile, gains,
                                                        fresh=force_fresh)
                    print(f"RAW{step} {profile} intended effect {tested[step][profile]['mean_gain']:+.5f}", flush=True)
        # Earlier checkpoints that only received an early preview remain
        # evidence. The inference deliverable uses a fully compared checkpoint.
        ranked = [(result["mean_gain"], step, profile, result)
                  for step in shortlist for profile, result in tested[step].items() if result["status"] == "pass"]
        maximum = max(item[0] for item in ranked)
        winners = []
        for value, step, profile, report in ranked:
            if value != maximum:
                continue
            row = report["cases"][0]["row"]
            observed = render(runtime, row["prompt"], row["seed"],
                ROOT / "independent-replay" / f"{'fresh-' if force_fresh else ''}raw{step:06d}-{profile}",
                gains=profiles[profile], variation=attached[step])
            expected = report["cases"][0]["on"]
            if observed["sha256"] != expected["sha256"]:
                raise RuntimeError("Fresh RAW replay differs from its original PNG bytes")
            winners.append(dict(step=step, profile=profile, result=report, raw=raw[step],
                                independent_replay=observed, gains=profiles[profile], byte_exact=True))
        save(ROOT / "qualified-result.json", result | {"selection_scope": "fully_compared_checkpoint_profiles",
            "best_gain": maximum, "equal_best": winners, "tested_profiles": {
                str(step): {name: {"mean_gain": arm["mean_gain"], "status": arm["status"]}
                            for name, arm in arms.items()} for step, arms in tested.items()}})
        print(f"Fresh independent RAW replay verified. Best intended effect {maximum:+.5f}", flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
