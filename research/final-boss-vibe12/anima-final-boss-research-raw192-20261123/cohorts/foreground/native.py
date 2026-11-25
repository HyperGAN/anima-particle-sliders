"""Native Anima stage runner for the character-only corpus."""

import hashlib
import json
import os
from pathlib import Path
import sys

from design import ROOT, save

MODEL_DIR = Path("/ml2/ntc-image-studio/artifacts/anima/model")


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(path.read_text())


def guarded():
    if os.environ.get("ANIMA_WORKER_GUARD") != str(os.getppid()):
        raise RuntimeError("Run through the owned Anima GPU guard")


def render(runtime, prompt, seed, folder, *, gains=None, variation=None):
    guarded()
    evidence = folder / "render.json"
    png = folder / "image.png"
    args = dict(prompt=prompt, seed=seed, model=runtime.identity, gains=gains, variation=variation)
    if evidence.exists():
        receipt = read(evidence)
        if receipt != dict(args, image=pin(png)):
            raise RuntimeError(f"Render identity changed: {folder}")
        return receipt["image"]
    folder.mkdir(parents=True, exist_ok=True)
    if png.exists():
        raise RuntimeError(f"Unreconciled render without its receipt: {png}")
    if gains is None:
        image = runtime.render(prompt, seed, width=512, height=512, steps=10)
    else:
        import torch

        with torch.no_grad():
            embedding = runtime.encode(prompt)
            latent = runtime.noise(seed, 512, 512)
            scheduler = runtime.scheduler(10)
            for timestep, gain in zip(scheduler.timesteps, gains, strict=True):
                guarded()
                with runtime.mixer.scales({variation: float(gain)}):
                    velocity = runtime.predict(latent, timestep, embedding)
                latent = runtime.step(scheduler, velocity, timestep, latent)
            image = runtime.decode(latent)
    temp = folder / "image.png.partial"
    image.convert("RGB").save(temp, format="PNG")
    temp.replace(png)
    result = pin(png)
    save(evidence, dict(args, image=result))
    return result


def collect(runtime):
    plan = read(ROOT / "design.json")
    for row in plan["rows"]:
        for side in ("neutral", "positive"):
            render(runtime, row[side], row["seed"], ROOT / "images" / row["id"] / side)
        print(f"Rendered matched pair {row['id']}", flush=True)


def main():
    guarded()
    import torch
    from lumen_studio.backends.anima import TurboRuntime
    from lumen_studio.core_source import verify_core

    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(19 * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    verify_core()
    stage, attempt = sys.argv[1:3]
    runtime = TurboRuntime(MODEL_DIR, device="cuda:0", verify=True, checkpointing=stage == "train")
    try:
        if stage == "collect":
            collect(runtime)
        else:
            from fit import fit
            fit(runtime, stage, attempt)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
