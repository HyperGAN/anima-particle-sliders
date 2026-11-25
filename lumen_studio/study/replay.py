"""Render a confirmed RAW adapter using its recorded native strength schedule."""

import argparse
from dataclasses import asdict
from pathlib import Path
import os
import tempfile

import torch
from particle_sliders.strength_schedule import interval_strengths
from lumen_studio_scoped import open_inference
from lumen_studio_scoped.training import Checkpoint
from ..backends.anima import TurboRuntime
from .outputs import read, sample_adapter


def qualified_strength(outcome):
    if outcome["selection"] != "raw" or outcome["ema_decay"] is not None:
        raise ValueError("Expected a RAW slider")
    setting = outcome["strength"]
    final = outcome["confirmation"]
    if (
        final["status"] != "pass"
        or not final["mean_gain"] > 0
        or any(
            final[key] != setting[key]
            for key in ("raw", "alpha", "profile", "gains", "sigmas")
        )
    ):
        raise ValueError(
            "Replay requires the setting that passed held-out confirmation"
        )
    return setting


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("choose a new output PNG")
    setting = qualified_strength(read(args.outcome))
    raw = Checkpoint(**setting["raw"])
    meta = read(Path(raw.path).with_suffix(".scoped.json"))
    variation = meta["options"]["variation"]
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("The native replay requires a CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, 19 * 2**30 / torch.cuda.get_device_properties(device).total_memory),
        device,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    rt = TurboRuntime(args.model_dir, device=args.device, verify=True)
    try:
        sigmas = rt.scheduler(10).sigmas.tolist()
        gains = list(
            interval_strengths(
                setting["alpha"],
                [b - a for a, b in zip(sigmas, sigmas[1:])],
                profile=setting["profile"],
            )
        )
        if gains != setting["gains"] or sigmas != setting["sigmas"]:
            raise ValueError("Qualified sampler intervals or gains differ")
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
            with owner.scheduled(gains):
                image = sample_adapter(
                    rt, variation, gains, args.prompt, args.seed
                ).convert("RGB")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                suffix=".png", dir=args.out.parent
            ) as temp:
                image.save(temp, format="PNG")
                temp.flush()
                os.fsync(temp.fileno())
                os.link(temp.name, args.out)
        finally:
            owner.close()
    finally:
        rt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
