"""Render a matched case with the published Anima Final Boss research default."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import torch

from lumen_studio.contracts import atomic_json, file_hash
from lumen_studio.provenance import same_portable_model
from release_tools.scheduled_render import render_scheduled
from release_tools.final_boss_targets import STUDY, VARIATION


@torch.no_grad()
def replay(runtime, checkpoint, case, config):
    from lumen_studio.particles import ParticleAdapter

    expected = config["checkpoints"][config["recipe"]]
    if checkpoint.stat().st_size != expected["bytes"] or file_hash(checkpoint) != expected["sha256"]:
        raise ValueError("The RAW checkpoint differs from the published research default")
    recorded = config["training_model_identity"]
    # An exact, output-qualified runtime may use a different package environment.
    # It must still reproduce the archived reference before rendering a new case.
    certified = runtime.identity in config.get("qualified_replay_models", ())
    if not certified and not same_portable_model(recorded, runtime.identity):
        raise ValueError("Use a native runtime qualified for this Final Boss checkpoint")
    adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
    # This manifest pins both the checkpoint's original identity and the
    # current runtime whose replay was checked against the original renders.
    metadata = adapter.load_export(checkpoint, model_identity=recorded)
    if metadata["weights"] != "live" or metadata["step"] != config["step"]:
        raise ValueError("Replay requires the selected RAW checkpoint")
    adapter.requires_grad_(False).eval()
    runtime.mixer.add(VARIATION, adapter)
    runtime.mixer.strengths[VARIATION] = 1.0  # The manifest already records effective gains.
    size = config["resolution"]

    def render(request):
        return render_scheduled(runtime, request["prompt"], request["seed"], size, size,
                                config["steps"], config["gains"])

    reference = config["reference_case"]
    if (case == reference or certified
            or runtime.identity["runtime_sha256"] not in config["qualified_replay_runtime_sha256"]):
        image, info = render(reference)
        with BytesIO() as buffer:
            image.convert("RGB").save(buffer, format="PNG")
            if hashlib.sha256(buffer.getvalue()).hexdigest() != config["reference_png_sha256"]:
                raise ValueError("Native replay differs from the winner’s measured reference image")
        if case == reference:
            return image, info
    return render(case)


def main():
    config = json.loads((STUDY / "default.json").read_text())
    results = json.loads((STUDY / "results.json").read_text())
    cases = {case["label"]: case for case in results["cases"] + results["confirmation"]["cases"]}
    reference = config["reference_case"]
    cases[reference["label"]] = reference
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=cases, default=reference["label"])
    parser.add_argument("--model", type=Path, required=True, help="Converted Anima Turbo v1.1 directory")
    parser.add_argument("--checkpoint", type=Path, help="Use a local copy of the published RAW checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="New output PNG")
    args = parser.parse_args()
    receipt = args.out.with_suffix(".json")
    if args.out.suffix.lower() != ".png" or args.out.exists() or receipt.exists():
        parser.error("Use a new .png output path")
    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        expected = config["checkpoints"][config["recipe"]]
        checkpoint = Path(hf_hub_download(config["checkpoint_repository"], expected["file"],
                                          revision=config["checkpoint_revision"]))

    from lumen_studio.backends.anima import TurboRuntime

    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = TurboRuntime(args.model, device="cuda:0", verify=True)
    try:
        case = cases[args.case]
        image, replay_info = replay(runtime, checkpoint, case, config)
        expected_grid = config["sigmas"]
        if replay_info["sigmas"] != expected_grid or replay_info["effective_strengths"] != config["gains"]:
            raise ValueError("Native grid or effective strength differs from the measured replay")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("xb") as stream:
            image.convert("RGB").save(stream, format="PNG")
        atomic_json(receipt, dict(case=case, recipe=config["recipe"], model=runtime.identity,
                    profile=config["profile"], alpha=config["alpha"], image_sha256=file_hash(args.out),
                    checkpoint_sha256=file_hash(checkpoint), **replay_info))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
