"""Separate the committed scripts, this campaign's outputs and the equipped parent study."""

import json
import os
from pathlib import Path

CODE = Path(__file__).resolve().parent
POLICY = json.loads((CODE / "policy.json").read_bytes())
# The campaign directory holds this checkout as ``product`` next to its outputs.
ROOT = Path(os.environ.get("ANIMA_POST_TRAINING_ROOT", CODE.parents[2])).resolve()
PARENT_STUDY = Path(os.environ.get("ANIMA_PARENT_STUDY", POLICY["parent"]["study"]))
MODEL = "/mnt/nvme/sliders/qualification/models/anima-turbo-v11"


def parent_arm():
    return PARENT_STUDY / POLICY["parent"]["arm"]


def open_runtime(**options):
    from importlib.metadata import version

    import torch
    from lumen_studio.backends.anima import TurboRuntime

    assert version("particlegan") == POLICY["pg"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, 19 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    )
    return TurboRuntime(MODEL, device="cuda:0", verify=True, **options)
