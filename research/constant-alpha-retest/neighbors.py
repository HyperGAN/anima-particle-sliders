"""Use the identical scalar search on each native non-Boss concept with its own pinned rubric."""

import sys
from importlib.metadata import version
from dataclasses import asdict
from statistics import fmean
import torch
from lumen_studio.contracts import atomic_json, digest
from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.study.outputs import Outputs, read
from lumen_studio.study.reference import compare_reference
from lumen_studio_scoped import open_inference
from lumen_studio_scoped.training import Checkpoint
from particle_sliders.candidate_failure import FileRecord
from release_tools.constant_alpha_retest import search_constant, verify_search
from locations import RESULTS as here

key = sys.argv[1]
source = here.parent / "native-prod" / key
old = here.parent / "constant-alpha" / key
root = here / "neighbor-effect-results" / key
raw = Checkpoint(**read(old / "training-selection.json")["raw"])
FileRecord(**asdict(raw)).verify()
cfg = read(source / "manifest.json")
rubric = cfg["collection_identity"]["effect"]
panels = read(source / "validation.json")
previous = read(source / "outcome.json")
index = read(source / "frozen/targets/index.json")
variation = index["identity"]["variation"]
cache = {}
archived = here / "neighbor-results" / key / "search"
records = [
    *read(source / "strengths.json"),
    *read(old / "strengths.json"),
    *(read(p) for p in sorted(archived.glob("*.json"))),
]
for row in records:
    if (
        row["raw"] != asdict(raw)
        or row["profile"] != "constant"
        or [p["case"] for p in row["pairs"]] != panels["steps"]
    ):
        continue
    assert row["gains"] == [row["alpha"]] * 10
    for p in row["pairs"]:
        FileRecord(**p["on"]["output"]).verify()
    cache[row["alpha"]] = row
assert not {c["seed"] for c in panels["steps"]} & {c["seed"] for c in panels["confirm"]}
assert version("particlegan") == "0.12.0"
torch.backends.cuda.matmul.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(
    min(1.0, 19 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
)
rt = TurboRuntime(
    "/mnt/nvme/sliders/qualification/models/anima-turbo-v11",
    device="cuda:0",
    verify=True,
)
try:
    assert rt.identity == index["identity"]["model"]
    outputs = Outputs(root, rt, previous["vibe"], rubric)
    assert outputs.integrity.calibrate()["passed"]
    owner = open_inference(
        rt,
        variation,
        raw=raw,
        owner_admission=lambda request: (
            request["model_identity"] == rt.identity and request["RAW"] == asdict(raw)
        ),
    )
    try:

        def trial(a):
            path = root / "search" / (a.hex() + ".json")
            if not path.is_file():
                row = (
                    cache[a]
                    if a in cache
                    else outputs.compare(
                        panels["steps"], variation, raw, a, "constant", owner=owner
                    )
                )
                atomic_json(path, row)
            row = read(path)
            assert row["raw"] == asdict(raw) and row["gains"] == [a] * 10
            return row["mean_gain"]

        strengths = search_constant(trial)
        chosen = max(
            (a for a, v in strengths.items() if v is not None),
            key=lambda a: (strengths[a], -a),
        )
        verify_search(list(strengths.items()), chosen)
        selection = dict(
            objective="matched_intended_effect",
            raw=asdict(raw),
            pg=version("particlegan"),
            alpha=chosen,
            gains=[chosen] * 10,
            trials=list(map(list, strengths.items())),
            rubric_sha256=digest(rubric),
        )
        pin = root / "selected.json"
        if pin.is_file():
            assert read(pin) == selection
        else:
            atomic_json(pin, selection)
        confirmed = outputs.compare(
            panels["confirm"], variation, raw, chosen, "constant", owner=owner
        )
        assert confirmed["raw"] == asdict(raw) and confirmed["gains"] == [chosen] * 10
        assert confirmed["mean_gain"] is None or confirmed["mean_gain"] == fmean(
            p["gain"] for p in confirmed["pairs"]
        )
        atomic_json(root / "ranking.json", confirmed)
        reference = compare_reference(previous["vibe"], confirmed, rubric)
        atomic_json(
            root / "result.json",
            dict(
                selection,
                slider=previous["vibe"],
                step=previous["selected_step"],
                score=confirmed["mean_gain"],
                status=confirmed["status"],
                reference=reference,
            ),
        )
        print(key, chosen, confirmed["mean_gain"], confirmed["status"], flush=True)
    finally:
        owner.close()
finally:
    rt.close()
