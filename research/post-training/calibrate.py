"""Search one constant alpha for an arm's checkpoint, then rank it on the held-out characters."""

from dataclasses import asdict
from pathlib import Path
import shutil
import sys

from campaign import CODE, PARENT_STUDY, POLICY, ROOT, open_runtime

# The measurement is the constant-alpha retest: four search and eight ranking characters.
panel = ROOT / "cohort-retest"
panel.mkdir(exist_ok=True)
if not (panel / "panels.json").exists():
    shutil.copyfile(PARENT_STUDY / "cohort-retest/panels.json", panel / "panels.json")
sys.path.insert(0, str(CODE.parent / "constant-alpha-retest"))

from evaluate import D, cases, cfg, retest  # noqa: E402

from lumen_studio.contracts import atomic_json  # noqa: E402
from lumen_studio.particles import ParticleAdapter  # noqa: E402
from lumen_studio.study.outputs import Outputs, read  # noqa: E402
from lumen_studio.study.scalar_search import verify_search  # noqa: E402
from lumen_studio.study.training import arm_path  # noqa: E402
from lumen_studio_scoped.training import Checkpoint, file_pin  # noqa: E402

name = sys.argv[1]
additional = int(sys.argv[2]) if len(sys.argv) > 2 else POLICY["additional_updates"]
arm, parent = POLICY["arms"][name], POLICY["parent"]
fork = arm["entry"] == "fork"
step = additional if fork else parent["step"] + additional
trained = arm_path(ROOT / "arms" / name, arm.get("rates", POLICY["rates"]))
base = ROOT / "arms" / name / "calibration" / str(additional)
base.mkdir(parents=True, exist_ok=True)
reference = PARENT_STUDY / parent["calibration"] / D


def attach(rt, key, raw, expected_step):
    adapter = ParticleAdapter(rt.transformer).to(rt.device)
    meta = adapter.load_export(raw.path, model_identity=rt.identity)
    assert meta["step"] == expected_step and meta["weights"] == "live"
    adapter.requires_grad_(False).eval()
    rt.mixer.add(key, adapter)


rt = open_runtime()
try:
    out = Outputs(base, rt, "Final Boss", {})
    assert out.integrity.calibrate()["passed"]
    if fork and not (ROOT / "arms" / name / "fork-parity.json").exists():
        # The untouched child must draw the parent's images, with the slider on and off.
        entry = Checkpoint(**read(trained / "updates" / "000001.json")["full"])
        raw0 = file_pin(Path(entry.path).with_name("RAW000000.safetensors"))
        attach(rt, "entry", raw0, 0)
        alpha = parent["alpha"]
        archived = read(reference / "ranking" / (alpha.hex() + ".json"))["rows"]
        rows = {}
        for case in cfg["panels"]["ranking"]:
            on = out.image(**cases[case]["request"], raw=raw0.sha256, variation="entry",
                           gains=[alpha] * 10)
            off = out.image(**cases[case]["request"])
            rows[case] = dict(on=on.sha256 == archived[case]["on"]["sha256"],
                              off=off.sha256 == cases[case]["off"]["sha256"])
        rt.mixer.close()
        assert all(v["on"] and v["off"] for v in rows.values()), rows
        atomic_json(ROOT / "arms" / name / "fork-parity.json",
                    dict(child_RAW0=asdict(raw0), parent_RAW=parent["raw_sha256"], alpha=alpha,
                         byte_identical=rows))
    fit = read(trained / "fit" / f"{step}.json")
    raw = Checkpoint(**fit["raw"])
    assert file_pin(raw.path) == raw
    key = f"post-training-{name}-{additional}"
    attach(rt, key, raw, step)
    retest(key, out, asdict(raw), D, archive=False)
    result = read(base / D / "result.json")
    verify_search(result["trials"], result.get("alpha"))
finally:
    rt.close()
print("Calibrated", name, additional, result.get("alpha"), result.get("score"), flush=True)
