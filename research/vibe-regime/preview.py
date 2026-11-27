"""Render a saved checkpoint of a regime run on the four search characters, to look at.

    preview.py <update> <alpha> [<alpha> ...]

This chooses nothing. The run's own search, ranking and confirmation are unaffected.
"""

from dataclasses import asdict
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(CODE), str(CODE.parent / "post-training")]

from lumen_studio.contracts import atomic_json  # noqa: E402
from lumen_studio.particles import ParticleAdapter  # noqa: E402
from lumen_studio.study.outputs import Outputs, read  # noqa: E402
from lumen_studio.study.training import arm_path  # noqa: E402

from campaign import PARENT_STUDY, ROOT, open_runtime  # noqa: E402

step, alphas = int(sys.argv[1]), [float(a) for a in sys.argv[2:]]
raw = read(arm_path(ROOT, dict(g=0.00075, d=0.009, particle=0.03)) / "fit" / f"{step}.json")["raw"]
board = read(PARENT_STUDY / "cohort-retest/panels.json")
cases = {c["id"]: c for c in board["cases"] if c["id"] in board["panels"]["search"]}
rt = open_runtime()
try:
    out = Outputs(ROOT / "preview", rt, "preview", {})
    adapter = ParticleAdapter(rt.transformer).to(rt.device)
    assert adapter.load_export(raw["path"], model_identity=rt.identity)["step"] == step
    adapter.requires_grad_(False).eval()
    rt.mixer.add("preview", adapter)
    try:
        rows = {str(alpha): {c: asdict(out.image(**case["request"], raw=raw["sha256"], variation="preview",
                                                 gains=[alpha] * 10)) for c, case in cases.items()} for alpha in alphas}
    finally:
        rt.mixer.close()
    atomic_json(ROOT / "preview" / f"{step}.json", dict(step=step, off={c: v["off"] for c, v in cases.items()}, rows=rows))
finally:
    rt.close()
