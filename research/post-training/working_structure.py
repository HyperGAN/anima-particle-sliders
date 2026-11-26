"""Run the structural check on each RAW's held-out images at its re-searched alpha."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

from particle_sliders.candidate_failure import FileRecord

from lumen_studio.contracts import atomic_json
from lumen_studio.vibe_integrity import FrameIntegrity

from campaign import ROOT

home = ROOT / "working-search"
integrity = FrameIntegrity(ROOT / "working-metric" / "judgments")
assert integrity.calibrate()["passed"]


def read(path):
    return json.loads(Path(path).read_bytes())


with ThreadPoolExecutor(8) as pool:
    for directory in sorted(p for p in home.iterdir() if (p / "result.json").exists()):
        result = read(directory / "result.json")
        target = directory / "structure.json"
        if result.get("alpha") is None or (target.exists() and read(target)["alpha"] == result["alpha"]):
            continue
        rows = read(directory / "ranking" / (result["alpha"].hex() + ".json"))["rows"]
        jobs = {c: pool.submit(integrity.check, FileRecord.from_path(Path(r["on"]["path"]))) for c, r in rows.items()}
        checks = {c: j.result() for c, j in jobs.items()}
        states = [v["status"] for v in checks.values()]
        atomic_json(target, dict(alpha=result["alpha"], checks=checks,
                                 passed=states.count("pass"), failed=states.count("fail"),
                                 unresolved=states.count("unmeasured")))
        print(directory.name, states.count("pass"), "pass", states.count("fail"), "fail",
              states.count("unmeasured"), "unresolved", flush=True)
