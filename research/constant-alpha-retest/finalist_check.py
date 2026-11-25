"""Confirm the completed leaderboard winner and best structurally qualified recipe on the production scenes."""

from importlib.metadata import version
from pathlib import Path
import torch
from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.contracts import atomic_json, digest, file_hash
from lumen_studio.particles import ParticleAdapter
from lumen_studio.study.objectives import DISTINCT_BOSS, protocol_id
from lumen_studio.study.outputs import Outputs, read
from lumen_studio.study.reference import REFERENCES, compare_reference
from lumen_studio_scoped.training import Checkpoint
from locations import RESULTS, STUDY

policy = read(Path(__file__).with_name("confirmation-policy.json"))
assert policy["objective"] == DISTINCT_BOSS and policy["protocol"] == protocol_id()
catalog = read(RESULTS / "inventory.json")
board = read(STUDY / "sample-gallery/constant-retest/constant-retest.json")
assert (
    board["completed"] == board["total"] == 38 and board["objective"] == DISTINCT_BOSS
)
eligible = [row for row in board["recipes"] if row.get("rank") is not None]
eligible.sort(key=lambda r: (-r["score"], r["method"]))
qualified = [row for row in eligible if row["integrity"] == "pass"]
roles = {
    "leader": eligible[0]["method"],
    "qualified": qualified[0]["method"] if qualified else None,
}
pinned = read(REFERENCES["final boss"])
assert (
    pinned["raw_sha256"] == policy["reference_raw"]
    and pinned["distinct_boss"]["protocol"] == protocol_id()
)
cases = read(STUDY / "native-prod/boss-short/validation.json")["confirm"]
assert [digest([c["prompt"], c["seed"]]) for c in cases] == policy["cases"]
assert not {digest([c["prompt"], c["seed"]]) for c in board["requests"].values()} & set(
    policy["cases"]
)
rubric = read(STUDY / "product/configs/anima/adult-study.json")["concepts"][
    "Final Boss"
]["effect"]
dest = RESULTS / "finalist-check"
entries = []
for name in dict.fromkeys(n for n in roles.values() if n is not None):
    row = next(r for r in eligible if r["method"] == name)
    entries.append(
        {
            field: row[field]
            for field in ("method", "raw_sha256", "alpha", "score", "rank", "integrity")
        }
    )
reservation = dict(
    protocol=policy, roles=roles, entries=entries, reference=pinned["raw_sha256"]
)
pin = dest / "selection.json"
if pin.exists():
    assert read(pin) == reservation
else:
    atomic_json(pin, reservation)
assert version("particlegan") == "0.12.0"
torch.backends.cuda.matmul.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(
    min(1.0, 19 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
)
runtime = TurboRuntime(
    "/mnt/nvme/sliders/qualification/models/anima-turbo-v11",
    device="cuda:0",
    verify=True,
)
try:
    for entry in entries:
        name = entry["method"]
        path = dest / name / "result.json"
        if path.exists():
            assert read(path)["reservation"] == digest(reservation)
            continue
        r = catalog[name]
        raw = Checkpoint(**r["raw"])
        assert file_hash(raw.path) == raw.sha256 == entry["raw_sha256"]
        parity = read(RESULTS / "methods" / name / "pg012-parity.json")
        assert runtime.identity == parity["replay_model"]
        adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
        m = adapter.load_export(raw.path, model_identity=parity["training_model"])
        assert m["weights"] == "live" and m["step"] == r["step"]
        runtime.mixer.add(name, adapter.requires_grad_(False).eval())
        outputs = Outputs(path.parent, runtime, "Final Boss", rubric)
        assert (
            outputs.integrity.calibrate()["passed"]
            and outputs.objective == DISTINCT_BOSS
        )
        comparison = outputs.compare(cases, name, raw, entry["alpha"], "constant")
        assert comparison["gains"] == [entry["alpha"]] * 10
        reference = compare_reference("Final Boss", comparison, rubric)
        if (
            reference["status"] != "matched"
            and comparison["objective_score"] is not None
        ):
            raise ValueError(
                "The new design score could not be compared on the frozen reference panel"
            )
        atomic_json(
            path,
            dict(
                method=name,
                reservation=digest(reservation),
                reference=reference,
                measurement=comparison,
            ),
        )
        print(
            name,
            comparison["objective_score"],
            comparison["status"],
            reference["decision"],
            flush=True,
        )
        runtime.mixer.close()
        del adapter
finally:
    runtime.close()
