"""Give one arm its additional stock updates from the pinned parent checkpoint."""

from dataclasses import asdict
import shutil
import sys

from lumen_studio.cache import TargetCache
from lumen_studio.contracts import atomic_json, digest, file_hash
from lumen_studio.study.objectives import DISTINCT_BOSS
from lumen_studio.study.outputs import Outputs, read
from lumen_studio.study.training import Catastrophe, arm_path, train
from lumen_studio_scoped.training import Checkpoint, file_pin

from campaign import PARENT_STUDY, POLICY, ROOT, open_runtime, parent_arm

name = sys.argv[1]
budget = int(sys.argv[2]) if len(sys.argv) > 2 else POLICY["additional_updates"]
arm, parent = POLICY["arms"][name], POLICY["parent"]
rates = arm.get("rates", POLICY["rates"])
seed = arm.get("seed", POLICY["seed"])
root = ROOT / "arms" / name
root.mkdir(parents=True, exist_ok=True)
cursor = read(parent_arm() / "cursor.json")
raw, full = Checkpoint(**cursor["raw"]), Checkpoint(**cursor["full"])
assert cursor["step"] == parent["step"] and cursor["rates"] == POLICY["rates"]
assert arm["entry"] == "fork" or (rates, seed) == (POLICY["rates"], POLICY["seed"])
assert file_pin(raw.path) == raw and raw.sha256 == parent["raw_sha256"]
assert file_pin(full.path) == full and full.sha256 == parent["full_sha256"]
cache = TargetCache(
    PARENT_STUDY / parent["cache"] if arm["targets"] == "parent" else ROOT / "targets" / arm["targets"],
    verify=True,
)
if arm["targets"] == "parent":
    assert cache.index["fingerprint"] == parent["cache_fingerprint"]
else:
    assert cache.index["identity"]["teacher_replacement"]["parent_cache"] == parent["cache_fingerprint"]
panels = read(PARENT_STUDY / parent["validation"])
pin = dict(arm=name, parent=asdict(raw), cache=cache.index["fingerprint"],
           validation=file_hash(PARENT_STUDY / parent["validation"]), rates=rates, preset="empf",
           seed=seed, batch_size=POLICY["batch_size"], ema_decay=None)
if (root / "plan.json").exists():
    # Adding an arm to the policy leaves every earlier arm's own plan unchanged.
    assert {k: v for k, v in read(root / "plan.json").items() if k != "policy"} == pin
else:
    atomic_json(root / "plan.json", pin)
# Earlier answers are content-addressed. Copies keep the parent study read-only.
(root / "judgments").mkdir(exist_ok=True)
for receipt in (PARENT_STUDY / "duration-study/published/judgments").glob("*.json"):
    if len(receipt.stem) == 64 and not (root / "judgments" / receipt.name).exists():
        shutil.copyfile(receipt, root / "judgments" / receipt.name)
rt = open_runtime(checkpointing=True)
try:
    outputs = Outputs(root, rt, "Final Boss",
                      read(PARENT_STUDY / parent["spec"])["concepts"]["Final Boss"]["effect"])
    assert outputs.objective == DISTINCT_BOSS and rt.identity == cache.index["identity"]["model"]
    check = outputs.integrity.calibrate()
    atomic_json(root / "detector.json", check)
    assert check["passed"]
    quarters = [budget * k // 4 for k in (1, 2, 3)]
    options = dict(schedule_horizon=POLICY["schedule_horizon"], always_check_rates=True)
    try:
        if arm["entry"] == "resume":
            # The exact continuation keeps the parent's cohort, critic, optimizers and streams.
            dest = arm_path(root, rates)
            if not (dest / "cursor.json").exists():
                dest.mkdir(parents=True, exist_ok=True)
                atomic_json(dest / "cursor.json", cursor)
            base = parent["step"]
            train(outputs, cache, panels, rates, base + budget, "fit",
                  observation_steps=[base + n for n in quarters], **options)
        else:
            warm = dict(parent_full=full) if arm.get("critic") == "parent" else {}
            train(outputs, cache, panels, rates, budget, "fit", parent=raw, seed=seed, **warm,
                  observation_steps=[n for n in quarters if n % (budget // 4)], **options)
        atomic_json(root / f"terminal-{budget}.json", dict(status="completed", additional=budget))
    except Catastrophe:
        failure = read(arm_path(root, rates) / "failure.json")
        atomic_json(root / f"terminal-{budget}.json",
                    dict(status="stopped", last=failure["step"]))
finally:
    rt.close()
