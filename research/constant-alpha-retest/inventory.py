"""Account for each recipe and learning-rate arm that entered the referenced studies."""

from dataclasses import asdict
from lumen_studio.contracts import atomic_json
from lumen_studio.study.outputs import read
from particle_sliders.candidate_failure import FileRecord
from locations import RESULTS as here

base = here.parent
camp = base.parent / "anima-final-boss-vibe12-20261124"
board = read(base / "product/docs/final-boss-raw192/vibe12/standings.json")
entries = {}
for method, state in read(camp / "campaign-state.json")["recipes"].items():
    d = camp / "cohorts" / method
    if not d.is_dir():
        continue
    if method in board["methods"]:
        result = read(d / "result.json")
        step = result["step"]
        raw = result["raw"]
        status = "completed"
        proof = None
    else:
        proof = max((d / "milestones").glob("*.json"), key=lambda p: p.stem)
        step = read(proof)["step"]
        matched = list(
            (camp / "storage" / method / "training").glob(
                f"*/RAW{step:06d}.safetensors"
            )
        )
        if len(matched) != 1:
            raise ValueError((method, matched))
        raw = asdict(FileRecord.from_path(matched[0].resolve()))
        proof = str(proof)
        status = "training stopped"
    entries[method] = dict(
        label=board["methods"].get(method, {}).get("label", method),
        run="Vibe12",
        pg="0.11.1",
        step=step,
        raw=raw,
        training=status,
        parity=proof,
    )
assert len(entries) == 26
for p in sorted((base / "native-prod").glob("*/training-config.json")):
    run = p.parent
    config = read(p)
    for arm in sorted((run / "rates").iterdir()):
        if not (arm / "cursor.json").is_file():
            continue
        cursor = read(arm / "cursor.json")
        fit = sorted(arm.glob("fit/[0-9]*.json"))
        stop = (arm / "failure.json").exists()
        chosen = None
        if not stop and fit:
            eligible = []
            for f in fit:
                m = read(f)
                screen = arm / "fit-screen" / f.name
                if m["status"] == "pass" and (
                    not screen.exists() or read(screen)["status"] == "pass"
                ):
                    eligible.append((f, m))
            if eligible:
                chosen = max(eligible, key=lambda pair: pair[1]["mean_gain"])
        if chosen:
            f, m = chosen
            step = int(f.stem)
            raw = m["raw"]
            category = "completed"
            proof = f
        else:
            step = cursor["step"]
            raw = cursor["raw"]
            category = (
                "training stopped"
                if stop
                else "rate screen only"
                if not fit
                else "training unresolved"
            )
            proof = next(
                (
                    d / f"{step}.json"
                    for d in (arm / "fit", arm / "screen")
                    if (d / f"{step}.json").is_file()
                )
            )
        name = run.name + "-" + arm.name[:8]
        if name in entries:
            raise ValueError(name)
        FileRecord(**raw).verify()
        entries[name] = dict(
            label=run.name,
            run=run.name,
            pg="0.12.0",
            step=step,
            raw=raw,
            training=category,
            rates=cursor["rates"],
            parity=str(proof),
            arm=arm.name,
            training_budget=config.get("max_updates", config["horizon"]),
        )
probe = base / "final-boss-comparison/version-probe"
check = read(probe / "comparison.json")
(proof,) = list(probe.glob("rates/*/fit/48.json"))
row = read(proof)
assert check["new_sha"] == row["raw"]["sha256"]
entries["pg012-late-cues-control"] = dict(
    label="Late-cues version control",
    run="version-probe",
    pg="0.12.0",
    step=48,
    raw=row["raw"],
    training="version control",
    rates=check["same_rates"],
    parity=str(proof),
    training_budget=48,
)
atomic_json(here / "inventory.json", entries)
print(
    len(entries),
    "trained arms",
    {pg: sum(r["pg"] == pg for r in entries.values()) for pg in ("0.11.1", "0.12.0")},
)
