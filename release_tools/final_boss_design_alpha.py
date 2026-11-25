"""Export or replay a distinct-design alpha search without changing the trained adapter."""

import argparse
import gzip
import json
from math import isclose
from pathlib import Path

from .final_boss_damage_search import FIRST_PROTOCOL_CEILING, LOWER, search, untried_side
from .final_boss_default import scheduled_gains, strongest_alpha
from .final_boss_design_scores import summarize
from .final_boss_visible_damage import score_case
from .publish_final_boss_vibe12 import read
from .verify_final_boss_designs import verify_orders


def protocol(data, profile):
    """The strongest alpha a record's search could probe, and whether it climbed past its budget."""
    lower, upper = data.get("bounds", (LOWER, FIRST_PROTOCOL_CEILING))
    if lower != LOWER or upper not in (None, FIRST_PROTOCOL_CEILING):
        raise ValueError("Unknown alpha search bounds")
    if upper is None:
        return strongest_alpha(profile, data["selected"]["sigmas"]), True
    return upper, False


def verify(data, receipts):
    settings = {}
    trials_by_key = {}
    if set(data["paths"]) != {"constant", "early", "late"}:
        raise ValueError("Expected all three native schedules")
    for trial in data["trials"]:
        phase, profile, alpha = (trial[k] for k in ("phase", "profile", "alpha"))
        declared = data["cases"][phase]
        rows = trial["rows"]
        if list(rows) != declared or set(data["cases"]["search"]) & set(
            data["cases"]["validation"]
        ):
            raise ValueError("Wrong character panel")
        if trial["gains"] != scheduled_gains(profile, alpha, data["selected"]["sigmas"]):
            raise ValueError("Trial gains differ from its schedule and alpha on the recorded sampler grid")
        scores = {c: score_case(row) for c, row in rows.items()}
        for pair in trial["pairs"]:
            hashes = [rows[c]["on"] for c in pair["cases"]]
            if pair["images"] != hashes:
                raise ValueError("Pair differs from generated images")
            verify_orders(data, pair, hashes, receipts, "repeat")
            if not isclose(
                pair["repeat"],
                (pair["forward"]["value"] + pair["reverse"]["value"]) / 2,
            ):
                raise ValueError("Pair average differs")
        scored = summarize(scores, trial["pairs"])
        if scored != trial["design"] or any(
            row["off"] != data["off"][c] for c, row in rows.items()
        ):
            raise ValueError("Distinct-design trial score differs")
        key = phase, profile, alpha
        if key in settings:
            raise ValueError("Duplicate alpha trial")
        settings[key] = scored["score"]
        trials_by_key[key] = trial
    for old in data["pruned"]:
        key = old["profile"], old["alpha"]
        if key not in {(r["profile"], r["alpha"]) for r in data["historical_bounds"]}:
            raise ValueError("Pruned strength lacks a measured upper bound")
        expected = next(
            r["base"]
            for r in data["historical_bounds"]
            if (r["profile"], r["alpha"]) == key
        )
        if expected != old["upper_bound"]:
            raise ValueError("Pruning upper bound changed")
    # A probe Jev did not rate is unmeasured. It has no score and is never a finalist.
    unmeasured = {(u["phase"], u["profile"], u["alpha"]) for u in data.get("unmeasured", ())}
    if unmeasured & set(settings):
        raise ValueError("A setting is recorded as both measured and unmeasured")

    def searched(profile, alpha):
        key = "search", profile, alpha
        return None if key in unmeasured else settings[key]

    for profile, recorded in data["paths"].items():
        tested, steps = search(lambda a: searched(profile, a), *protocol(data, profile))
        if (
            steps != recorded["steps"]
            or [{"alpha": a, "score": v} for a, v in tested.items()]
            != recorded["trials"]
        ):
            raise ValueError("Adaptive path did not use the distinct-design scores")
        measured = [alpha for alpha in tested if tested[alpha] is not None]
        for alpha in sorted(measured, key=tested.get, reverse=True)[:2]:
            if not {("validation", profile, alpha)} & (set(settings) | unmeasured) and not any(
                (v["profile"], v["alpha"]) == (profile, alpha) for v in data["pruned"]
            ):
                raise ValueError("Missing finalist")
    selected = data["selected"]
    winning = trials_by_key["validation", selected["profile"], selected["alpha"]]
    if (
        selected["effective_strengths"] != winning["gains"]
        or selected["design"] != winning["design"]
    ):
        raise ValueError(
            "Saved strengths or design scores differ from the winning native trial"
        )
    score = winning["design"]["score"]
    if (
        selected["design"]["score"] != score
        or score
        != max(v for (phase, _, _), v in settings.items() if phase == "validation")
        or any(r["upper_bound"] > score for r in data["pruned"])
    ):
        raise ValueError("Saved replay is not the best measured selection")
    return selected


def export(campaign, method, output, run=None):
    """Copy one finished search out of the research host. ``run`` is its results folder."""
    root = (run or campaign / "strength-search/leaderboard-retune-design") / method
    result = read(root / "result.json")
    policy = read(root / "request.json")
    tuning = read(root / "tuning.json")
    if (
        result["selected"] != tuning
        or policy["search_objective"] != tuning["objective"]
    ):
        raise ValueError("Tuning file and completed result differ")
    selected = {
        k: tuning[k]
        for k in (
            "method",
            "objective",
            "profile",
            "alpha",
            "effective_strengths",
            "sigmas",
            "model",
            "design",
        )
    }
    selected.update(
        checkpoint_sha256=tuning["checkpoint"]["sha256"],
        definition_sha256=tuning["definition"]["sha256"],
    )
    rows, proofs, off = [], {}, {}
    store = campaign / "metric-calibration/design-range"
    for phase in ("search", "validation"):
        for path in sorted(root.glob(f"renders/*/*/{phase}.json")):
            r = read(path)
            setting = r["setting"]
            observations = {}
            for c, b in r["rows"].items():
                observations[c] = {
                    k: b[k]
                    for k in (
                        "boss_distinction",
                        "original_boss_distinction",
                        "image_breakage",
                    )
                }
                observations[c].update(on=b["on"]["sha256"], off=b["off"]["sha256"])
                if off.setdefault(c, b["off"]["sha256"]) != b["off"]["sha256"]:
                    raise ValueError("Mismatched original")
            rows.append(
                dict(
                    profile=setting["profile"],
                    alpha=setting["alpha"],
                    phase=phase,
                    rows=observations,
                    pairs=r["pairs"],
                    design=r["design"],
                    seconds=r["seconds"],
                    pair_seconds=r["pair_seconds"],
                    rendered=r["rendered"],
                    gains=setting["effective_strengths"],
                )
            )
            for pair in r["pairs"]:
                for direction in ("forward", "reverse"):
                    key = pair[direction]["receipt"]
                    if key in proofs:
                        continue
                    for name in ("receipts", "receipts-retry-1", "receipts-retry-2"):
                        proof = store / name / (key + ".json")
                        if proof.exists():
                            receipt = read(proof)
                            if receipt["status"] == "ok":
                                proofs[key] = receipt
                                break
                    if key not in proofs:
                        raise ValueError("Missing successful Jev observation")
    data = {
        "model": policy["model"],
        "state": policy["state"],
        "questions": policy["questions"],
        "bounds": policy["bounds"],
        "selected": selected,
        "cases": {
            "search": policy["search_cases"],
            "validation": policy["board_cases"],
        },
        "off": off,
        "trials": rows,
        "paths": result["search"],
        "pruned": result["pruned"],
        "historical_bounds": [
            {
                "profile": r["profile"],
                "alpha": r["alpha"],
                "base": r["means"]["final_boss_score"],
            }
            for r in read(root / "incumbents.json")
        ],
    }
    if result.get("unmeasured"):
        data["unmeasured"] = [{k: u[k] for k in ("phase", "profile", "alpha", "missing")} for u in result["unmeasured"]]
    verify(data, proofs)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tuning.json").write_text(json.dumps(selected, indent=2) + "\n")
    (output / "search.json").write_text(json.dumps(data, indent=2) + "\n")
    (output / "receipts.json.gz").write_bytes(
        gzip.compress(
            json.dumps(proofs, sort_keys=True, separators=(",", ":")).encode(), mtime=0
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--export", type=Path, metavar="CAMPAIGN")
    p.add_argument("--method")
    p.add_argument("--run", type=Path, help="results folder of the search, when it is not the campaign's first design search")
    args = p.parse_args()
    if args.export:
        if not args.method:
            p.error("--export requires --method")
        export(args.export, args.method, args.folder, args.run)
    data = read(args.folder / "search.json")
    with gzip.open(args.folder / "receipts.json.gz", "rt") as f:
        selected = verify(data, json.load(f))
    if selected != read(args.folder / "tuning.json"):
        raise ValueError("Saved tuning differs from verified selection")
    print(
        f"{selected['method']} · {selected['profile']} α {selected['alpha']:g} · {100 * selected['design']['score']:.2f}"
    )
    for profile, path in data["paths"].items():
        strongest, climbed = protocol(data, profile)
        trials = {trial["alpha"]: trial["score"] for trial in path["trials"]}
        side = untried_side(trials, strongest if climbed else None)
        if not climbed and profile == next(iter(data["paths"])):
            print(f"  searched under the withdrawn α {strongest:g} ceiling")
        if side:
            print(f"  {profile}: no probe on the {side} side of its best search alpha")
        elif climbed and max(trials, key=lambda a: (trials[a] is not None, trials[a] or 0)) == strongest:
            print(f"  {profile}: best at α {strongest:g}, the strongest setting Anima applies for this schedule")


if __name__ == "__main__":
    main()
