"""Publish the shared-target training comparison as qualified replays arrive."""

import argparse
from statistics import fmean
import sys

from control import ROOT, SOURCE, STORAGE


sys.path.insert(0, str(SOURCE / "release_tools"))
from final_boss_scoring import BOSS, DAMAGE, GAIN, ORIGINAL, SCORE, VERSION, score_case, score_panel

from optimization_worker import PRIOR, pin, read, write
from publish_optimization import PUBLIC, gallery_data, sample, write_gallery
LABELS = read(ROOT / "recipes.json")


CONDITIONS = {
    "fixed_strength": ("early", 2 / 3, "final-boss-vibe12-leaderboard.html"),
    "matched_constant": ("constant", .6375, "final-boss-vibe12-leaderboard-constant.html"),
}
REPORT = "boss-effect-tuning.json"


def facts():
    files = [
        *ROOT.glob("waves/*/campaign-*.json"),
        *(STORAGE / "targets").glob("receipt.json"),
        *(ROOT.glob("campaign-state.json")),
        *((ROOT / "tuning").glob(f"*/{REPORT}")),
        *((ROOT / "tuning").glob("*/score-recovery/*.json")),
        *(path for profile, alpha, _ in CONDITIONS.values()
          for path in (ROOT / "strength-search/leaderboard-retune-damage").glob(
              f"*/renders/{profile}/{alpha.hex()}/validation.json")),
        *((ROOT / "cohorts").glob("*/milestones/RAW*.json")),
        *((ROOT / "cohorts").glob("*/result.json")),
    ]
    return [(path, path.stat().st_mtime_ns) for path in sorted(files)]


def publish():
    history = read(SOURCE / "docs/final-boss-raw192/results.json")
    baseline = read(ROOT.parent / "anima-final-boss-signal-raw192-20261123/jev-review/report.json")
    case_ids = [item["label"] for item in history["cases"]]
    originals = {case: read(PRIOR / "shared-off" / case / "render.json")["image"] for case in case_ids}
    references = {case: sample(item, f"ranked-{item['sha256']}.png") for case, item in originals.items()}
    snapshot = {"scoring_version": VERSION, "cases": case_ids, "rankings": {}, "qualification": {}}
    receipt_path = STORAGE / "targets" / "receipt.json"
    if receipt_path.is_file():
        snapshot["teacher"] = read(receipt_path)
    for key, (profile, alpha, path) in CONDITIONS.items():
        rows = {}
        for method, label in LABELS.items():
            report = ROOT / "tuning" / method / REPORT
            cohort = ROOT / "cohorts" / method
            milestones = sorted((cohort / "milestones").glob("RAW*.json"))
            row = {"method": method, "label": label, "profile": profile, "alpha": alpha, "images": {},
                   "cases": {}, "qualification": "unmeasured", "means": None, "readability": {"status": "pending"}}
            if milestones:
                milestone = read(milestones[-1])
                failed = any(case["integrity"]["status"] == "fail" for case in milestone["cases"])
                row["label"] += f" · RAW{milestone['step']}" + (" · stopped" if failed else "")
                if failed:
                    row["qualification"] = "fail"
            if report.is_file():
                native = read(report)
                result = read(cohort / "result.json")
                if (native["raw"] != result["raw"] or result["step"] != 192 or result["ema_decay"] is not None
                        or native["development_cases"] != case_ids or native["sampler"] != history["sigmas"]):
                    raise ValueError(f"Evaluation does not match the declared RAW192 checkpoint for {method}")
                fixed = native[key]
                screen = native["screen"][profile][str(alpha)]
                follow_up = report.parent / "score-recovery" / f"{key}.json"
                if follow_up.is_file():
                    checked = read(follow_up)
                    if checked["native_report"] != pin(report)["sha256"]:
                        raise ValueError(f"Follow-up review is bound to a different native report for {method}")
                    for group, reviewed in ((fixed, checked["fixed"]), (screen, checked["screen"])):
                        if [(case, row["off"], row["on"]) for case, row in group["cases"].items()] != [
                            (case, row["off"], row["on"]) for case, row in reviewed["cases"].items()
                        ]:
                            raise ValueError(f"Follow-up review images changed for {method}")
                    fixed, screen = checked["fixed"], checked["screen"]
                control_path = (ROOT / "strength-search/leaderboard-retune-damage" / method / "renders" /
                                profile / alpha.hex() / "validation.json")
                control = read(control_path) if control_path.is_file() else None
                if control and control["setting"] != {"profile": profile, "alpha": alpha,
                        "cases": case_ids, "effective_strengths": fixed["gains"]}:
                    raise ValueError("Fixed strength differs from the current alpha search")
                if list(fixed["cases"]) != case_ids or (fixed["profile"], fixed["alpha"]) != (profile, alpha):
                    raise ValueError(f"The fitted strength or development cases changed for {method}")
                cases, images = {}, {}
                for case, measured in fixed["cases"].items():
                    if measured["off"]["sha256"] != originals[case]["sha256"]:
                        raise ValueError(f"Shared original changed for {method}/{case}")
                    on = measured["on"]
                    images[case] = sample(on, f"ranked-{on['sha256']}.png")
                    ratings = (measured["scores"] or {}) | {
                        DAMAGE: (fixed.get("readability") or {}).get("severity", {}).get(case),
                    }
                    if control:
                        scored = control["rows"][case]
                        if (scored["on"]["sha256"], scored["off"]["sha256"]) != (on["sha256"], originals[case]["sha256"]):
                            raise ValueError("Fixed images differ from the current alpha search")
                        ratings = scored
                    boss, original = ratings.get(BOSS), ratings.get(ORIGINAL)
                    cases[case] = {name: ratings.get(name) for name in (BOSS, ORIGINAL, DAMAGE)} | {
                        GAIN: None if boss is None or original is None else max(0, boss - original),
                        SCORE: score_case(ratings), "on_sha256": on["sha256"],
                        "off_sha256": measured["off"]["sha256"], "integrity": measured["integrity"]["status"],
                    }
                status = "pass" if all(case[SCORE] is not None for case in cases.values()) else "unmeasured"
                overall = fmean(case[SCORE] for case in cases.values()) if status == "pass" else None
                means = {field: fmean(value[field] for value in cases.values())
                         for field in (BOSS, GAIN, DAMAGE) if all(value[field] is not None for value in cases.values())}
                row = row | {"label": label, "qualification": status, "means": means | {SCORE: overall},
                             "images": images, "cases": cases, "raw_sha256": result["raw"]["sha256"],
                             "cache": result["cache"], "gains": fixed["gains"], "readability": {"status": "measured"}}
                row["screen_status"] = screen["status"]
                if follow_up.is_file():
                    row["follow_up_review_sha256"] = pin(follow_up)["sha256"]
            rows[method] = row
        board = gallery_data(baseline, case_ids, references, rows)
        board.update({"scoring_version": VERSION, "metrics": {
            SCORE: ["Final Boss score", "Positive Boss gain weighted by finished Boss distinction and visible damage."],
            BOSS: ["Boss distinction", "Jev's rating of the edited character."],
            GAIN: ["Boss gain", "Positive change compared with the same render with the slider off."],
            DAMAGE: board["metrics"][DAMAGE],
        }, "selection_metric": SCORE, "tiebreak_metric": None, "show_original_gain": False,
            "scoring_description": "Positive Boss gain is weighted by the edited character’s Boss distinction and reduced by visible damage.",
            "objective": "Compare training recipes on the same twelve frozen Final Boss scenes."})
        ranked = sorted((row for row in rows.values() if row["qualification"] == "pass"),
                        key=lambda row: row["means"][SCORE], reverse=True)
        with_replays = sum(bool(row["images"]) for row in rows.values())
        scope = (f"{profile.title()} α {alpha:.4g} · RAW192 · {with_replays} of {len(rows)} recipes with samples"
                 f" · {len(ranked)} ranked")
        links = [(item[2], "Switch matched strength") for name, item in CONDITIONS.items() if name != key]
        if (PUBLIC / "final-boss-vibe12-alpha-leaderboard.html").is_file():
            links.insert(0, ("final-boss-vibe12-alpha-leaderboard.html", "Damage-adjusted alpha leaderboard"))
        links.insert(0, ("final-boss-vibe-corrected-teachers.html", "Twelve matched teacher pairs"))
        links.append(("final-boss-ranked.html", "Previous leaderboard"))
        links.append(("final-boss-ranked-seed-split.html", "Second-seed samples"))
        if (PUBLIC / "final-boss-shared-stock-alpha.html").is_file():
            links.append(("final-boss-shared-stock-alpha.html", "Stock alpha sweep"))
        write_gallery(PUBLIC / path, board, title="Final Boss · 12-scene retraining", intro=scope, links=links)
        snapshot["rankings"][key] = [{name: value for name, value in row.items() if name != "images"}
                                      for row in ranked]
        snapshot["qualification"][key] = {method: row["qualification"] for method, row in rows.items()}
    write(STORAGE / "leaderboard.json", snapshot)
    return snapshot


def main():
    from time import sleep

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    previous = None
    while True:
        revision = facts()
        if revision != previous:
            publish()
            previous = revision
        if not args.watch:
            break
        sleep(25)


if __name__ == "__main__":
    main()
