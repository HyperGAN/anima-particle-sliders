"""Publish the live optimization without changing the original training leaderboard."""

from html import escape
import json
from pathlib import Path
from statistics import fmean
import sys
from time import sleep

from boss_review import FOLDER, saved_distinction
from image_breakage import CACHE as BREAKAGE_CACHE, saved as saved_breakage, summarize as summarize_breakage
from optimization_targets import ARCHIVE
from optimization_worker import PRIOR, ROOT, pin, read, write
from score import (BREAKAGE, DISTINCTION, EFFECT, METRIC, OBJECTIVE, ORIGINAL_DISTINCTION, PRESERVATION, PRIOR_REPORTS,
                   REPORT, TOTAL_HELP, adjusted_effect, case_effect, preservation)


sys.path.append(str(PRIOR))
from build_gallery import PUBLIC

NAMES = {
    "incumbent": "Soft weighting · saved adapter",
    "prompt-only": "Prompt-only",
    "soft-contexts-65-blend50": "Soft weighting + paired contexts",
    "soft-projection-65-blend50": "Soft weighting + lighting / scene projection",
    "soft-floor45": "Soft weighting · background 0.45",
    "soft-floor80": "Soft weighting · background 0.80",
    "soft45-background-orthogonal": "Exterior lighting and scene projection",
    "soft45-adapter-rollout": "Training on saved-slider trajectories",
    "confirm0": "Independent comparison, first four cases",
    "confirm1": "Independent comparison, second four cases",
}
PAGE = PUBLIC / "final-boss-optimization.html"
FORMULATIONS = PUBLIC / "final-boss-formulations.html"
FORMULATIONS_CONSTANT = PUBLIC / "final-boss-formulations-constant.html"


def display_method(method):
    if method in NAMES:
        return NAMES[method]
    path = ROOT / "plans" / f"{method}.json"
    if not path.is_file():
        return method
    recipe = read(path)
    if recipe.get("teacher") == "explicit-boss-cues-low-mix-v1":
        return f"Explicit Final Boss teacher · {recipe['cue_mix']:.0%} cue mix · background {recipe['floor']:.2f}"
    if recipe.get("teacher") in ("explicit-boss-cues-v1", "explicit-boss-cues-low-floor-v1"):
        return f"Explicit Final Boss teacher · {recipe['target_dose']:.0%} target strength · background {recipe['floor']:.2f}"
    if recipe["donor"] is None:
        return f"Soft weighting · background {recipe['floor']:.2f}"
    donor = {"nuisance-projection": "lighting and scene projection", "protected-contexts": "paired contexts",
             "prompt-consensus": "prompt consensus"}[recipe["donor"]]
    return (f"Soft weighting + {donor} · {recipe['blend']:.0%} blend"
            f" · background {recipe['floor']:.2f}")


def sample(record, name):
    public = PUBLIC / "images" / name
    if public.is_file() and pin(public)["sha256"] == record["sha256"]:
        return f"images/{name}"
    source = Path(record["path"])
    with source.open("rb") as stream:
        signature = stream.read(8)
    if signature != b"\x89PNG\r\n\x1a\n" or pin(source)["sha256"] != record["sha256"]:
        raise ValueError(f"Scored render changed: {source}")
    pending = public.with_suffix(".link")
    pending.unlink(missing_ok=True)
    pending.symlink_to(source)
    pending.replace(public)
    return f"images/{name}"


def scored_row(row, reference_hashes):
    cases = {}
    severities = {}
    for case, values in row["cases"].items():
        off_sha = reference_hashes[case]
        on_sha = pin(PUBLIC / row["images"][case])["sha256"] if case in row["images"] else None
        severities[case] = saved_breakage(off_sha, on_sha) if on_sha else None
        if values is None:
            cases[case] = None
            continue
        value = values.get(DISTINCTION)
        if DISTINCTION not in values and on_sha:
            value = saved_distinction(off_sha, on_sha)
        before = values.get(ORIGINAL_DISTINCTION) if ORIGINAL_DISTINCTION in values else saved_distinction(off_sha, off_sha)
        scored = values | {DISTINCTION: value, ORIGINAL_DISTINCTION: before}
        effect = case_effect(scored)
        damage = severities[case]
        cases[case] = scored | {EFFECT: effect, BREAKAGE: damage, METRIC: adjusted_effect(effect, damage),
                                 PRESERVATION: preservation(scored)}
    means = row["means"]
    review = summarize_breakage(severities)
    if means is not None:
        metrics = {}
        for key in (EFFECT, METRIC):
            complete = all(case is not None and case[key] is not None for case in cases.values())
            metrics[key] = round(fmean(case[key] for case in cases.values()), 12) if complete else None
        means = means | metrics | {BREAKAGE: review["mean"], PRESERVATION: preservation(means)}
    measured = row.get("status") != "unmeasured" and means is not None and means[METRIC] is not None
    qualification = "fail" if row.get("status") == "fail" else "pass" if measured else "unmeasured"
    return row | {"means": means, "cases": cases, "readability": review, "qualification": qualification}


def gallery_data(baseline, cases, references, rows):
    return {
        "metrics": {METRIC: ["Final score", TOTAL_HELP],
                    EFFECT: ["Final Boss effect", "Average matched change in Jev's character-only Final Boss rating."],
                    BREAKAGE: ["Image damage", "Average Jev severity of visible detail loss or malformed forms. Lower is better."],
                    **{key: value for key, value in baseline["metrics"].items() if key not in ("overall", "character")}},
        "effect_question": baseline["effect_question"],
        "model": baseline.get("model"),
        "cases": cases,
        "references": references,
        "rows": rows,
        "selection_metric": METRIC,
        "tiebreak_metric": PRESERVATION,
        "objective": OBJECTIVE,
        "head_to_head": None,
        "baseline_match": None,
    }


def write_gallery(destination, board, *, title, intro, links, status=""):
    document = (ROOT / "optimization_gallery.html").read_text()
    nav = "".join(f'<a href="{escape(url, quote=True)}">{escape(name)}</a>' for url, name in links)
    payload = json.dumps(board, allow_nan=False).replace("<", "\\u003c")
    for token, value in (("TITLE", escape(title)), ("INTRO", escape(intro)), ("NAV", nav), ("STATUS", status), ("DATA", payload)):
        document = document.replace(f"%%{token}%%", value)
    pending = destination.with_suffix(".partial")
    pending.write_text(document)
    pending.replace(destination)


def published_rows(baseline, reports, references):
    rows = {}
    reference_hashes = {case: pin(PUBLIC / path)["sha256"] for case, path in references.items()}
    wanted = {"soft-subject": (2 / 3, "incumbent"), "prompt-only": (1.2, "prompt-only")}
    for identifier, row in baseline["rows"].items():
        if row["method"] not in wanted or (row["alpha"], row["profile"]) != (wanted[row["method"]][0], "early"):
            continue
        label = NAMES[wanted[row["method"]][1]]
        images = {case: f"images/{case}-signal-study-{identifier}.png" for case in baseline["cases"]}
        rows[identifier] = scored_row(row | {"label": label, "images": images}, reference_hashes)
    for report in reports:
        method = report["method"]
        title = display_method(method)
        if report.get("search_running"):
            title += " · search in progress"
        for setting in report.get("reviewed_settings", report["qualifying_settings"]):
            profile, alpha = setting["profile"], setting["alpha"]
            identifier = f"{method}-{profile}-{alpha.hex()}"
            if method == "incumbent" and (profile, alpha) == ("early", 2 / 3):
                rows.pop(f"soft-subject-early-{alpha.hex()}", None)
            cases = setting["cases"]
            images = {case: sample(item["on"], f"{case}-opt-{identifier}.png")
                      for case, item in cases.items()}
            values = {case: item["scores"] | {"gain": item["gain"]} if item["scores"] is not None else None
                      for case, item in cases.items()}
            checks = {item["integrity"]["status"] for item in cases.values()}
            status = "fail" if "fail" in checks else "pass" if checks == {"pass"} else "unmeasured"
            rows[identifier] = scored_row({"method": method, "label": title, "profile": profile, "alpha": alpha,
                                             "means": setting["means"], "cases": values, "images": images,
                                             "status": status}, reference_hashes)
    return rows


def status_text():
    paths = [ROOT / "campaign-status.json", *(ROOT / "waves").glob("*/campaign-status.json")]
    observed = [read(path) for path in paths if path.exists()]
    live = {job["method"]: job for status in observed for job in status["live"]}
    terminal = {job["method"]: job for status in observed for job in status["terminal"]}
    formulation = read(ROOT / "search-protocol.json").get("formulation_comparison", {})
    rows = []
    planned = [path.stem for path in sorted((ROOT / "plans").glob("*.json"))]
    for method in dict.fromkeys((*planned, *live, *terminal)):
        if method in live:
            job = live[method]
            stage = ("Training" if job["stage"] == "train" else "Evaluating shared strengths"
                     if method in formulation.get("methods", ()) else "Searching alpha")
            status = stage + f" on GPU {job['gpu']}"
        elif method in terminal:
            job = terminal[method]
            complete = ("Training complete" if job["stage"] == "train" else "Evaluated at shared strengths"
                        if method in formulation.get("methods", ()) else "Tuned")
            status = {"complete": complete,
                      "scientific-stop": "Stopped by the image check",
                      "admission-refused": "Waiting for GPU capacity"}.get(job["outcome"], "Needs recovery")
        else:
            status = "Queued"
        milestones = sorted((ROOT / "cohorts" / method / "milestones").glob("RAW*.json"))
        last = f" · {read(milestones[-1])['step']}/192 updates reviewed" if milestones else ""
        rows.append(f"<li>{escape(display_method(method))} · {escape(status + last)}</li>")
    incumbent = ROOT / "incumbent-cache/report.json"
    if incumbent.exists():
        rows.append("<li>Saved soft adapter · 16 archived strengths rescored</li>")
    return "<details><summary>Training and search progress</summary><ul>" + "".join(rows) + "</ul></details>"


def publish_formulations(baseline, cases, references, rows, config):
    pages = ((FORMULATIONS, config["shared_strength"], FORMULATIONS_CONSTANT, "Shared constant strength"),
             (FORMULATIONS_CONSTANT, config["secondary_control"], FORMULATIONS, "Shared early strength"))
    for path, strength, alternative, label in pages:
        profile, alpha = strength
        fixed = {key: row for key, row in rows.items() if (row["profile"], row["alpha"]) == (profile, alpha)}
        board = gallery_data(baseline, cases, references, fixed)
        write_gallery(path, board, title="Final Boss training formulations",
                      intro=f"Every row uses 192 updates and {profile} α{alpha:.4g} on the same eight development cases.",
                      links=[(alternative.name, label), ("final-boss-optimization.html", "All evaluated settings")],
                      status=status_text())


def publish():
    baseline = read(ARCHIVE / "jev-review/report.json")
    trials = []
    measured = set()
    for method in sorted((ROOT / "tuning").glob("*")):
        ranked = method / REPORT
        candidates = (ranked, *(method / name for name in PRIOR_REPORTS))
        source = next((path for path in candidates if path.is_file()), None)
        if source is not None:
            report = read(source)
            if source == ranked and (report["selection_metric"] != METRIC or report["objective"] != OBJECTIVE):
                raise ValueError(f"{method.name} uses a different score")
            trials.append(report)
            measured.add(method.name)
            continue
        settings = [read(path) for path in sorted(method.glob("development/*/*.json"))]
        passing = [setting for setting in settings if setting["status"] == "pass"]
        if passing:
            trials.append({"method": method.name, "qualifying_settings": passing, "search_running": True})
    cases = baseline["cases"]
    references = {case: f"images/{case}-signal-study-off.png" for case in cases}
    rows = published_rows(baseline, trials, references)
    board = gallery_data(baseline, cases, references, rows)
    protocol = read(ROOT / "search-protocol.json")
    benchmark = protocol["experiment_target"]
    intro = (f"Eight matched development cases at RAW192. The benchmark is {100 * benchmark['development_score']:.1f}"
             f" points at {benchmark['profile']} α{benchmark['alpha']:.4g}.")
    links = [("final-boss-jev-review.html", "Original Jev review")]
    if formulation := protocol.get("formulation_comparison"):
        publish_formulations(baseline, cases, references, rows, formulation)
        links.insert(0, ("final-boss-formulations.html", "Formulations at shared strength"))
    if (PUBLIC / "final-boss-optimization-confirmed.html").exists():
        links.append(("final-boss-optimization-confirmed.html", "Independent confirmation"))
    write_gallery(PAGE, board, title="Final Boss optimization", intro=intro, links=links, status=status_text())
    write(ROOT / "leaderboard.json", {"scope": "development", "cases": cases, "rows": rows,
                                     "selection_metric": METRIC, "objective": OBJECTIVE})
    print(f"Published {len(rows)} settings and {len(measured)} completed alpha searches", flush=True)


def watch():
    previous = None
    while True:
        paths = [ROOT / "campaign-status.json", ROOT / "incumbent-cache/report.json", ROOT / "optimization_gallery.html",
                 PUBLIC / "final-boss-optimization-confirmed.html"]
        for name in (REPORT, *PRIOR_REPORTS):
            paths.extend(ROOT.glob(f"tuning/*/{name}"))
        paths.extend(ROOT.glob("tuning/*/development/*/*.json"))
        paths.extend(ROOT.glob("cohorts/*/milestones/RAW*.json"))
        paths.extend(ROOT.glob("waves/*/campaign-status.json"))
        paths.extend(FOLDER.glob("*.json"))
        paths.extend(BREAKAGE_CACHE.glob("*.json"))
        state = {path: path.stat().st_mtime_ns for path in paths if path.exists()}
        if state != previous:
            publish()
            previous = state
        sleep(25)


if __name__ == "__main__":
    watch() if sys.argv[1:] == ["watch"] else publish()
