"""Freeze evidence for offline replay and quantify how sensitive the rank is to individual characters."""

import gzip
import json
from hashlib import sha256
from statistics import fmean
from locations import RESULTS, STUDY
from lumen_studio.study.objectives import protocol, protocol_id
from release_tools.constant_alpha_retest import score_panel, verify_snapshot

web = STUDY / "sample-gallery/constant-retest"
out = RESULTS / "verified"
out.mkdir(exist_ok=True)


def load(path):
    return json.loads(path.read_bytes())


def dump(name, data):
    path = out / name
    path.with_suffix(".pending").write_bytes(data)
    path.with_suffix(".pending").replace(path)


board = load(web / "constant-retest.json")
assert (
    board["objective"] == "distinct_boss_score"
    and board["completed"] == board["total"] == 38
)
assert len(board["other_sliders"]) == 2 and all(
    "score" in r for r in board["other_sliders"]
)
assert board["finalist_confirmation"]
methods = {
    r["method"]: load(web / (r["method"] + "-distinct_boss_score.json"))
    for r in board["recipes"]
}
ranked = [r for r in board["recipes"] if r.get("rank") is not None]
snapshot = dict(
    protocol=protocol_id(),
    judge=protocol()["model"],
    objective=board["objective"],
    panels=board["panels"],
    methods=methods,
    ranking=[{k: r[k] for k in ("method", "rank", "score")} for r in ranked],
)
order = verify_snapshot(snapshot)
numeric = {}
for case in board["panels"]["ranking"]:
    scores = {}
    for n in order:
        row = methods[n]["ranking"]
        scores[n] = score_panel(
            {k: v for k, v in row["rows"].items() if k != case},
            board["objective"],
            [pair for pair in row["pairs"] if case not in pair["cases"]],
        )
    numeric[case] = {n: 1 + sum(s > scores[n] for s in scores.values()) for n in order}
historical = {
    r["method"]: r["previous_design"]
    for r in ranked
    if r["previous_design"] is not None
}
constant = {r["method"]: r["score"] for r in ranked if r["method"] in historical}


def place(scores, n):
    return 1 + sum(v > scores[n] for v in scores.values())


old_board = load(STUDY / "product/docs/final-boss-raw192/vibe12/design-range.json")
prior_settings = {row["method"]: row for row in old_board["development"]["rows"]}
brief = []
for r in board["recipes"]:
    n = r["method"]
    data = {
        k: r.get(k)
        for k in (
            "rank",
            "method",
            "label",
            "pg",
            "step",
            "training",
            "raw_sha256",
            "alpha",
            "score",
            "probes",
            "search_score",
            "mean_gain",
            "integrity",
            "previous_design",
        )
    }
    data["original_measurement_recovered"] = r.get("original_measurement") is not None
    data["missing_search_scores"] = sum(
        t["score"] is None for t in methods[n]["search"]
    )
    if n in order:
        data["leave_one_out_ranks"] = {case: t[n] for case, t in numeric.items()}
    if n in historical:
        data["original_24_prior_rank"] = place(historical, n)
        data["previous_profile"] = prior_settings[n]["profile"]
        data["previous_alpha"] = prior_settings[n]["alpha"]
        data["original_24_constant_rank"] = place(constant, n)
    brief.append(data)
compressed = gzip.compress(
    json.dumps(
        snapshot, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode(),
    mtime=0,
)
dump("constant-alpha-evidence.json.gz", compressed)
summary = dict(
    protocol=protocol_id(),
    judge=protocol()["model"],
    panels_sha256=sha256((RESULTS / "panels.json").read_bytes()).hexdigest(),
    search_implementation_sha256=sha256(
        (STUDY / "product/lumen_studio/study/scalar_search.py").read_bytes()
    ).hexdigest(),
    objective=board["objective"],
    score_display_multiplier=100,
    panels=board["panels"],
    replay_pg=board["replay_pg"],
    trained=40,
    final_boss_retested=38,
    completed_fit_ranks=len(order),
    alpha_trials=sum(r["probes"] for r in brief),
    historical_replay_images=sum(
        len(methods[n]["replay"]["cases"]) for n in historical
    ),
    historical_matched_24=dict(
        prior_mean_score=fmean(historical.values()),
        constant_mean_score=fmean(constant.values()),
        constant_higher=sum(constant[n] > historical[n] for n in historical),
    ),
    evidence_sha256=sha256(compressed).hexdigest(),
    recipes=brief,
    other_sliders=board["other_sliders"],
    finalist_confirmation=board["finalist_confirmation"],
)
dump(
    "constant-alpha-retest.json",
    (
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode(),
)
for name in ("constant-alpha-evidence.json.gz", "constant-alpha-retest.json"):
    p = web / name
    p.with_suffix(".pending").write_bytes((out / name).read_bytes())
    p.with_suffix(".pending").replace(p)
print(
    f"{len(order)} ranked recipes, {summary['alpha_trials']} Final Boss search probes, {len(compressed)} bytes of replay evidence"
)
