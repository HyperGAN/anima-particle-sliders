"""Recompute published objectives and audit each reserved scalar."""

import math
from lumen_studio.study.scalar_search import (
    search_constant as search_constant,
    verify_search,
)


def score_panel(rows, objective, pairs=()):
    """Compute the registered Final Boss metric on a complete matched panel."""
    from statistics import fmean
    from .final_boss_design_scores import summarize
    from .final_boss_visible_damage import score_case

    if objective not in ("matched_boss_gain", "distinct_boss_score"):
        raise ValueError(f"Unknown Final Boss objective {objective}")
    if not rows:
        raise ValueError("An empty panel cannot be scored")
    values = [
        row.get(key)
        for row in rows.values()
        for key in ("boss_distinction", "original_boss_distinction")
    ]
    if any(x is None for x in values):
        return None
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("Invalid Boss intensity")
    if objective == "matched_boss_gain":
        return fmean(
            r["boss_distinction"] - r["original_boss_distinction"]
            for r in rows.values()
        )
    cases = {case: score_case(row) for case, row in rows.items()}
    if any(v is None for v in cases.values()) or any(p is None for p in pairs):
        return None
    for pair in pairs:
        hashes = [rows[c]["on_sha256"] for c in pair["cases"]]
        if hashes != pair["images"] or pair["repeat"] != fmean(
            pair[k]["value"] for k in ("forward", "reverse")
        ):
            raise ValueError("Design judgment does not match the two images")
    return summarize(cases, pairs)["score"]


def verify_recipe(data, panels):
    """Reproduce the search, reservation and ranking score from the exported evidence."""
    if set(panels["search"]) & set(panels["ranking"]):
        raise ValueError("Search and ranking cases overlap")
    if (data["ranking"] is None) != (data["alpha"] is None) or (
        data["alpha"] is None and data["score"] is not None
    ):
        raise ValueError(
            "A reserved alpha requires its own ranking panel and measured or missing score"
        )
    record = []
    for phase, trials in (("search", data["search"]), ("ranking", [data["ranking"]])):
        for trial in trials:
            if trial is None:
                continue
            if (
                list(trial["rows"]) != panels[phase]
                or trial["gains"] != [trial["alpha"]] * 10
            ):
                raise ValueError("Unexpected cases or nonconstant strengths")
            measured = score_panel(trial["rows"], data["objective"], trial["pairs"])
            if measured != trial["score"]:
                raise ValueError("Trial objective differs from its measurements")
            if phase == "search":
                record.append((trial["alpha"], measured))
            elif trial["alpha"] != data["alpha"] or measured != data["score"]:
                raise ValueError(
                    "Ranking did not use the reserved scalar and objective"
                )
    return verify_search(record, data["alpha"])


def verify_snapshot(snapshot):
    """Recompute the per-recipe scalar and every numerical rank without a GPU or judge."""
    if snapshot["objective"] != "distinct_boss_score":
        raise ValueError(
            "Final Boss leaderboard requires its registered design objective"
        )
    panels = snapshot["panels"]
    if len(set(panels["search"])) != 4 or len(set(panels["ranking"])) != 8:
        raise ValueError("Expected four tuning characters and eight ranking characters")
    observed = {}
    for method, doc in snapshot["methods"].items():
        if method != doc["method"] or doc["objective"] != snapshot["objective"]:
            raise ValueError("Mixed recipe or objective in the Final Boss snapshot")
        if (
            not doc["replay"]["cases"]
            or doc["raw_sha256"] != doc["replay"]["raw_sha256"]
            or any(r["expected"] != r["actual"] for r in doc["replay"]["cases"])
        ):
            raise ValueError("Incomplete RAW or replay binding")
        verify_recipe(doc, panels)
        if doc["training"] == "completed" and doc["score"] is not None:
            observed[method] = doc["score"]
    order = sorted(observed, key=lambda name: (-observed[name], name))
    displayed = [
        (row["method"], row["rank"], row["score"]) for row in snapshot["ranking"]
    ]
    expected = [
        (
            name,
            1 + sum(value > observed[name] for value in observed.values()),
            observed[name],
        )
        for name in order
    ]
    if expected != displayed:
        raise ValueError("Published ranks differ from the measured objectives")
    return order


if __name__ == "__main__":
    import argparse
    import gzip
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=verify_snapshot.__doc__)
    parser.add_argument(
        "snapshot",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1]
        / "docs/final-boss-raw192/vibe12/constant-alpha-evidence.json.gz",
    )
    from hashlib import sha256

    source = parser.parse_args().snapshot
    raw = source.read_bytes()
    evidence = json.loads(gzip.decompress(raw))
    manifest = json.loads(source.with_name("constant-alpha-retest.json").read_bytes())
    if sha256(raw).hexdigest() != manifest["evidence_sha256"]:
        raise ValueError("Scoring evidence differs from its published digest")
    roster = {r["method"]: r["raw_sha256"] for r in manifest["recipes"]}
    if (
        set(roster) != set(evidence["methods"])
        or len(roster) != manifest["final_boss_retested"]
        or any(evidence["methods"][n]["raw_sha256"] != sha for n, sha in roster.items())
    ):
        raise ValueError(
            "Frozen recipe roster or RAW identity differs from the published inventory"
        )
    ranked = verify_snapshot(evidence)
    print(
        f"Verified {len(evidence['methods'])} Final Boss recipes, {len(ranked)} ranked. Winner: {ranked[0]}"
    )
