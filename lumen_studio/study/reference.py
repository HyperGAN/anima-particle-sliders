"""Report the pinned slider effect on exactly matched confirmation cases."""

from pathlib import Path
from statistics import fmean

from particle_sliders.alpha_tune_jev import JEV_MODEL

from ..contracts import digest
from .outputs import read
from .objectives import DISTINCT_BOSS, protocol_id, selection_score


CONFIGS = Path(__file__).resolve().parents[2] / "configs/anima"
REFERENCES = {
    "final boss": CONFIGS / "final-boss-reference.json",
    "1990s anime": CONFIGS / "1990s-anime-reference.json",
    "visually appealing character": CONFIGS / "character-appeal-reference.json",
}


def compare_reference(concept, confirmation, rubric):
    reference = read(REFERENCES[concept.casefold()])
    report = dict(
        reference=reference["name"],
        raw_sha256=reference["raw_sha256"],
        role="study_candidate",
        status="unmatched",
        decision="reference_unmeasured",
    )
    if reference["judge"] != JEV_MODEL or reference["rubric_sha256"] != digest(rubric):
        return report
    expected = {
        (r["prompt_sha256"], r["seed"], r["off_sha256"]): r for r in reference["cases"]
    }
    design = confirmation.get("objective") == DISTINCT_BOSS
    rows = []
    for pair in confirmation["pairs"]:
        case = pair["case"]
        key = (digest(case["prompt"]), case["seed"], case["off"]["output"]["sha256"])
        matched = expected.pop(key, None)
        if (
            matched is None
            or (not design and (pair["gain"] is None or matched["off_score"] != case["off"]["judgment"]["score"]))
        ):
            return report
        rows.append(
            dict(
                seed=case["seed"],
                off_sha256=key[2],
                reference_gain=matched["on_score"] - matched["off_score"],
                reference_integrity=matched["integrity"],
                reference_on_sha256=matched["on_sha256"],
                trial_gain=pair["gain"],
                trial_integrity=pair["on"]["integrity"]["status"],
                trial_on_sha256=pair["on"]["output"]["sha256"],
            )
        )
    if expected:
        return report
    baseline = fmean(r["reference_gain"] for r in rows)
    gains = [r["trial_gain"] for r in rows]
    trial = fmean(gains) if None not in gains else None
    trial_value, reference_value = trial, baseline
    if design:
        pinned = reference.get("distinct_boss", {})
        if not (pinned.get("protocol") == confirmation.get("design_protocol") == protocol_id()):
            return report
        reference_value, trial_value = pinned["score"], selection_score(confirmation)
        if reference_value is None or trial_value is None:
            return report
        report.update(objective=DISTINCT_BOSS, reference_score=reference_value, trial_score=trial_value)
    decision = "retain_reference"
    if confirmation.get("status") == "pass" and trial_value > reference_value:
        decision = "exceeds_reference"
    return dict(
        report,
        status="matched",
        decision=decision,
        cases=rows,
        reference_mean_gain=baseline,
        trial_mean_gain=trial,
        difference=trial_value - reference_value,
    )


def compare_boss_reference(confirmation, rubric):
    """Replay comparisons recorded by the original Final Boss study scripts."""
    return compare_reference("Final Boss", confirmation, rubric)
