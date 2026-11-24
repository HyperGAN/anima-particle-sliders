import json
from pathlib import Path

import pytest

from release_tools.final_boss_scoring import BOSS, DAMAGE, ORIGINAL, SCORE, score_case, score_panel


def test_boss_score_rewards_distinct_character_edits_and_visible_readability():
    subtle = {BOSS: 0.35, ORIGINAL: 0.1, DAMAGE: 0.05}
    readable_boss = {BOSS: 0.8, ORIGINAL: 0.1, DAMAGE: 0.25}
    damaged_boss = readable_boss | {BOSS: 0.9, DAMAGE: 0.55}
    assert score_case(readable_boss) > score_case(damaged_boss) > score_case(subtle)
    assert score_case(readable_boss | {ORIGINAL: 0.8}) == 0


def test_panel_uses_qualified_cases_with_equal_weight():
    subtle = {BOSS: 0.35, ORIGINAL: 0.1, DAMAGE: 0.05}
    readable = {BOSS: 0.8, ORIGINAL: 0.1, DAMAGE: 0.25}
    cases = {"subtle": subtle, "readable": readable}
    assert score_panel(cases, "pass") == pytest.approx((score_case(subtle) + score_case(readable)) / 2)
    assert score_panel(cases, "fail") is None
    assert score_panel({"missing": {}}, "unmeasured") is None
    with pytest.raises(ValueError, match="complete Jev judgments"):
        score_panel({"missing": {}}, "pass")


def test_published_rankings_reproduce_from_the_recorded_jev_scores():
    root = Path(__file__).resolve().parents[1] / "docs/final-boss-raw192"
    original = json.loads((root / "results.json").read_text())
    rescored = json.loads((root / "boss-distinction-results.json").read_text())
    for key, rows in rescored["rankings"].items():
        scores = []
        for row in rows:
            if key == "best_saved_replay":
                cases = row["cases"]
            else:
                cases = original["methods"][row["method"]][key]["cases"]
            assert len(cases) == 8
            assert row[SCORE] == pytest.approx(score_panel(cases, "pass"))
            scores.append(row[SCORE])
        assert scores == sorted(scores, reverse=True)
