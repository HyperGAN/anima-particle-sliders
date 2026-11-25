import json
from pathlib import Path

import pytest
from release_tools.fixed_strength_board import rank_training


def test_archived_methods_share_a_single_strength_and_all_eight_cases():
    source = Path(__file__).resolve().parents[1] / "docs/final-boss-raw192/vibe12/standings.json"
    board = json.loads(source.read_text())
    ranked = rank_training(board, 0.6375)
    assert len(ranked) == 24
    assert ranked[0]["method"] == "strong-boss-45-mix35"
    assert ranked[0]["mean_gain"] == pytest.approx(0.3225)
    assert ranked[4]["rank"] == ranked[5]["rank"] == 5
    with pytest.raises(ValueError, match="Every method"):
        rank_training(board, 1.0)

    row = next(r for r in board["settings"] if r["alpha"] == 0.6375 and r["profile"] == "constant")
    row["gains"][0] *= 1.5
    with pytest.raises(ValueError, match="Unequal"):
        rank_training(board, 0.6375)


def test_published_scalar_board_pins_one_raw_and_replays_matched_scores():
    from statistics import fmean

    folder = Path(__file__).resolve().parents[1] / "docs/final-boss-raw192/vibe12"
    evidence = json.loads((folder / "constant-leaderboard.json").read_text())
    ranking = rank_training(json.loads((folder / "standings.json").read_text()), evidence["benchmark_alpha"])
    assert [row["method"] for row in ranking] == [r["method"] for r in evidence["methods"]]
    winner = next(r for r in evidence["methods"] if r["training_status"] == r["benchmark_status"] == "pass")
    assert winner["method"] == evidence["selected"]
    boss, anime, appeal = evidence["sliders"]
    assert boss["raw_sha256"] == winner["raw_sha256"]
    for slider in (boss, anime, appeal):
        chosen, = [t for t in slider["trials"] if t["alpha"] == slider["alpha"]]
        assert chosen["status"] == "pass"
        assert all(t["gains"] == [t["alpha"]] * 10 for t in slider["trials"])
        assert len(slider["cases"]) == 4
        assert slider["confirmation_gain"] == pytest.approx(fmean(p["trial_gain"] for p in slider["cases"]))
        assert not {p["seed"] for p in chosen["pairs"]} & {p["seed"] for p in slider["cases"]}
