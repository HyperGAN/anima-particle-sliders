import json
from pathlib import Path
from statistics import fmean

import pytest
import torch

from release_tools.final_boss_targets import RECIPES, edited_target


STUDY = Path(__file__).resolve().parents[1] / "docs/final-boss-raw192"


def test_default_uses_the_winning_fixed_recipe_and_confirmed_strength():
    default = json.loads((STUDY / "default.json").read_text())
    results = json.loads((STUDY / "results.json").read_text())
    eligible = {name: row["fixed_strength"]["means"]["final_score"] for name, row in results["methods"].items()
                if row.get("fixed_strength", {}).get("status") == "pass"}
    assert default["recipe"] == max(eligible, key=eligible.get)
    assert default["recipe"] in RECIPES
    checkpoint = default["checkpoints"][default["recipe"]]
    assert checkpoint["sha256"] == results["methods"][default["recipe"]]["raw_sha256"]

    confirmation = results["confirmation"]
    chosen = confirmation["qualified"][confirmation["default"]]
    assert chosen["gains"] == default["gains"]
    assert chosen["profile"] == default["profile"] and chosen["alpha"] == default["alpha"]
    assert chosen["means"]["final_score"] == max(row["means"]["final_score"] for row in confirmation["qualified"].values())
    assert fmean(case["final_score"] for case in chosen["cases"].values()) == pytest.approx(default["independent_score"])


@pytest.mark.parametrize("recipe", ("soft-floor45", "strong-boss-45-mix35"))
def test_fitted_teacher_contrast_keeps_the_original_l2_dose(recipe):
    floor, cue_mix = RECIPES[recipe]
    neutral = torch.tensor([2.0, 3.0])
    positive = neutral + torch.tensor([1.0, 1.0])
    explicit = neutral + torch.tensor([3.0, 1.0]) if cue_mix else None
    weight = torch.tensor([1.0, floor])

    edit = edited_target(neutral, positive, weight, cue_mix, explicit) - neutral

    torch.testing.assert_close(edit.norm(), (positive - neutral).norm())
    assert edit[1] / edit[0] == pytest.approx(floor / (1 + 2 * cue_mix))
