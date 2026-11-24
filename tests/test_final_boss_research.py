import json
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

import pytest
import torch

from release_tools.final_boss_targets import RECIPES, STEPS, capture_row, edited_target


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


def test_new_cue_recipes_reuse_trajectories_and_apply_the_declared_contrast():
    class Runtime:
        device = "cpu"

        def encode(self, prompt):
            velocities = {"plain": (0, 0), "boss": (2, 3), "specific": (6, 1), "other": (1, 9)}
            return torch.tensor(velocities[prompt], dtype=torch.float32).reshape(1, 1, 1, 1, 2)

        def noise(self, seed, width, height):
            return torch.zeros((1, 1, 1, 1, 2))

        def scheduler(self, steps):
            return SimpleNamespace(timesteps=torch.arange(steps))

        def predict(self, latent, timestep, embedding):
            return latent * 0.125 + embedding

        def step(self, scheduler, velocity, timestep, latent):
            return latent + velocity * 0.1

    runtime = Runtime()
    row = {"id": "fixture", "seed": 1, "neutral": "plain", "positive": "boss",
           "explicit_positive": "specific", "alternate_positive": "other"}
    matte = STUDY / "mattes/bridge.png"
    recipes = ("soft-floor45", "strong-boss-45-mix35", "strong-cues-early3", "strong-cues-late3", "diverse-mean35",
               "sega-early35", "sega-all35")
    captures = [capture_row(runtime, row, matte, recipe)["records"] for recipe in recipes]
    for index, fields in enumerate(zip(*captures, strict=True)):
        control, explicit, first, final, averaged, percentile_early, percentile_all = fields
        step = index % STEPS
        for field in fields:
            assert field["position"] == step and field["trajectory"] == control["trajectory"]
            assert torch.equal(field["latent"], control["latent"])
            assert torch.equal(field["neutral"], control["neutral"])
        assert torch.equal(first["positive"], (explicit if step < 3 else control)["positive"])
        assert torch.equal(final["positive"], (explicit if step >= 7 else control)["positive"])
        assert torch.equal(percentile_early["positive"], (percentile_all if step < 3 else control)["positive"])
        assert not torch.equal(percentile_all["positive"], control["positive"])
        neutral = control["neutral"]
        pure = runtime.predict(control["latent"], step, runtime.encode(row["positive"]))
        control_edit = control["positive"] - neutral
        cue = (runtime.encode("specific") + runtime.encode("other")) * 0.5
        mixed = (pure - neutral).lerp(cue, 0.35)
        # Infer the relative matte weights from the masked baseline contrast.
        expected = mixed * (control_edit / (pure - neutral))
        expected = expected * ((pure - neutral).norm() / expected.norm())
        torch.testing.assert_close(averaged["positive"] - neutral, expected)
        torch.testing.assert_close((percentile_all["positive"] - neutral).norm(), control_edit.norm())
