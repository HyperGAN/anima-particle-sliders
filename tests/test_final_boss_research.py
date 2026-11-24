import json
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

import pytest
import torch

from release_tools.final_boss_apg import project_subject_target
from release_tools.final_boss_cues import ARMOR_CREST_START, ROLE_AFTER_COSTUME, ROLE_BEFORE_COSTUME
from release_tools.final_boss_targets import (
    FOREGROUND_APG, GROUNDED_COSTUME, GROUNDED_WAVELET, GUIDED_TEACHER, RECIPES, STEPS, capture_row,
    edited_target, final_target, subject_matte, teacher_rows, verify_guided_trajectory,
)
from release_tools.final_boss_wavelet import low_band


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


def test_regional_attire_preserves_the_stronger_cue_exterior_at_matched_dose():
    def field(values):
        return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 1, 3)

    runtime = SimpleNamespace(predict=lambda latent, timestep, embedding: embedding)
    neutral = field((0.1, 0.25, 0.3))
    positive = field((1., 2., 4.))
    stronger = field((4., 2., 4.))
    ordinary = field((0.2, 0.9, 0.12))
    matte = field((1., 0.8, 0.))
    weight = 0.45 + 0.55 * matte
    baseline = final_target(runtime, None, 0, neutral, positive, weight, 0.35, (stronger,))
    matched = final_target(runtime, None, 0, neutral, positive, weight, 0.35,
                           (stronger, ordinary), matched_attire=True)
    localized = final_target(runtime, None, 0, neutral, positive, weight, 0.35,
                             (stronger, ordinary), regional=(matte, 0), matched_attire=True)

    original = (neutral + (positive - neutral)) - neutral
    expected = original.lerp(stronger - ordinary, 0.35) * weight
    torch.testing.assert_close(matched - neutral, expected * (original.norm() / expected.norm()))
    assert not torch.equal(matched[matte == 0], baseline[matte == 0])
    assert torch.equal(localized[matte == 0], baseline[matte == 0])
    torch.testing.assert_close((localized - neutral).norm(), (baseline - neutral).norm())
    torch.testing.assert_close((localized - baseline).norm(), (matched - baseline).norm())


def test_wavelet_exterior_keeps_the_stronger_foreground_and_reduces_coarse_exterior_energy():
    runtime = SimpleNamespace(predict=lambda latent, timestep, embedding: embedding)
    generator = torch.Generator().manual_seed(21)
    shape = (1, 1, 1, 64, 64)
    neutral = torch.randn(shape, generator=generator)
    positive = neutral + torch.randn(shape, generator=generator)
    explicit = positive + torch.linspace(-1, 1, 64).expand(shape)
    matte = torch.zeros_like(neutral)
    matte[..., 16:48, 16:48] = 1
    weight = 0.45 + 0.55 * matte
    foreground = matte > 0.01

    baseline = final_target(runtime, None, 0, neutral, positive, weight, 0., ())
    stronger = final_target(runtime, None, 0, neutral, positive, weight, 0.35, (explicit,))
    wavelet = final_target(runtime, None, 0, neutral, positive, weight, 0.35, (explicit,), wavelet_matte=matte)

    assert torch.equal(wavelet[foreground], stronger[foreground])
    torch.testing.assert_close((wavelet - neutral).norm(), (stronger - neutral).norm())
    filtered = torch.where(foreground, 0., (wavelet - baseline).double())
    source = torch.where(foreground, 0., (stronger - baseline).double())
    assert low_band(filtered, 3).norm() < low_band(source, 3).norm()


def test_foreground_apg_removes_the_clean_latent_component_at_fixed_dose():
    neutral = torch.tensor([[[[[0.25, 0.5, 0.625]]]]])
    positive = torch.tensor([[[[[2.0, 1.75, 1.25]]]]])
    sigma = 0.75
    latent = sigma * positive + torch.tensor([[[[[1.0, 0., 7.0]]]]])
    matte = torch.tensor([[[[[1.0, 0.7, 0.]]]]])
    weight = 0.45 + 0.55 * matte
    assert RECIPES[FOREGROUND_APG] == RECIPES["soft-floor45"]
    baseline = final_target(None, latent, 0, neutral, positive, weight, 0., ())
    target = final_target(None, latent, 0, neutral, positive, weight, 0., (), subject_projection=(matte, sigma))
    assert target[..., 0].item() == neutral[..., 0].item()
    assert target[..., 1].item() > baseline[..., 1].item()
    assert torch.equal(target[..., -1], baseline[..., -1])
    torch.testing.assert_close((target - neutral).norm(), (baseline - neutral).norm())
    with pytest.raises(ValueError, match="Degenerate conditional clean-latent axis"):
        project_subject_target(sigma * positive, sigma, neutral, positive, baseline, matte)


def test_split_cues_keep_the_original_teacher_text_and_each_recorded_insertion():
    fixture = STUDY / "training-pairs.json"
    arms = [teacher_rows(fixture, name) for name in ("strong-boss-45-mix35", "role-cue35", "crest-cue35")]
    for anchor, role_row, crest_row in zip(*arms, strict=True):
        role, crest = role_row["explicit_positive"], crest_row["explicit_positive"]
        assert all(role_row[key] == crest_row[key] == anchor[key] for key in ("neutral", "positive", "matte", "seed"))
        subject, background = anchor["positive"].split(" Background: ", 1)
        introduced_crest = crest.split(" Background: ", 1)[0][len(subject) + 1:]
        assert introduced_crest.startswith(ARMOR_CREST_START)
        assert crest == f"{subject} {introduced_crest} Background: {background}"
        assert role.replace(ROLE_BEFORE_COSTUME + " ", "").replace(ROLE_AFTER_COSTUME + " ", "") == anchor["positive"]
        assert role.replace(ROLE_AFTER_COSTUME, f"{introduced_crest} {ROLE_AFTER_COSTUME}") == anchor["explicit_positive"]


def test_guided_teacher_changes_only_the_positive_path_and_preserves_full_targets():
    fixture = STUDY / "training-pairs.json"
    row = teacher_rows(fixture, GUIDED_TEACHER)[0]
    rollout = row["guided_rollout"]
    results = json.loads((STUDY / "results.json").read_text())
    reference = results["methods"]["soft-floor45"]["fixed_strength"]
    assert (rollout["profile"], rollout["alpha"], rollout["gains"]) == (
        reference["profile"], reference["alpha"], reference["gains"])
    assert rollout["sigmas"] == results["sigmas"]

    class Runtime:
        device = "cpu"

        def encode(self, prompt):
            return torch.tensor(float(prompt == row["positive"]))

        def noise(self, seed, width, height):
            return torch.full((1, 1, 1, 4, 4), seed % 7 / 5.)

        def scheduler(self, steps):
            return SimpleNamespace(sigmas=torch.tensor(rollout["sigmas"]), timesteps=torch.arange(steps))

        def predict(self, latent, timestep, embedding):
            return latent * 0.125 + embedding * (torch.ones_like(latent) + latent * 0.06)

        def step(self, scheduler, velocity, timestep, latent):
            return latent - velocity * (scheduler.sigmas[timestep] - scheduler.sigmas[timestep + 1])

    runtime = Runtime()
    matte = STUDY / row["matte"]["file"]
    baseline = capture_row(runtime, row, matte, "soft-floor45")
    guided = capture_row(runtime, row, matte, GUIDED_TEACHER)
    assert baseline["row"] == guided["row"] and baseline["seed"] == guided["seed"]
    for original, new in zip(baseline["records"], guided["records"], strict=True):
        position = new["position"]
        assert new["trajectory"] == original["trajectory"]
        assert position == original["position"] and new["timestep"] == original["timestep"]
        if new["trajectory"] == "neutral" or position == 0:
            assert all(torch.equal(new[key], original[key]) for key in ("latent", "neutral", "positive"))
        else:
            assert not torch.equal(new["latent"], original["latent"])
            positive = runtime.predict(new["latent"], position, 1.)
            weight = 0.45 + 0.55 * subject_matte(matte, new["latent"].shape[-2:], runtime.device)
            contrast = positive - new["neutral"]
            masked = contrast * weight
            assert torch.equal(new["neutral"], runtime.predict(new["latent"], position, 0.))
            torch.testing.assert_close(new["positive"], new["neutral"] + masked * (contrast.norm() / masked.norm()))
    records = [record | {"row": guided["row"]} for record in guided["records"]]
    assert verify_guided_trajectory(runtime, records, 0, rollout, row)
    records[-1]["latent"] = baseline["records"][-1]["latent"]
    assert not verify_guided_trajectory(runtime, records, 0, rollout, row)
    with pytest.raises(ValueError, match="Native sigmas differ"):
        schedule = runtime.scheduler(STEPS)
        schedule.sigmas[1] -= 0.01
        invalid_runtime = SimpleNamespace(scheduler=lambda steps: schedule)
        verify_guided_trajectory(invalid_runtime, records, 0, rollout, row)


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
    schedules = {recipe: teacher_rows(STUDY / "training-pairs.json", recipe)[0]["grounded_schedule"]
                 for recipe in GROUNDED_COSTUME}
    assert teacher_rows(STUDY / "training-pairs.json", GROUNDED_WAVELET) == teacher_rows(
        STUDY / "training-pairs.json", "scene-grounded35")
    recipes = ("soft-floor45", "strong-boss-45-mix35", "strong-cues-early3", "strong-cues-late3", "diverse-mean35",
               "sega-early35", "sega-all35", *GROUNDED_COSTUME)
    rows = [row | {"grounded_positive": "other", "grounded_schedule": schedules[recipe]}
            if recipe in GROUNDED_COSTUME else row for recipe in recipes]
    captures = [capture_row(runtime, source, matte, recipe)["records"]
                for source, recipe in zip(rows, recipes, strict=True)]
    for index, fields in enumerate(zip(*captures, strict=True)):
        control, explicit, first, final, averaged, percentile_early, percentile_all, *grounded = fields
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
        for recipe, captured in zip(GROUNDED_COSTUME, grounded, strict=True):
            weight = control_edit / (pure - neutral)
            cue = runtime.predict(control["latent"], step, runtime.encode("other"))
            expected = edited_target(neutral, pure, weight, schedules[recipe][step], cue)
            torch.testing.assert_close(captured["positive"], expected)
