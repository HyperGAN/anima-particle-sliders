"""Verifiers refuse evidence whose parts no longer describe one measured setting."""
import gzip
import json
from pathlib import Path

import pytest

from release_tools import publish_final_boss_vibe12 as board_export
from release_tools.final_boss_damage_search import FIRST_PROTOCOL_CEILING, TRIAL_BUDGET, search, untried_side
from release_tools.final_boss_default import scheduled_gains, select_default, strongest_alpha
from release_tools.final_boss_design_alpha import verify as verify_alpha
from release_tools.verify_final_boss_designs import verify_panel
from release_tools.verify_final_boss_preferences import verify as verify_preferences

STUDY = Path(__file__).resolve().parents[1] / "docs/final-boss-raw192"
VIBE12 = STUDY / "vibe12"


def read(path):
    return json.loads(path.read_text())


def receipts(name):
    with gzip.open(name, "rt") as stream:
        return json.load(stream)


def test_released_gains_follow_from_schedule_alpha_and_sampler_grid():
    default = read(STUDY / "default.json")
    assert scheduled_gains(default["profile"], default["alpha"], default["sigmas"]) == default["gains"]
    panel = read(VIBE12 / "design-range.json")["development"]
    board, catalog = read(VIBE12 / "standings.json"), read(VIBE12 / "checkpoints.json")
    assert select_default(panel, board, catalog) == default
    winner = panel["rows"][0]
    setting = next(s for s in board["settings"]
                   if all(s[k] == winner[k] for k in ("method", "profile", "alpha")))
    setting["gains"] = [1.0] * 10
    with pytest.raises(ValueError, match="schedule and alpha"):
        select_default(panel, board, catalog)


@pytest.mark.parametrize("gains", [[9.0] * 10, [1.0]])
def test_alpha_search_refuses_gains_that_its_schedule_does_not_produce(gains):
    folder = VIBE12 / "alpha-tuning/late-cues"
    evidence = read(folder / "search.json")
    selected = evidence["selected"]
    winning = next(t for t in evidence["trials"] if t["phase"] == "validation"
                   and (t["profile"], t["alpha"]) == (selected["profile"], selected["alpha"]))
    # Changing both copies together used to pass, because they were only compared with each other.
    winning["gains"] = selected["effective_strengths"] = gains
    with pytest.raises(ValueError, match="schedule and alpha"):
        verify_alpha(evidence, receipts(folder / "receipts.json.gz"))


def peaked(best):
    return lambda alpha: 1 - abs(alpha - best) / 16


def replay(score):
    return search(score, FIRST_PROTOCOL_CEILING, climb=False)


def test_search_has_no_ceiling_and_measures_past_its_best_alpha():
    trials, _ = search(peaked(2.5))
    best = max(trials, key=trials.get)
    assert best == 2.1875 and max(trials) > best and untried_side(trials) is None
    # The published protocol stopped at its ceiling with the stronger side unmeasured.
    capped, _ = replay(peaked(2.5))
    assert max(capped, key=capped.get) == max(capped) == 1.75 and untried_side(capped) == "stronger"


def test_search_keeps_climbing_after_the_budget_until_a_stronger_probe_scores_lower():
    trials, steps = search(peaked(7))
    best = max(trials, key=trials.get)
    assert len(trials) > TRIAL_BUDGET and max(trials) > best > 4 and untried_side(trials) is None
    assert steps[-1]["next_anchor"] == best
    # Past the budget only the stronger side is measured.
    assert sorted(trials)[:4] == [.75, .875, 1.0, 1.0625]


def test_search_stops_at_the_strongest_setting_the_product_applies():
    sigmas = read(STUDY / "default.json")["sigmas"]
    limits = {profile: strongest_alpha(profile, sigmas) for profile in ("constant", "early", "late")}
    assert limits["constant"] == 5 and limits["early"] == limits["late"] < 3.34
    for profile, limit in limits.items():
        assert max(scheduled_gains(profile, limit, sigmas)) <= 5 < max(scheduled_gains(profile, limit + 1e-9, sigmas))
        trials, _ = search(lambda alpha: alpha, limit)
        assert max(trials) == limit and untried_side(trials, limit) is None and untried_side(trials) == "stronger"
    with pytest.raises(ValueError, match="unbracketed"):
        search(lambda alpha: alpha / 100)


def test_search_within_the_old_range_is_unchanged_and_reports_an_untried_weaker_side():
    for best in (.9, 1.1, 1.3):
        assert search(peaked(best)) == replay(peaked(best))
    falling, _ = search(lambda alpha: -alpha)
    assert min(falling) == .25 and untried_side(falling) == "weaker"
    assert untried_side({1.0: .5, 2.0: None}) == "stronger" and untried_side({}) is None


def test_published_searches_replay_under_their_recorded_ceiling_only():
    folder = VIBE12 / "alpha-tuning/late-cues"
    evidence, proofs = read(folder / "search.json"), receipts(folder / "receipts.json.gz")
    assert "bounds" not in evidence
    verify_alpha(evidence, proofs)
    sides = {profile: untried_side({trial["alpha"]: trial["score"] for trial in path["trials"]})
             for profile, path in evidence["paths"].items()}
    assert sides == dict(constant="stronger", early=None, late="stronger")
    with pytest.raises(KeyError):
        # Without the ceiling the same scores call for α 2.1875, which was never measured.
        verify_alpha(evidence | dict(bounds=[.25, None]), proofs)
    with pytest.raises(ValueError, match="Unknown alpha search bounds"):
        verify_alpha(evidence | dict(bounds=[.25, 2.5]), proofs)


def test_design_panel_ranks_every_trained_recipe_exactly_once():
    report = read(VIBE12 / "design-range.json")["development"]
    proofs, board = receipts(VIBE12 / "design-receipts.json.gz"), read(VIBE12 / "standings.json")
    verify_panel(report, proofs, board)
    repeated = dict(report, rows=[report["rows"][0], *report["rows"]])
    with pytest.raises(ValueError, match="exactly once"):
        verify_panel(repeated, proofs, board)
    with pytest.raises(ValueError, match="exactly once"):
        verify_panel(dict(report, rows=report["rows"][1:]), proofs, board)


def test_published_preference_order_requires_its_margins():
    report = read(VIBE12 / "preferences.json")["finalists"]
    proofs = receipts(VIBE12 / "preference-receipts.json.gz")
    verify_preferences(report, proofs)
    assert "order" in report
    del report["margins"]
    report["order"] = report["order"][::-1]
    with pytest.raises(ValueError, match="requires its margins"):
        verify_preferences(report, proofs)


@pytest.mark.parametrize("result", [dict(step=144, ema_decay=None), dict(step=192, ema_decay=.999), dict(step=192)])
def test_board_refuses_unfinished_or_averaged_recipes(tmp_path, result):
    cohort = tmp_path / "cohorts/recipe"
    cohort.mkdir(parents=True)
    (cohort / "result.json").write_text(json.dumps(result | dict(raw={"sha256": "0" * 64}, cache="c")))
    with pytest.raises(ValueError, match="RAW192 result without weight EMA"):
        board_export.raw_result(tmp_path, "recipe")
    (cohort / "result.json").write_text(json.dumps(dict(step=192, ema_decay=None)))
    assert board_export.raw_result(tmp_path, "recipe")["step"] == 192


def test_interrupted_sample_strip_is_never_mistaken_for_a_finished_one(tmp_path, monkeypatch):
    from hashlib import sha256
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), "red").save(source)
    record = dict(path=str(source), sha256=sha256(source.read_bytes()).hexdigest())
    save = Image.Image.save

    def interrupted(image, target, *args, **kwargs):
        if str(target).endswith(".partial"):
            Path(target).write_bytes(b"RIFF")
            raise KeyboardInterrupt
        return save(image, target, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", interrupted)
    with pytest.raises(KeyboardInterrupt):
        board_export.write_sheet([record], tmp_path / "board")
    monkeypatch.setattr(Image.Image, "save", save)
    name = board_export.write_sheet([record], tmp_path / "board")
    with Image.open(tmp_path / "board" / name) as strip:
        assert strip.size == (144, 144)


def test_replay_refuses_a_changed_checkpoint_or_setting_without_a_model(tmp_path):
    from release_tools.final_boss_replay import check_release

    default = read(STUDY / "default.json")
    weights = tmp_path / "RAW000192.safetensors"
    weights.write_bytes(b"not the published weights")
    with pytest.raises(ValueError, match="RAW checkpoint differs"):
        check_release(default, weights)
    from hashlib import sha256
    local = default | dict(checkpoints={default["recipe"]: dict(
        bytes=weights.stat().st_size, sha256=sha256(weights.read_bytes()).hexdigest())})
    check_release(local, weights)
    for change in (dict(alpha=1.25), dict(profile="constant"), dict(gains=default["gains"][:-1]),
                   dict(sigmas=default["sigmas"][:-1])):
        with pytest.raises(ValueError):
            check_release(local | change, weights)


def test_unmeasured_sides_at_native_strength_boundaries():
    assert untried_side({.25: 1, .5: 0}, strongest=5) == "weaker"
    assert untried_side({1: None, 5: 1}, strongest=5) == "weaker"
    assert untried_side({1: 0, 5: 1}, strongest=5) is None


@pytest.mark.parametrize("sigmas", [[1, 0], [1] * 10 + [0], [1, .9, .8, .7, .6, float("nan"), .4, .3, .2, .1, 0]])
def test_constant_gains_require_the_original_ten_interval_grid(sigmas):
    with pytest.raises(ValueError, match="native Euler intervals"):
        scheduled_gains("constant", 1, sigmas)
