"""Verifiers refuse evidence whose parts no longer describe one measured setting."""
import gzip
import json
from pathlib import Path

import pytest

from release_tools import publish_final_boss_vibe12 as board_export
from release_tools.final_boss_damage_search import bound_limited, search
from release_tools.final_boss_default import scheduled_gains, select_default
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


def test_search_reports_when_its_best_probe_is_a_bound():
    evidence = read(VIBE12 / "alpha-tuning/late-cues/search.json")
    limited = {profile: bound_limited({trial["alpha"]: trial["score"] for trial in path["trials"]})
               for profile, path in evidence["paths"].items()}
    assert limited == dict(constant=True, early=False, late=True)
    rising, _ = search(lambda alpha: alpha)
    assert max(rising) == 1.75 and bound_limited(rising)
    assert not bound_limited({1.0: .5, 1.75: None}) and not bound_limited({})


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
