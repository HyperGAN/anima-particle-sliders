import json

import pytest

from release_tools.final_boss_damage_search import best_setting, main, panel_scores, rank_snapshot, search
from release_tools.final_boss_scoring import BOSS, DAMAGE, GAIN, ORIGINAL, SCORE, score_case


def setting(method, alpha, cases, profile="early"):
    return {"method": method, "alpha": alpha, "profile": profile, "cases": cases}


def test_search_uses_both_sides_and_the_ninth_probe_can_win():
    calls = []

    def evaluate(alpha):
        calls.append(alpha)
        return 1 - abs(alpha - .65625)

    trials, steps = search(evaluate)
    assert calls == list(trials) == [1., 1.25, .75, 1.125, .375, .9375, .5625, .84375, .65625]
    assert max(trials, key=trials.get) == calls[-1]
    assert steps[0]["next_step"] == .375
    assert steps[1]["next_step"] == .1875


def test_search_ties_hold_the_anchor_and_unmeasured_trials_stay_missing():
    trials, steps = search(lambda _: 0.)
    assert list(trials)[-2:] == [1.03125, .96875]
    assert all(step["next_anchor"] == 1. for step in steps)
    missing, _ = search(lambda _: None)
    assert len(missing) == 9 and set(missing.values()) == {None}
    with pytest.raises(ValueError, match="finite"):
        search(lambda _: float("nan"))


def test_damage_is_the_only_quality_penalty_and_panel_weights_cases_equally():
    unchanged = {BOSS: .3, ORIGINAL: .3, DAMAGE: 0.}
    boss = {BOSS: .85, ORIGINAL: .1, DAMAGE: .2}
    cases = {"a": unchanged, "b": boss}
    expected = panel_scores(cases, ["a", "b"])
    assert expected[SCORE] == pytest.approx(score_case(boss) / 2)
    assert expected[GAIN] == pytest.approx(.375)
    assert expected[DAMAGE] == pytest.approx(.1)
    for verdict in ("pass", "unmeasured", "fail"):
        flagged = {name: case | {"integrity": verdict, SCORE: 0} for name, case in cases.items()}
        assert panel_scores(flagged, ["b", "a"]) == expected
    strong = setting("method", 1.25, {"a": boss, "b": boss}) | {"status": "fail"}
    damaged = setting("method", 1.5, {"a": boss | {DAMAGE: .8}, "b": boss | {DAMAGE: .8}})
    assert best_setting([damaged, strong], ["a", "b"])["alpha"] == 1.25


def test_missing_case_rejected_and_missing_rating_never_treated_as_zero():
    row = {BOSS: .4, ORIGINAL: .1, DAMAGE: None}
    for cases, names in (({"a": row}, ["a", "b"]), ({"a": row}, ["a", "a"]), ({}, [])):
        with pytest.raises(ValueError, match="same complete"):
            panel_scores(cases, names)
    assert best_setting([setting("method", .5, {"a": row})], ["a"]) is None
    assert panel_scores({"a": row | {DAMAGE: 0, ORIGINAL: .4}}, ["a"])[SCORE] == 0.


def test_offline_cli_reproduces_deterministic_leaderboard_ties(tmp_path, capsys):
    cases = {"a": {BOSS: .75, ORIGINAL: .25, DAMAGE: .1}}
    first = setting("z", 1.25, cases)
    second = setting("z", .75, cases, "constant")
    assert best_setting([first, second], ["a"])["alpha"] == 1.25
    snapshot = {"cases": ["a"], "settings": [first, second, setting("a", 1, cases)]}
    assert [(r["method"], r["alpha"]) for r in rank_snapshot(snapshot)] == [("a", 1), ("z", .75)]
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(snapshot))
    main([str(path)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split("\t") == ["1", "a", "early", "1", "17.09", "50.00", "10.00"]
    assert lines[2].startswith("2\tz\tconstant\t0.75\t")
