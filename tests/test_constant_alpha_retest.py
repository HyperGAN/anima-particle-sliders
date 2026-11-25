import pytest

from release_tools.constant_alpha_retest import search_constant, verify_search


@pytest.mark.parametrize("peak", [0.08, 0.68, 1.4375, 2.75, 4.8, 5.0])
def test_search_refines_both_sides_and_reaches_a_rising_native_edge(peak):
    observations = search_constant(lambda a: 1 - (a - peak) ** 2)
    selected = max(observations, key=observations.get)
    assert abs(selected - peak) <= 1 / 32
    assert all(0 < a <= 5 for a in observations)
    assert verify_search(list(observations.items()), selected) == selected
    if selected < 5:
        assert min(a for a in observations if a > selected) - selected <= 1 / 32
    if selected > 1 / 32:
        assert selected - max(a for a in observations if a < selected) <= 1 / 32


def test_signed_effect_can_be_negative_and_a_missing_trial_cannot_win():
    observations = search_constant(
        lambda a: -((a - 1.375) ** 2) - 0.2 if a < 1.5 else None
    )
    selected = max(
        (a for a in observations if observations[a] is not None), key=observations.get
    )
    assert selected == pytest.approx(1.375, abs=1 / 32)
    with pytest.raises(ValueError, match="Reserved alpha"):
        verify_search(list(observations.items()), 1.5)
    with pytest.raises(ValueError, match="Missing"):
        verify_search(list(observations.items())[1:], selected)


def test_two_objectives_keep_damage_and_repetition_specific_to_the_design_board():
    from release_tools.constant_alpha_retest import score_panel

    rows = {
        "a": dict(
            boss_distinction=0.8,
            original_boss_distinction=0.2,
            image_breakage=0.1,
            on_sha256="left",
        ),
        "b": dict(
            boss_distinction=0.5,
            original_boss_distinction=0.7,
            image_breakage=0.2,
            on_sha256="right",
        ),
    }
    pair = dict(
        cases=["a", "b"],
        images=["left", "right"],
        repeat=0.5,
        forward={"value": 0.6},
        reverse={"value": 0.4},
    )
    assert score_panel(rows, "matched_boss_gain") == pytest.approx(0.2)
    assert score_panel(rows, "distinct_boss_score", [pair]) == pytest.approx(
        0.6 * 0.8**3 * 0.9 / 2
    )
    rows["b"]["original_boss_distinction"] = 0.1
    distinct = score_panel(rows, "distinct_boss_score", [pair])
    assert distinct < score_panel(
        rows,
        "distinct_boss_score",
        [dict(pair, repeat=0.0, forward={"value": 0.0}, reverse={"value": 0.0})],
    )
    with pytest.raises(ValueError, match="two images"):
        score_panel(rows, "distinct_boss_score", [dict(pair, images=["right", "left"])])


def test_reservation_uses_search_cases_and_ranking_cannot_switch_metric_or_alpha():
    from copy import deepcopy
    from release_tools.constant_alpha_retest import verify_recipe

    def result(alpha, case):
        intensity = 0.75 - 0.05 * abs(1.4 - alpha)
        return dict(
            alpha=alpha,
            gains=[alpha] * 10,
            score=intensity - 0.2,
            pairs=[],
            rows={
                case: dict(original_boss_distinction=0.2, boss_distinction=intensity)
            },
        )

    points = search_constant(lambda a: result(a, "search")["score"])
    best = max(points, key=points.get)
    rank = result(best, "board")
    study = dict(
        objective="matched_boss_gain",
        alpha=best,
        score=rank["score"],
        search=[result(a, "search") for a in points],
        ranking=rank,
    )
    panels = dict(search=["search"], ranking=["board"])
    assert verify_recipe(study, panels) == best
    changed = deepcopy(study)
    changed["ranking"]["alpha"] = best + 0.1
    changed["ranking"]["gains"] = [best + 0.1] * 10
    with pytest.raises(ValueError, match="reserved"):
        verify_recipe(changed, panels)
    changed = deepcopy(study)
    changed["search"][0]["score"] += 0.1
    with pytest.raises(ValueError, match="objective"):
        verify_recipe(changed, panels)
    with pytest.raises(ValueError, match="overlap"):
        verify_recipe(study, dict(search=["search"], ranking=["search"]))


def test_a_numeric_score_cannot_replace_the_whole_ranking_panel():
    from release_tools.constant_alpha_retest import verify_recipe

    with pytest.raises(ValueError, match="ranking panel"):
        verify_recipe(
            dict(alpha=1.0, score=0.9, ranking=None, search=[]),
            dict(search=["a"], ranking=["b"]),
        )


def test_committed_38_method_snapshot_replays_and_rejects_a_proxy_ranking():
    from copy import deepcopy
    import gzip
    import json
    from hashlib import sha256
    from pathlib import Path
    from release_tools.constant_alpha_retest import verify_snapshot

    archive = Path(__file__).resolve().parents[1] / "docs/final-boss-raw192/vibe12"
    raw = (archive / "constant-alpha-evidence.json.gz").read_bytes()
    manifest = json.loads((archive / "constant-alpha-retest.json").read_bytes())
    assert sha256(raw).hexdigest() == manifest["evidence_sha256"]
    snapshot = json.loads(gzip.decompress(raw))
    assert len(snapshot["methods"]) == 38
    assert len(verify_snapshot(snapshot)) == manifest["completed_fit_ranks"] == 28
    wrong = deepcopy(snapshot)
    wrong["ranking"][0], wrong["ranking"][1] = wrong["ranking"][1], wrong["ranking"][0]
    with pytest.raises(ValueError, match="Published ranks"):
        verify_snapshot(wrong)
