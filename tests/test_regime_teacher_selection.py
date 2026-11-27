from pathlib import Path

import pytest


@pytest.fixture
def regime(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "research/vibe-regime"))
    import teacher_selection
    return teacher_selection


def row(scene, seed, scores, chosen=None):
    def image(name):
        return dict(path=f"/images/{name}.png", sha256=name)

    options = [dict(positive=f"edit {i}", explicit=f"cue {i}", image=image(f"{seed}-{i}"), boss=score, case=.75)
               for i, score in enumerate(scores)]
    return dict(scene=scene, seed=seed, framing="portrait", neutral=image(str(seed)),
                options=options, chosen=chosen)


def test_rank_all_measured_rounds_before_cutoff_and_rebuild_training_pairs(regime):
    rows = [row("first", 30, [.80, .71]), row("weak", 40, [.51]), row("strong", 50, [.90, .82]),
            row("first", 31, [.91, .95], chosen="edit 0")]
    rows[-1]["options"][1].update(baseline_structure="unmeasured", candidate_structure="fail", separate=None, case=None)
    scenes = {r["scene"]: {"neutral": r["scene"]} for r in rows}
    def select():
        return regime.select_teachers(rows, scenes, lambda neutral: neutral["path"] + ".mask", limit=2)

    kept = select()
    assert kept == select()
    assert list(kept) == ["first", "strong"]
    assert [r["chosen"] for r in rows] == [None, None, "edit 0", "edit 1"]
    assert kept["first"] == dict(neutral="first", positive="edit 1", explicit="cue 1", seed=31,
                                 framing="portrait", boss=.95, case=None, matte="/images/31.png.mask",
                                 neutral_image=rows[-1]["neutral"], positive_image=rows[-1]["options"][1]["image"])


def test_mask_fallback_floor_missing_scores_and_stable_ties(regime):
    rows = [row("station", 10, [.99, .97]), row("station", 11, [.8, .8]),
            row("hall", 12, [float("nan"), None, .34]), row("park", 13, [.35]), row("cafe", 14, [.35])]
    checked = []
    def matte(neutral):
        checked.append(neutral["sha256"])
        return None if neutral["sha256"] == "10" else neutral["path"]

    scenes = {r["scene"]: {"neutral": r["scene"]} for r in rows}
    kept = regime.select_teachers(rows, scenes, matte, limit=2)
    assert list(kept) == ["station", "park"]
    assert rows[0]["mask_usable"] is False
    assert rows[1]["chosen"] == "edit 0"
    assert checked == ["10", "11", "13"]


def test_literal_attribute_and_role_share_the_grid_and_keep_legacy_prompts(regime):
    import library
    import questions

    for phrase in ["forked tail", "pirate captain", "glass marbles everywhere"]:
        current = library.spec(phrase)
        legacy = library.spec(phrase, version=library.WARDROBE_VERSION)
        assert "phrase_version" not in legacy and current["phrase_version"] == library.PHRASE_VERSION
        assert current["confirmation"] == legacy["confirmation"] and current["watched"] == legacy["watched"]
        for new, old in zip(current["teaching"], legacy["teaching"], strict=True):
            assert new["neutral"] == old["neutral"] and new["seed"] == old["seed"]
            assert new["treatments"][0]["positive"].startswith(phrase + ". ")
            assert new["treatments"][1]["positive"] == f"{new['neutral']} {phrase}."
            assert len({t["positive"] for t in new["treatments"]}) == 3
            for treatment in new["treatments"]:
                assert all(phrase in text and "costume" not in text for text in treatment.values())
            assert "hat, coat and accessories" in old["treatments"][0]["positive"]
        for rubric in (questions.build(phrase)["effect"]["questions"]["effect"], questions.build(phrase)["rubric"]):
            assert phrase in rubric["instructions"] and "whole image" in rubric["instructions"]
        assert "signature costume details" in questions.build(phrase, version=library.WARDROBE_VERSION)["effect"]["questions"]["effect"]["instructions"]
        prior = library.spec(phrase, version=library.CHARACTER_VERSION)
        assert prior["phrase_version"] == "literal-v1"
        assert f"visibly embodying {phrase}" in prior["teaching"][0]["treatments"][0]["positive"]
        assert "main character" in questions.build(phrase, version=library.CHARACTER_VERSION)["effect"]["state"]
    all_over = library.spec("glass marbles everywhere")["teaching"][0]["treatments"][2]
    assert "In the foreground: glass marbles." in all_over["positive"]
    assert "In the background: glass marbles." in all_over["positive"]
    localized = library.spec("two horns on the forehead")["teaching"][0]
    assert all("On every surface" not in t["positive"] for t in localized["treatments"])
