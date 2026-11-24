import json
from itertools import combinations
from pathlib import Path

import pytest

from lumen_studio import vibe_cli, vibe_curation
from lumen_studio.vibe import validate_manifest
from lumen_studio.vibe_observer import decision_values
from lumen_studio.vibe_portfolio import distinct_finish, exclude_duplicates, select
from lumen_studio.vibe_profiles import ANIME, FINAL_BOSS, profile_for


def test_distinct_finish_uses_best_compatible_set_and_all_slots():
    rows = [dict(case=case, finish_gain=gain) for case, gain in zip("abcd", (.8, .7, .65, -.1))]
    proof = {
        "pairs": [dict(left=a, right=b, mean=.65 if a == "a" and b in "bc" else .2)
                  for a, b in combinations("abcd", 2)],
    }
    assert distinct_finish(rows, proof, panel_size=5) == dict(gain=.25, credited=["b", "c"], panel_size=5)
    with pytest.raises(ValueError, match="complete pair audit"):
        distinct_finish(rows, {"pairs": proof["pairs"][:-1]})


def test_pair_repair_respects_family_limit_and_never_fills_with_unknown():
    def row(case, method, family, usable):
        return dict(case=case, method=method, observed_family=family, integrity="pass", usable=usable)
    groups = [
        [row("a", "same", "x", .96), row("a", "other", "z", .1)],
        [row("b", "same", "x", .9), row("b", "other", "y", .8)],
        [row("c", "unknown", None, 1)],
    ]

    def audit(rows):
        return {"pairs": [dict(left=a["case"], right=b["case"],
                               mean=.8 if a["method"] == b["method"] else .1)
                          for a, b in combinations(rows, 2)]}

    chosen = select(groups, 2, allow_partial=True)
    selected, proof, repairs = exclude_duplicates(groups, chosen, audit, limit=2, allow_partial=True)
    assert [(r["case"], r["method"]) for r in selected] == [("a", "same"), ("b", "other")]
    assert proof["pairs"][0]["mean"] == .1 and len(repairs) == 1


@pytest.mark.parametrize("concept,profile", [("1990s anime", ANIME), ("final boss", FINAL_BOSS)])
def test_named_vibe_cli_saves_audited_teachers_and_resumes(tmp_path, monkeypatch, concept, profile):
    pytest.importorskip("particle_sliders_vibe")
    pytest.importorskip("nanoodle")
    assert profile_for(concept) is profile
    monkeypatch.setenv("NANOGPT_API_KEY", "local-fixture-only")
    actions = {"render": 0, "judge": 0, "propose": 0, "pair": 0}

    class Native:
        identity = {"model": "fixture-native", "product": "anima-particle-sliders"}

        def __init__(self, *args, **kwargs):
            pass

        def propose(self, vibe, count, history, **kwargs):
            actions["propose"] += 1
            assert profile.scene_instruction in vibe
            scenes = [f"A distinct in-world scene {len(history)}-{actions['propose']}-{i}" for i in range(count)]
            return [dict(neutral=scene, positive=scene + ". Unusual finished treatment.") for scene in scenes]

        def render(self, prompt, seed, directory, *, request_id):
            actions["render"] += 1
            directory.mkdir(parents=True)
            path = directory / "image.png"
            path.write_text(prompt)
            return dict(path=str(path), prompt=prompt, seed=seed, request_id=request_id)

        def close(self):
            pass

    def judge(provider, model, images, state, questions):
        actions["judge"] += 1
        prompt = Path(images[0]).read_text().casefold()
        assert provider == "nanogpt" and model == "fixture"
        if len(images) == 2:
            assert questions["near"]["instructions"] == profile.lookalike
            actions["pair"] += 1
        answers = {}
        for name, question in questions.items():
            if question["type"] == "score":
                score = 3 if name in {"effect", "usable"} and "continuous shot" in prompt else 0
                answers[name] = dict(type="score", score=score, probabilities={str(i): int(i == score) for i in range(4)})
            elif question["type"] == "choice":
                tag = next((key for key, cue in profile.families.items() if cue.casefold() in prompt), "unclassified")
                answers[name] = dict(type="choice", choice=tag,
                                     probabilities={key: int(key == tag) for key in question["criteria"]})
            else:
                family = name.removeprefix("family_")
                present = family in profile.families and profile.families[family].casefold() in prompt
                answers[name] = dict(type="noul", noul=.96 if present else .01)
        raw = {"answers": answers}
        return dict(values=decision_values(raw, questions), raw=raw)

    monkeypatch.setattr(vibe_cli, "AnimaVibe", Native)
    monkeypatch.setattr(vibe_curation, "observe", judge)
    output = tmp_path / "run"
    cli = ["--vibe", concept, "--out", str(output), "--model-dir", str(tmp_path), "--prompt-model", "fixture",
           "--judge-model", "fixture", "--pool-size", "3", "--replace", "1", "--best-of", "1"]
    assert vibe_cli.main(cli) == 0
    state = json.loads((output / "portfolio.json").read_text())
    chosen = state["rounds"][0]
    assert len(chosen["selected"]) == 3 and len(chosen["pair_audit"]["pairs"]) == 3 and actions["pair"] == 6
    if profile is ANIME:
        assert {row["prompt"]["kind"] for row in chosen["ranking"]} == {"environment", "object", "animal"}
    assert chosen["distinct_finish"]["gain"] == 1 and chosen["pair_audit"]["near_duplicates"] == 0
    assert sorted(len([r for r in chosen["selected"] if r["observed_family"] == look])
                  for look in {r["observed_family"] for r in chosen["selected"]}) == [1, 2]
    manifests = list((output / "training").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    validate_manifest(manifest)
    assert len(manifest["rows"]) == 3
    snapshot = actions.copy()
    assert vibe_cli.main(cli) == 0 and actions == snapshot
    assert vibe_cli.main(cli + ["--rounds", "2"]) == 0
    next_state = json.loads((output / "portfolio.json").read_text())
    assert len(next_state["rounds"]) == 2 and next_state["rounds"][0] == chosen
    assert len(list((output / "training").glob("*/manifest.json"))) == 2


def test_legacy_output_uses_original_entrypoint(tmp_path, monkeypatch):
    (tmp_path / "collection.json").write_text("{}")
    argv = ["--vibe", "final boss", "--out", str(tmp_path)]
    monkeypatch.setattr(vibe_cli, "free_text_main", lambda forwarded: forwarded)
    assert vibe_cli.main(argv) is argv
