import json
from pathlib import Path

import pytest


@pytest.fixture
def writer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "research/vibe-regime"))
    import prompt_writer
    monkeypatch.setattr(prompt_writer.time, "sleep", lambda _: None)
    return prompt_writer


def completion(settings, inputs):
    rows = [
        dict(id=a["id"], neutral=f"an empty vessel at {a['id']}",
             treatments=[dict(positive=f"{a['id']} concept {n}", explicit=f"{a['id']} stronger {n}")
                         for n in range(3)])
        for a in reversed(inputs["scenes"])
    ]
    return "served-model", dict(scenes=rows)


def test_writes_all_prompts_and_continues_from_last_good_batch(writer, tmp_path, monkeypatch):
    phrase = "an impossible tide"
    calls = []

    def interrupted(settings, inputs):
        assert settings["target"] == "Anima Turbo v1.1"
        assert settings["model"] == "first-model"
        assert inputs["idea"] == phrase
        assert all(set(anchor) == {"id", "framing"} for anchor in inputs["scenes"])
        calls.append(inputs["scenes"][0]["id"])
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return completion(settings, inputs)

    monkeypatch.setenv("ANIMA_PROMPT_MODEL", "first-model")
    with pytest.raises(RuntimeError, match="interrupted"):
        writer.load_spec(tmp_path, phrase, ask=interrupted)
    assert not (tmp_path / "spec.json").exists()
    progress = json.loads((tmp_path / "prompt-rewrites/progress.json").read_bytes())
    assert len(progress["teaching"]) == writer.BATCH_SIZE
    monkeypatch.setenv("ANIMA_PROMPT_MODEL", "changed-between-runs")
    def resume(settings, inputs):
        assert settings["model"] == "first-model"
        assert inputs["scenes"][0]["id"] != calls[0]
        return completion(settings, inputs)

    grid = writer.load_spec(tmp_path, phrase, ask=resume)
    assert grid["prompt_writer"]["served_models"] == ["served-model"]
    assert len(grid["teaching"]) == 24 and len(grid["confirmation"]) == 16
    for s in grid["teaching"]:
        assert s["neutral"] == f"an empty vessel at {s['id']}"
        assert [t["positive"] for t in s["treatments"]] == [f"{s['id']} concept {n}" for n in range(3)]
    copied = tmp_path / "continued-job"
    copied.mkdir()
    (copied / "spec.json").write_bytes((tmp_path / "spec.json").read_bytes())
    assert writer.load_spec(copied, phrase, ask=lambda *_: pytest.fail("rewrote saved text")) == grid
    with pytest.raises(ValueError, match="phrase changed"):
        writer.load_spec(copied, "different", ask=completion)
    grid["teaching"][0]["neutral"] = "unrelated edit"
    (copied / "spec.json").write_text(json.dumps(grid))
    with pytest.raises(ValueError, match="specification"):
        writer.load_spec(copied, phrase, ask=completion)


def test_bad_rewrites_retry_and_never_fall_back_to_image_templates(writer, tmp_path):
    calls = []
    def malformed(settings, inputs):
        calls.append(inputs)
        model, reply = completion(settings, inputs)
        edits = reply["scenes"][0]["treatments"]
        edits[1]["positive"] = edits[0]["positive"]
        return model, reply
    with pytest.raises(RuntimeError, match="repeated a render prompt"):
        writer.load_spec(tmp_path, "shattered sky", ask=malformed)
    assert len(calls) == 3 and not (tmp_path / "spec.json").exists()


@pytest.mark.parametrize("version", ["wardrobe-v0", "literal-v1", "literal-scene-v2"])
def test_saved_template_runs_never_contact_writer(writer, tmp_path, version):
    old = writer.library.spec("old phrase", version=version)
    (tmp_path / "spec.json").write_text(json.dumps(old))
    assert writer.load_spec(tmp_path, "old phrase", ask=lambda *_: pytest.fail("unexpected call")) == old
