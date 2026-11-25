"""Regression cases for unattended qualification, replacement and checkpoint selection."""

from types import SimpleNamespace
import json
import re

import pytest

from lumen_studio.contracts import atomic_json
from lumen_studio.study.outputs import Unmeasured, read
from lumen_studio.study.teachers import collect
from lumen_studio.study.training import rate_candidates, select_checkpoint, arm_path
from lumen_studio.vibe_integrity import FrameIntegrity


class RenderFixture:
    def __init__(self, root):
        self.root = root
        self.runtime = SimpleNamespace(identity={"model": "fixture"})
        self.rubric = {}
        self.calls = []

    def image(self, prompt, seed):
        self.calls.append((prompt, seed))
        return (prompt, seed)

    def score(self, record):
        prompt, seed = record
        score = 0.9 if prompt.startswith("treatment") else 0.1
        status = (
            "fail"
            if prompt == "neutral adult" and seed == 11
            else "unmeasured"
            if prompt == "treatment unresolved"
            else "pass"
        )
        return {
            "output": {"path": "test", "sha256": "0" * 64},
            "judgment": {"score": score},
            "integrity": {"status": status},
        }


def test_broken_neutral_and_unknown_positive_cannot_enter_cache(tmp_path):
    outputs = RenderFixture(tmp_path)
    config = {
        "neutral_ceiling": 0.35,
        "teachers": [
            {
                "neutral": "neutral adult",
                "treatments": ["treatment unresolved", "treatment clear"],
                "seed": 11,
            }
        ],
    }
    result = collect(outputs, config, "demo", 1, 2)
    (row,) = result["rows"]
    assert row["positive"] == "treatment clear"
    assert row["seeds"] == [104740]
    assert ("treatment clear", 11) not in outputs.calls
    attempts = read(tmp_path / "teacher-decisions.json")
    assert len(attempts) == 2 and not attempts[0]["treatments"]
    assert len([r for r in attempts[1]["treatments"] if r["eligible"]]) == 1


def test_exhaustion_produces_an_explicit_incomplete_collection(tmp_path):
    config = {
        "neutral_ceiling": 0.35,
        "teachers": [
            {
                "neutral": "neutral adult",
                "treatments": ["treatment unresolved"],
                "seed": 11,
            }
        ],
    }
    with pytest.raises(Unmeasured, match="0 of 1"):
        collect(RenderFixture(tmp_path), config, "demo", 1, 2)
    assert (tmp_path / "teacher-decisions.json").is_file()
    assert not (tmp_path / "manifest.json").exists()


def test_screen_changes_exactly_one_rate_and_selects_by_finished_effect(tmp_path):
    rates = {"g": 0.001, "d": 0.01, "particle": 0.03}
    arms = list(rate_candidates(rates))
    assert arms[0] == rates and len(arms) == 7
    for candidate in arms[1:]:
        assert sum(candidate[k] != rates[k] for k in rates) == 1
    for step, score in zip((16, 32, 48, 64), (0.1, 0.3, 0.2, 0.0)):
        atomic_json(
            arm_path(tmp_path, rates) / "fit" / f"{step}.json",
            dict(step=step, status="pass", mean_gain=score),
        )
    assert select_checkpoint(tmp_path, rates, 64)["step"] == 32
    atomic_json(
        arm_path(tmp_path, rates) / "fit" / "32.json",
        dict(step=32, status="unmeasured", mean_gain=0.3),
    )
    assert select_checkpoint(tmp_path, rates, 64)["step"] == 48


def test_calibration_requires_the_real_failure_examples(tmp_path, monkeypatch):
    detector = FrameIntegrity(tmp_path)
    monkeypatch.setattr(detector, "check", lambda _: {"status": "pass", "checks": {}})
    result = detector.calibrate()
    assert not result["passed"] and result["cases"]["broken"]["expected"] == "fail"
    assert result["cases"]["split"]["expected"] == "fail"


def test_adult_subjects_appear_only_in_generation_prompts():
    from lumen_studio.vibe_integrity import DEFECTS, STRUCTURE_DETAILS
    from lumen_studio.study.cli import arguments

    spec = json.loads(
        arguments(
            ["--vibe", "Final Boss", "--out", ".", "--model-dir", "."]
        ).spec.read_text()
    )
    age_question = re.compile(r"\b(adult|age|minor|child|underage|teen)\b", re.I)
    subject = re.compile(r"\badults?\b|\b\d+-year-old\b", re.I)
    for concept in spec["concepts"].values():
        assert not age_question.search(json.dumps(concept["effect"]))
        for row in concept["teachers"]:
            assert all(
                subject.search(prompt)
                for prompt in (row["neutral"], *row["treatments"])
            )
    assert all(
        subject.search(row["prompt"])
        for panel in spec["validation"].values()
        for row in panel
    )
    assert not age_question.search(json.dumps([DEFECTS, STRUCTURE_DETAILS]))


def test_replay_requires_the_exact_confirmed_setting():
    from lumen_studio.study.replay import qualified_strength

    setting = dict(
        raw={"sha256": "f" * 64}, alpha=0.5, profile="late", gains=[0.5], sigmas=[1, 0]
    )
    final = dict(setting, status="pass", mean_gain=0.2)
    run = dict(selection="raw", ema_decay=None, strength=setting, confirmation=final)
    assert qualified_strength(run) == setting
    for change in (
        {"mean_gain": 0},
        {"status": "unmeasured"},
        {"alpha": 1.0},
        {"status": "fail"},
    ):
        with pytest.raises(ValueError, match="confirmation"):
            qualified_strength(dict(run, confirmation=final | change))
