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

    def adult_prompt(text):
        numeric_ages = [int(n) for n in re.findall(r"\b(\d+)-year-old\b", text, re.I)]
        return (
            bool(re.search(r"\badults?\b", text, re.I)) or bool(numeric_ages)
        ) and all(age >= 18 for age in numeric_ages)

    for concept in spec["concepts"].values():
        assert not age_question.search(json.dumps(concept["effect"]))
        for row in concept["teachers"]:
            assert all(
                adult_prompt(prompt) for prompt in (row["neutral"], *row["treatments"])
            )
    assert all(
        adult_prompt(row["prompt"])
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


def test_screen_reuses_finished_fits_quarter_raw(tmp_path, monkeypatch):
    from dataclasses import asdict
    from lumen_studio.study import training

    rates = {"g": 0.001, "d": 0.002, "particle": 0.003}
    arm = arm_path(tmp_path, rates)
    raw = {
        "path": str(tmp_path / "quarter.safetensors"),
        "bytes": 0,
        "sha256": "0" * 64,
    }
    atomic_json(arm / "cursor.json", {"step": 256})
    atomic_json(arm / "updates/000064.json", {"step": 64, "raw": raw})
    observed = []
    inference = SimpleNamespace(close=lambda: observed.append("closed"))
    monkeypatch.setattr(
        training,
        "open_training",
        lambda *a, **k: pytest.fail("completed updates replayed"),
    )
    monkeypatch.setattr(training, "open_inference", lambda *a, **k: inference)

    def compare(cases, variation, checkpoint, alpha, profile, *, owner):
        assert cases == ["rates"] and owner is inference and asdict(checkpoint) == raw
        return dict(raw=raw, status="pass", mean_gain=0.2)

    outputs = SimpleNamespace(root=tmp_path, runtime=object(), compare=compare)
    cache = SimpleNamespace(index={"identity": {"variation": "fixture"}})
    result = training.train(outputs, cache, {"rates": ["rates"]}, rates, 256, "screen")
    assert result["mean_gain"] == 0.2 and observed == ["closed"]
    assert read(arm / "screen/64.json") == result


def test_unattended_search_measures_critic_and_bank_at_viable_generator(
    tmp_path, monkeypatch
):
    from lumen_studio.study import cli

    ref = dict(g=0.004, d=0.01, particle=0.1)
    calls = []

    def train_stub(outputs, cache, panels, rates, steps, phase):
        assert phase == "screen"
        calls.append(tuple(rates.values()))
        if rates["g"] >= ref["g"]:
            raise cli.Catastrophe("split")
        gain = 0.3 if rates["d"] == 0.02 else 0.1
        return dict(status="pass", mean_gain=gain)

    monkeypatch.setattr(cli, "train", train_stub)
    options = cli.screened_rates(SimpleNamespace(root=tmp_path), None, None, ref, 256)
    assert next(options) == dict(g=0.002, d=0.02, particle=0.1)
    assert len(calls) == len(set(calls)) == 13
    assert (0.002, 0.01, 0.05) in calls and (0.002, 0.01, 0.2) in calls


def test_extending_teacher_attempts_preserves_validation_membership(tmp_path):
    from dataclasses import asdict
    from particle_sliders.candidate_failure import FileRecord
    from lumen_studio.study.teachers import validation_panels

    target = tmp_path / "off.png"
    target.write_bytes(b"frozen OFF")
    off = dict(
        integrity={"status": "pass"}, output=asdict(FileRecord.from_path(target))
    )
    accepted = [
        dict(prompt="one", seed=10, off=off),
        dict(prompt="two", seed=20, off=off),
    ]
    atomic_json(tmp_path / "validation.json", {"steps": accepted})
    groups = {
        "steps": [
            dict(prompt=p, seed=s) for p, s in (("one", 10), ("two", 20), ("three", 30))
        ]
    }
    assert validation_panels(SimpleNamespace(root=tmp_path), groups, 12) == {
        "steps": accepted
    }
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed OFF image"):
        validation_panels(SimpleNamespace(root=tmp_path), groups, 12)


def test_confirmed_raw_can_move_with_its_outcome(tmp_path):
    from dataclasses import asdict
    from lumen_studio_scoped.training import file_pin
    from lumen_studio.study.replay import locate_raw

    original = tmp_path / "RAW000064.safetensors"
    original.write_bytes(b"selected adapter")
    record = asdict(file_pin(original))
    copied = tmp_path / "export"
    copied.mkdir()
    moved = original.rename(copied / original.name)
    assert locate_raw(record, copied).path == str(moved)
    moved.write_bytes(b"different adapter")
    with pytest.raises(ValueError, match="RAW bytes"):
        locate_raw(record, copied)


def test_late_split_on_a_previously_screened_case_stops_promotion(tmp_path):
    from dataclasses import asdict
    from lumen_studio_scoped.training import file_pin
    from lumen_studio.study.training import Catastrophe, recheck_screened_cases

    rates = {"g": 0.1, "d": 0.2, "particle": 0.3}
    arm = arm_path(tmp_path, rates)
    path = tmp_path / "RAW000256.safetensors"
    path.write_bytes(b"full-budget adapter")
    raw = asdict(file_pin(path))
    atomic_json(arm / "screen/64.json", dict(status="pass"))
    atomic_json(
        arm / "fit-rate-panel.json", dict(raw=raw, status="fail", mean_gain=0.44)
    )
    with pytest.raises(Catastrophe, match="screened case"):
        recheck_screened_cases(
            SimpleNamespace(root=tmp_path), None, None, rates, 256, dict(raw=raw)
        )
    stop = read(arm / "failure.json")
    assert stop["step"] == 256 and stop["phase"] == "fit-rate-panel"
