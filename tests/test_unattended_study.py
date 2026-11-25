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

    def train_stub(
        outputs, cache, panels, rates, steps, phase, *, schedule_horizon=None
    ):
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


def test_measured_defaults_are_scoped_and_existing_runs_keep_their_rates(tmp_path):
    from lumen_studio.study.cli import arguments, training_plan, REFERENCE

    args = arguments(
        ["--vibe", "1990s anime", "--out", str(tmp_path), "--model-dir", "."]
    )
    config = json.loads(args.spec.read_text())["concepts"]["1990s anime"]
    plan = training_plan(args, tmp_path, "1990s anime", config)
    chosen = dict(g=0.000375, d=0.018, particle=0.03)
    assert plan == dict(rates=chosen, budget=192, horizon=256)
    assert training_plan(
        args, tmp_path, "1990s anime", config | {"different": True}
    ) == dict(rates=REFERENCE, budget=256, horizon=256)
    args.search_rates = True
    assert training_plan(args, tmp_path, "1990s anime", config) == dict(
        rates=REFERENCE, budget=256, horizon=256
    )
    atomic_json(
        tmp_path / "training-config.json",
        {"rates": chosen, "horizon": 256, "max_updates": 128},
    )
    args.particle_lr = 0.02
    assert training_plan(args, tmp_path, "1990s anime", config) == dict(
        rates=chosen | {"particle": 0.02}, budget=128, horizon=256
    )
    args.steps = 64
    assert training_plan(args, tmp_path, "1990s anime", config)["budget"] == 64


def test_default_falls_back_lazily_and_explicit_rate_remains_fixed(monkeypatch):
    from lumen_studio.study import cli

    original = dict(g=0.1, d=0.2, particle=0.3)
    second = original | {"d": 0.1}
    called = []

    def alternatives(*args):
        called.append(True)
        yield original
        yield second

    monkeypatch.setattr(cli, "screened_rates", alternatives)
    args = SimpleNamespace(
        search_rates=False, steps=256, g_lr=None, d_lr=None, particle_lr=None
    )
    plan = dict(rates=original, budget=256, horizon=256)
    it = cli.fit_candidates(None, None, None, plan, args)
    assert next(it) == original and not called
    assert list(it) == [second] and called == [True]
    args.g_lr = original["g"]
    assert list(cli.fit_candidates(None, None, None, plan, args)) == [original]
    assert called == [True]


def test_strength_regression_gate_does_not_rank_by_screen_panel_score(tmp_path):
    from lumen_studio.study.training import select_strength

    rows = [
        dict(alpha=a, mean_gain=score, status="pass")
        for a, score in [(1.0, 0.2), (0.5, 0.1), (1.5, 0.4)]
    ]
    trials = []

    def check(row):
        trials.append(row["alpha"])
        return dict(status="fail" if row["alpha"] == 1.5 else "pass", mean_gain=-0.7)

    assert select_strength(rows, tmp_path / "strengths.json", check)["alpha"] == 1
    assert trials == [1.5, 1.0]
    assert read(tmp_path / "strengths.json")[2]["status"] == "fail"


def test_shorter_execution_cap_preserves_numeric_horizon(tmp_path, monkeypatch):
    from lumen_studio.study import training

    def inspect(*args, **kwargs):
        assert kwargs["total_steps"] == 256
        raise RuntimeError("verified numerical horizon")

    monkeypatch.setattr(training, "open_training", inspect)
    outputs = SimpleNamespace(root=tmp_path, runtime=object())
    cache = SimpleNamespace(
        index={"identity": {"variation": "x"}, "fingerprint": "fixture"}
    )
    with pytest.raises(RuntimeError, match="verified numerical horizon"):
        training.train(
            outputs,
            cache,
            {},
            dict(g=0.001, d=0.01, particle=0.03),
            192,
            "fit",
            schedule_horizon=256,
        )


def test_confirmation_reserves_one_setting_across_interruptions(tmp_path, monkeypatch):
    from lumen_studio.study import cli

    rates = dict(g=0.000375, d=0.009, particle=0.03)
    strength = dict(
        raw={"path": str(tmp_path / "RAW.safetensors")}, alpha=0.5, profile="early"
    )
    atomic_json(tmp_path / "RAW.scoped.json", {"completed_steps": 192})
    outputs = SimpleNamespace(root=tmp_path)

    def candidates(*args):
        yield rates
        pytest.fail("Another rate was tried after reserving the holdout")

    monkeypatch.setattr(cli, "fit_candidates", candidates)
    monkeypatch.setattr(cli, "train", lambda *a, **kw: {})
    monkeypatch.setattr(cli, "recheck_screened_cases", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "select_checkpoint", lambda *a: {})
    monkeypatch.setattr(cli, "tune", lambda *a, **kw: strength)
    plan = {"budget": 256, "horizon": 256}
    selection = cli.choose_for_confirmation(outputs, None, {}, plan, None)
    assert selection == dict(rates=rates, strength=strength, selected_step=192)
    assert read(tmp_path / "confirmation-selection.json") == selection
    monkeypatch.setattr(
        cli, "fit_candidates", lambda *a: pytest.fail("Repeated the search")
    )
    assert cli.choose_for_confirmation(outputs, None, {}, plan, None) == selection


def test_preexisting_outcome_preserves_original_holdout_selection(tmp_path):
    from lumen_studio.study.cli import choose_for_confirmation

    selection = dict(
        rates={"g": 0.01, "d": 0.02, "particle": 0.03},
        selected_step=64,
        strength={"alpha": 1.0, "profile": "constant", "raw": {"sha256": "archive"}},
    )
    atomic_json(
        tmp_path / "outcome.json", dict(confirmation={"status": "fail"}, **selection)
    )
    assert (
        choose_for_confirmation(SimpleNamespace(root=tmp_path), None, {}, {}, None)
        == selection
    )
    assert read(tmp_path / "confirmation-selection.json") == selection


@pytest.mark.parametrize(
    "structural_result,expected", [("fail", "fail"), ("pass", "unmeasured")]
)
def test_missing_effect_cannot_hide_a_defect_or_rank_a_partial_panel(
    tmp_path, structural_result, expected
):
    from lumen_studio.study.outputs import Outputs
    from lumen_studio_scoped.training import file_pin

    record = tmp_path / "raw"
    record.write_bytes(b"test")
    import torch

    runtime = SimpleNamespace(
        scheduler=lambda _: SimpleNamespace(sigmas=torch.tensor([1.0, 0.0]))
    )
    outputs = SimpleNamespace(
        runtime=runtime,
        image=lambda prompt, seed, **kwargs: seed,
        score=lambda seed: {
            "judgment": {"score": 0.8 if seed else None},
            "integrity": {"status": "pass" if seed else structural_result},
        },
    )
    cases = [
        {"prompt": "adult portrait", "seed": seed, "off": {"judgment": {"score": 0.2}}}
        for seed in (0, 1)
    ]
    result = Outputs.compare(outputs, cases, "demo", file_pin(record), 1.0, "constant")
    assert result["status"] == expected and result["mean_gain"] is None
    assert result["pairs"][0]["gain"] is None and result["pairs"][1][
        "gain"
    ] == pytest.approx(0.6)


def test_missing_effect_replaces_the_teacher_seed(tmp_path):
    class Missed(RenderFixture):
        def score(self, record):
            result = super().score(record)
            if record[1] == 37 or record[0] == "treatment missing":
                result["judgment"]["score"] = None
            return result

    config = {
        "neutral_ceiling": 0.35,
        "teachers": [
            {
                "neutral": "adult portrait",
                "seed": 37,
                "treatments": ["treatment missing", "treatment clear"],
            }
        ],
    }
    manifest = collect(Missed(tmp_path), config, "demo", 1, 2)
    assert manifest["rows"][0]["seeds"] == [37 + 104729]
    attempts = read(tmp_path / "teacher-decisions.json")
    assert attempts[0]["treatments"] == []
    assert attempts[1]["treatments"][0]["gain"] is None
    assert attempts[1]["treatments"][0]["eligible"] is False


def test_midrun_screen_regression_stops_before_another_update(tmp_path):
    from dataclasses import asdict
    from lumen_studio.study.training import score_rate_panel, Catastrophe
    from lumen_studio_scoped.training import file_pin

    adapter = tmp_path / "RAW000128.safetensors"
    adapter.write_bytes(b"matched split at 128")
    raw = file_pin(adapter)
    receipt = tmp_path / "fit-screen/128.json"
    failed = {"status": "fail", "raw": asdict(raw), "mean_gain": 0.47}
    with pytest.raises(Catastrophe, match="update 128"):
        score_rate_panel(tmp_path, receipt, 128, raw, lambda: failed)
    assert read(receipt) == failed
    assert read(tmp_path / "failure.json")["step"] == 128


def test_uncertain_original_scene_excludes_that_quarter(tmp_path):
    rates = {"g": 0.1, "d": 0.2, "particle": 0.3}
    arm = arm_path(tmp_path, rates)
    for step, gain in zip((64, 128, 192, 256), (0.1, 0.8, 0.4, 0.3)):
        atomic_json(
            arm / "fit" / f"{step}.json",
            {"step": step, "status": "pass", "mean_gain": gain},
        )
    atomic_json(arm / "fit-screen/128.json", {"status": "unmeasured"})
    assert select_checkpoint(tmp_path, rates, 256)["step"] == 192


def test_known_split_stays_in_fresh_final_boss_runs_and_prior_panels_are_frozen(
    tmp_path,
):
    from lumen_studio.study.cli import arguments, study_panels
    from lumen_studio.contracts import digest

    spec = read(
        arguments(["--vibe", "Final Boss", "--out", ".", "--model-dir", "."]).spec
    )
    (extra,) = spec["screen_cases"]["Final Boss"]
    assert extra["seed"] == 553304 and "older man" in extra["prompt"]
    panels, required = study_panels(spec, "Final Boss", tmp_path)
    assert required and panels["rates"] == [*spec["validation"]["rates"], extra]
    assert (
        panels["steps"] == spec["validation"]["steps"]
        and panels["confirm"] == spec["confirmation_cases"]["Final Boss"]
    )
    assert study_panels(spec, "1990s anime", tmp_path) == (spec["validation"], False)
    atomic_json(
        tmp_path / "data-config.json",
        {
            "spec": digest(
                dict(
                    teachers=spec["concepts"]["Final Boss"],
                    validation=spec["validation"],
                )
            )
        },
    )
    assert study_panels(spec, "Final Boss", tmp_path) == (spec["validation"], False)


def test_selection_only_keeps_heldout_unconsumed_and_preserves_existing_status(
    tmp_path, monkeypatch
):
    from lumen_studio.study import cli

    summary = {"status": "selected", "raw": {"sha256": "fixture"}, "steps": 32}
    args = [
        "--vibe",
        "Final Boss",
        "--model-dir",
        ".",
        "--out",
        str(tmp_path),
        "--select-only",
    ]
    monkeypatch.setattr(
        cli,
        "run",
        lambda options: (
            summary if options.select_only else pytest.fail("Unexpected phase")
        ),
    )
    assert cli.main(args) == 0
    assert read(tmp_path / "selection-summary.json") == summary
    assert read(tmp_path / "status.json") == summary
    assert not (tmp_path / "outcome.json").exists()
    atomic_json(tmp_path / "outcome.json", {"confirmed": True})
    atomic_json(tmp_path / "status.json", {"status": "confirmed"})
    assert (
        cli.main(args) == 0 and read(tmp_path / "status.json")["status"] == "confirmed"
    )


def test_bundled_recipes_bind_the_confirmed_budgets_and_holdouts(tmp_path):
    from lumen_studio.study.cli import arguments, study_panels, training_plan

    for title, budget in (
        ("Final Boss", 128),
        ("1990s anime", 192),
        ("Visually appealing character", 256),
    ):
        args = arguments(["--vibe", title, "--out", str(tmp_path), "--model-dir", "."])
        spec = read(args.spec)
        measured = read(args.spec.with_name("study-recipes.json"))["concepts"][title]
        plan = training_plan(args, tmp_path, title, spec["concepts"][title])
        assert plan == {"rates": measured["rates"], "budget": budget, "horizon": 256}
        groups, _ = study_panels(spec, title, tmp_path)
        assert [case["seed"] for case in groups["confirm"]] == measured[
            "held_out_seeds"
        ]
        assert (
            len(measured["per_case_gains"]) == 4 and min(measured["per_case_gains"]) > 0
        )
        assert all(
            re.search(r"\badult\b|\b\d+-year-old\b", case["prompt"], re.IGNORECASE)
            for case in groups["confirm"]
        )
