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
    assert len(calls) == len(set(calls))
    assert len(calls) <= 31
    assert (0.002, 0.04, 0.1) in calls
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
        raw={"path": str(tmp_path / "RAW.safetensors")}, alpha=0.5, profile="constant"
    )
    atomic_json(tmp_path / "RAW.scoped.json", {"completed_steps": 192})
    outputs = SimpleNamespace(root=tmp_path, concept="demo")

    def candidates(*args):
        yield rates
        pytest.fail("Another rate was tried after reserving the holdout")

    monkeypatch.setattr(cli, "fit_candidates", candidates)
    monkeypatch.setattr(cli, "train", lambda *a, **kw: {})
    monkeypatch.setattr(cli, "recheck_screened_cases", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "select_checkpoint", lambda *a: {"raw": strength["raw"]})
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
        ("Final Boss", 192),
        ("1990s anime", 192),
        ("Visually appealing character", 256),
    ):
        args = arguments(["--vibe", title, "--out", str(tmp_path), "--model-dir", "."])
        spec = read(args.spec)
        measured = read(args.spec.with_name("study-recipes.json"))["concepts"][title]
        plan = training_plan(args, tmp_path, title, spec["concepts"][title])
        start = measured.get("fresh_training", measured)
        assert plan == {"rates": start["rates"], "budget": budget, "horizon": start.get("horizon", 256)}
        if title == "Final Boss":
            assert plan == dict(rates={"g": 0.00075, "d": 0.009, "particle": 0.03}, budget=192, horizon=192)
            old = tmp_path / "failed"
            atomic_json(old / "training-config.json", dict(rates=measured["rates"], max_updates=128, horizon=256))
            assert training_plan(args, old, title, spec["concepts"][title]) == dict(
                rates=measured["rates"], budget=128, horizon=256
            )
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


@pytest.mark.parametrize("concept", ["Final Boss", "1990s anime", "Visually appealing character"])
def test_teacher_requires_its_concept_pair_check(tmp_path, concept):
    from lumen_studio.contracts import digest
    from lumen_studio.vibe import validate_manifest
    outputs = RenderFixture(tmp_path)

    def paired(neutral, positive):
        return dict(
            status="fail" if len(outputs.calls) == 2 else "pass",
            **({"separate_boss": 0.95 if len(outputs.calls) == 2 else 0.03}
               if concept == "Final Boss" else {"paired_effect": 0.1 if len(outputs.calls) == 2 else 0.95}),
            neutral=neutral["output"]["sha256"],
            positive=positive["output"]["sha256"],
        )

    outputs.teacher_pair = paired
    cfg = dict(
        neutral_ceiling=0.35,
        teachers=[dict(neutral="original", treatments=["treatment"], seed=11)],
    )
    result = collect(outputs, cfg, concept, 1, 2)
    assert result["rows"][0]["seeds"] == [104740]
    tried = read(tmp_path / "teacher-decisions.json")
    assert tried[0]["treatments"][0]["pair_check"]["status"] == "fail"
    assert not tried[0]["treatments"][0]["eligible"]
    assert result["rows"][0]["selection"]["pair_check"]["status"] == "pass"
    result["rows"][0]["selection"]["pair_check"]["neutral"] = "other image"
    result["sha256"] = digest({k: v for k, v in result.items() if k != "sha256"})
    with pytest.raises(ValueError, match="paired check"):
        validate_manifest(result)


def test_alpha_search_finds_intact_interior_and_checks_rising_edge(tmp_path):
    from particle_sliders.candidate_failure import FileRecord
    from lumen_studio.study.training import refine_strength

    path = tmp_path / "raw"
    path.write_bytes(b"saved weights")
    raw = FileRecord.from_path(path)
    case = dict(prompt="portrait", seed=3, off={"judgment": {"score": 0.0}})

    def run(center, ceiling):
        tried = {}

        def evaluate(alpha):
            gain = 1 / (1 + abs(center - alpha))
            status = "pass" if alpha <= ceiling else "unmeasured"
            tried[alpha] = dict(
                alpha=alpha,
                status=status,
                mean_gain=gain,
                pairs=[dict(case=case, on={"judgment": {"score": gain}})],
            )
            return tried[alpha]

        refine_strength(raw, [case], evaluate, strongest=5.0)
        return tried

    observations = run(1.1875, 1.3)
    assert (
        max(
            (a for a in observations if a <= 1.3),
            key=lambda a: observations[a]["mean_gain"],
        )
        == 1.1875
    )
    assert 5.0 in run(5.0, 5.0)
    assert 0.125 in run(0.125, 0.25)


def test_boss_effect_comparison_requires_exact_offs_cases_and_rubric():
    from lumen_studio.study.cli import arguments
    from lumen_studio.study.reference import compare_boss_reference

    spec_file = arguments(
        ["--vibe", "Final Boss", "--out", ".", "--model-dir", "."]
    ).spec
    spec = read(spec_file)
    rubric = spec["concepts"]["Final Boss"]["effect"]
    ref = read(spec_file.with_name("final-boss-reference.json"))
    cases = spec["confirmation_cases"]["Final Boss"]
    rows = [
        dict(
            case=dict(
                prompt=c["prompt"],
                seed=r["seed"],
                off=dict(
                    output=dict(sha256=r["off_sha256"]),
                    judgment=dict(score=r["off_score"]),
                ),
            ),
            on=dict(output=dict(sha256="abc"), integrity=dict(status="pass")),
            gain=0.1,
        )
        for c, r in zip(cases, ref["cases"], strict=True)
    ]
    comparison = compare_boss_reference(dict(pairs=rows[::-1]), rubric)
    assert comparison["status"] == "matched" and comparison["difference"] < -0.8
    assert compare_boss_reference(dict(pairs=rows), {})["status"] == "unmatched"
    rows[0]["case"]["off"]["output"]["sha256"] = "changed"
    assert compare_boss_reference(dict(pairs=rows), rubric)["status"] == "unmatched"


def test_pinned_boss_targets_use_actual_neutral_and_only_late_cues(tmp_path):
    import torch
    from PIL import Image
    from lumen_studio.contracts import file_hash
    from lumen_studio.study.boss_targets import field_context, target_field

    mask = tmp_path / "matte.png"
    Image.new("L", (2, 2), 255).save(mask)
    recipe = dict(
        rows={
            "target": dict(
                neutral_sha256="source",
                matte=mask.name,
                matte_sha256=file_hash(mask),
                explicit="strong cue",
            )
        }
    )
    row = dict(
        selection={
            side: {"output": {"sha256": sha}}
            for side, sha in [("neutral", "source"), ("positive", "target")]
        }
    )
    calls = []

    def predict(latent, step, text):
        calls.append((step, text))
        return torch.tensor([[[[[0.0, 1.0], [0.0, 1.0]]]]])

    rt = SimpleNamespace(device="cpu", encode=lambda prompt: prompt, predict=predict)
    ctx = field_context(rt, recipe, tmp_path, row, (2, 2))
    neutral = torch.zeros((1, 1, 1, 2, 2))
    positive = torch.tensor([[[[[1.0, 0.0], [1.0, 0.0]]]]])
    record = dict(latent=neutral, timestep=5.0, position=6)
    assert torch.equal(target_field(rt, record, neutral, positive, ctx), positive)
    adjusted = target_field(rt, dict(record, position=7), neutral, positive, ctx)
    assert calls == [(5.0, "strong cue")] and not torch.equal(adjusted, positive)
    assert torch.isclose(adjusted.norm(), positive.norm())
    row["selection"]["neutral"]["output"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="selected image pair"):
        field_context(rt, recipe, tmp_path, row, (2, 2))


def test_builtin_boss_recipe_is_available_to_installed_cli():
    from lumen_studio.study.cli import arguments
    from lumen_studio.contracts import file_hash

    args = arguments(
        [
            "--vibe",
            "Final Boss",
            "--out",
            ".",
            "--model-dir",
            ".",
            "--boss-targets",
            "late-cues",
        ]
    )
    recipe = read(args.boss_targets)
    assert recipe["method"] == "late-cues-v1" and len(recipe["rows"]) == 8
    for entry in recipe["rows"].values():
        assert (
            file_hash(args.boss_targets.parent / entry["matte"])
            == entry["matte_sha256"]
        )


@pytest.mark.parametrize("concept", ["Final Boss", "1990s anime", "Visually appealing character"])
def test_publish_command_keeps_completed_trial_below_reference(tmp_path, monkeypatch, concept):
    from lumen_studio.study import cli

    args = ["--vibe", concept, "--out", str(tmp_path), "--model-dir", "."]
    outcome = dict(
        vibe=concept,
        strength={"raw": {"sha256": "a"}},
        selected_step=128,
        confirmation={"mean_gain": 0.1},
    )
    atomic_json(tmp_path / "outcome.json", outcome)
    monkeypatch.setattr(cli, "run", lambda _: outcome)
    for decision, code in [
        ("retain_reference", 3),
        ("reference_unmeasured", 3),
        ("exceeds_reference", 0),
    ]:
        atomic_json(tmp_path / "reference-comparison.json", {"decision": decision})
        assert cli.main(args) == code
        assert cli.main([*args, "--require-reference-gain"]) == code
        assert cli.main([*args, "--no-require-reference-gain"]) == 0
        assert read(tmp_path / "status.json")["release_decision"] == decision
        assert read(tmp_path / "outcome.json") == outcome


@pytest.mark.parametrize(
    "http_status,succeeds", [(502, True), (400, False), (None, False)]
)
def test_transient_judge_keeps_first_receipt_and_never_repeats_unknown_transport(
    tmp_path, monkeypatch, http_status, succeeds
):
    from io import BytesIO
    from urllib.error import HTTPError
    from particle_sliders import alpha_tune_jev as api
    from particle_sliders.candidate_failure import FileRecord
    from lumen_studio.study import judge

    png = tmp_path / "image.png"
    png.write_bytes(bytes.fromhex("89504e470d0a1a0a") + b"fixture")
    img = api._Image.read(FileRecord.from_path(png))
    questions = {"check": {"type": "noul", "instructions": "Check"}}
    calls = []

    def request(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            if http_status is None:
                raise OSError("response lost")
            raise HTTPError(api.JEV_URL, http_status, "test status", {}, BytesIO())
        return BytesIO(
            json.dumps({"answers": {"check": {"type": "noul", "noul": 0.1}}}).encode()
        )

    monkeypatch.setattr(api, "urlopen", request)
    monkeypatch.setenv("NANOGPT_API_KEY", "fixture")
    monkeypatch.setattr(judge.time, "sleep", lambda _: None)
    receipts = tmp_path / "receipts"
    scorer = judge.StudyJudge(receipts, "fixture")
    for _ in range(2):
        decision = scorer._decide((img,), "state", questions)
        assert decision == ({"check": 0.1} if succeeds else None)
    assert len(calls) == (2 if succeeds else 1)
    if http_status is not None:
        (first,) = receipts.glob("*.json")
        assert read(first)["response"] == {"http_status": http_status}
    else:
        assert list(receipts.glob("*.pending"))


def test_failed_alpha_search_cannot_change_the_training_decision(tmp_path, monkeypatch):
    from lumen_studio.study import cli

    frozen = {"raw": {"path": "saved", "sha256": "fixed"}, "rates": {"g": 1}, "screened": False}
    atomic_json(tmp_path / "training-selection.json", frozen)
    monkeypatch.setattr(cli, "fit_candidates", lambda *a: pytest.fail("Re-entered training selection"))
    def failed(outputs, cache, panels, selected, *, screened):
        assert selected == frozen and not screened
        raise Unmeasured("No measured usable strength")
    monkeypatch.setattr(cli, "tune", failed)
    with pytest.raises(Unmeasured, match="usable strength"):
        cli.choose_for_confirmation(SimpleNamespace(root=tmp_path), None, {}, {}, None)
    assert read(tmp_path / "training-selection.json") == frozen
    assert not (tmp_path / "confirmation-selection.json").exists()


def test_constant_calibration_varies_only_alpha_on_the_frozen_raw(tmp_path):
    from dataclasses import asdict
    from particle_sliders.candidate_failure import FileRecord
    from lumen_studio.study.training import calibrate_constant

    path = tmp_path / "RAW"
    path.write_bytes(b"one training selection")
    raw = FileRecord.from_path(path)
    case = dict(prompt="portrait", seed=3, off={"judgment": {"score": 0.0}})
    probes = []

    def compare(cases, variation, checkpoint, alpha, profile, *, owner):
        assert checkpoint == raw and profile == "constant" and variation == "vibe"
        assert cases == [case]
        gain = 1 / (1 + abs(1.1875 - alpha))
        probes.append(alpha)
        return dict(raw=asdict(raw), alpha=alpha, profile=profile, gains=[alpha] * 10,
                    mean_gain=gain, status="pass",
                    pairs=[dict(case=case, on={"judgment": {"score": gain}})])

    outputs = SimpleNamespace(root=tmp_path, compare=compare)
    panels = {"steps": [case]}
    selected = calibrate_constant(outputs, panels, "vibe", raw)
    assert selected["alpha"] == pytest.approx(1.1875, abs=1/32)
    assert len(probes) == len(set(probes)) > 5
    assert all(row["gains"] == [row["alpha"]] * 10 for row in read(tmp_path / "strengths.json"))


def test_gallery_displays_fixed_training_probes_before_scalar_calibration(tmp_path):
    from lumen_studio.study.leaderboard import gallery

    raw = {"path": str(tmp_path / "RAW.safetensors"), "sha256": "chosen"}
    picture = {"path": str(tmp_path / "image.png")}
    pair = dict(case={"seed": 3, "off": {"output": picture}}, on={"output": picture}, gain=0.2)
    trial = dict(raw=raw, profile="constant", alpha=1.25, gains=[1.25] * 10,
                 pairs=[pair], status="pass", mean_gain=0.2)
    rates = dict(g=0.001, d=0.01, particle=0.02)
    arm = tmp_path / "rates" / "arm"
    atomic_json(arm / "cursor.json", dict(rates=rates))
    atomic_json(arm / "fit/64.json", dict(trial, alpha=1.0, gains=[1.0] * 10))
    atomic_json(tmp_path / "training-selection.json", dict(raw=raw, rates=rates))
    atomic_json(tmp_path / "strengths.json", [trial])
    gallery(tmp_path, dict(vibe="test", strength=trial, selected_step=64, confirmation=trial))
    page = (tmp_path / "leaderboard.md").read_text()
    assert page.index("## Training checkpoint comparison") < page.index("## Constant alpha calibration")
    assert "[64](rates/arm/fit/64.json) · selected" in page
    assert "1.25" in page


def test_lr_search_follows_rising_critic_and_particle_edges_at_fixed_g(tmp_path, monkeypatch):
    from math import log2
    from types import SimpleNamespace
    from lumen_studio.study import cli

    ref = dict(g=.001, d=.01, particle=.03)
    def train(outputs, cache, panels, rates, steps, phase, *, schedule_horizon):
        assert phase == "screen" and steps == 64 and schedule_horizon == 512
        score = 1 - sum(abs(log2(rates[role] / target)) for role, target in (("g", .001), ("d", .04), ("particle", .06))) / 10
        return dict(status="pass", mean_gain=score)
    monkeypatch.setattr(cli,"train",train)
    gen=cli.screened_rates(SimpleNamespace(root=tmp_path), None, None, ref, 64, 512)
    assert next(gen) == dict(g=.001, d=.04, particle=.06)
    result=read(tmp_path/"rate-search-coverage.json")
    assert result["screen_updates"] == 16 and result["alpha"] == 1.0
    assert result["status"] == "neighborhood-tested" and not result["untested"]
    attempts=read(tmp_path/"rate-search.json")
    assert len(attempts) == len({tuple(r["rates"].values()) for r in attempts})


def test_lr_search_reports_untested_edge_at_the_arm_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from lumen_studio.study import cli
    reference = dict(g=.001,d=.01,particle=.03)
    def screen(outputs, cache, panels, rates, *args, **kwargs):
        return dict(status="pass", mean_gain=rates["d"])
    monkeypatch.setattr(cli,"train",screen)
    assert next(cli.screened_rates(SimpleNamespace(root=tmp_path),None,None,reference,64,arm_budget=7))["d"]==.02
    coverage=read(tmp_path/"rate-search-coverage.json")
    assert coverage["status"]=="arm-budget" and dict(reference,d=.04) in coverage["untested"]


def test_interrupted_transaction_resumes_only_from_verified_checkpoints(tmp_path):
    from dataclasses import asdict
    from lumen_studio.study.training import recover_pending
    from lumen_studio_scoped.training import file_pin

    attempt = tmp_path / "attempts" / "first"
    attempt.mkdir(parents=True)
    base = attempt / "FULL000103.pt"
    base.write_bytes(b"committed optimizer state")
    rates = dict(g=0.1, d=0.2, particle=0.3)
    previous = dict(step=103, full=asdict(file_pin(base)), rates=rates)
    output = attempt / "FULL000104.pt"
    pending = dict(step=104, base=previous["full"], output=str(output))
    intent = tmp_path / "pending-update.json"
    receipt = tmp_path / "updates/000104.json"

    atomic_json(intent, pending)
    assert recover_pending(tmp_path, previous) == (previous, False)
    assert not intent.exists()
    archived = read(attempt / "interrupted-000104.json")
    assert archived["intent"] == pending and archived["in_memory_completion"] == "unknown"

    atomic_json(intent, pending)
    output.write_bytes(b"atomically published next optimizer state")
    row = dict(step=104, full=asdict(file_pin(output)), rates=rates)
    assert recover_pending(tmp_path, previous) == (row, True)
    atomic_json(receipt, row | {"metrics": {"step": 104}})
    assert recover_pending(tmp_path, previous) == (read(receipt), True)
    output.unlink()
    with pytest.raises(RuntimeError, match="FULL is missing"):
        recover_pending(tmp_path, previous)
    assert intent.is_file()
    receipt.unlink()
    base.write_bytes(b"wrong base")
    with pytest.raises(ValueError, match="before recovery"):
        recover_pending(tmp_path, previous)
