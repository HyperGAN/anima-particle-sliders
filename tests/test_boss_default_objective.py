"""Exercise the native default, including cases that reward the wrong proxy."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
from particle_sliders.candidate_failure import FileRecord

from lumen_studio.contracts import digest
from lumen_studio.study import objectives
from lumen_studio.study.design_score import score_case, summarize
from lumen_studio.study.outputs import Outputs, read
from lumen_studio.study.reference import compare_reference
from lumen_studio.study.training import calibrate_constant

CONFIG = Path(__file__).resolve().parents[1] / "configs/anima"


def test_default_measures_damage_and_reuse_in_both_orders(tmp_path):
    pngs = []
    for i in range(4):
        path = tmp_path / f"{i}.png"
        path.write_bytes(bytes.fromhex("89504e470d0a1a0a") + str(i).encode())
        pngs.append(asdict(FileRecord.from_path(path)))
    cases = []
    for i in range(2):
        case = dict(prompt=f"portrait {i}", seed=i, off={"output": pngs[i]})
        cases.append(dict(case=case, on={"output": pngs[2 + i]}))
    seen = []

    class Judge:
        repeat = 1.0
        damage = 0.5

        def _decide(self, images, state, questions):
            if "repeat" in questions:
                seen.append([im.identity()["sha256"] for im in images])
                return {"repeat": self.repeat}
            if "damage" in questions:
                return {"damage": self.damage, "collapse": 0.0}
            return {
                "boss_distinction": 0.0
                if images[0].identity() == images[1].identity()
                else 1.0
            }

    judge = Judge()
    tested = Outputs(tmp_path, None, "Final Boss", {})
    assert tested.objective == objectives.DISTINCT_BOSS
    tested.judge = judge
    row = dict(raw={"sha256": "fixed"}, gains=[1.0] * 10, pairs=cases, mean_gain=0.99)
    rated = tested.with_objective(row)
    assert seen == [
        [pngs[2]["sha256"], pngs[3]["sha256"]],
        [pngs[3]["sha256"], pngs[2]["sha256"]],
    ]
    assert rated["mean_gain"] == 0.99 and objectives.selection_score(rated) == 0.25
    assert tested.with_objective(row) == rated and len(seen) == 2
    judge.damage = judge.repeat = 0.0
    assert objectives.measure_design(cases, judge)["score"] == 1.0
    judge.repeat = None
    assert objectives.measure_design(cases, judge)["score"] is None
    assert (
        Outputs(tmp_path, None, "1990s anime", {}).objective == objectives.MATCHED_GAIN
    )
    with pytest.raises(ValueError, match="differ"):
        objectives.selection_score(dict(mean_gain=1.0), objectives.DISTINCT_BOSS)


def test_native_alpha_uses_design_even_when_intensity_has_another_winner(tmp_path):
    raw_path = tmp_path / "RAW"
    raw_path.write_bytes(b"fixed")
    raw = FileRecord.from_path(raw_path)

    def compare(cases, variation, checkpoint, alpha, profile, *, owner):
        design = 1 / (1 + abs(alpha - 1.4375))
        gain = 1 / (1 + abs(alpha - 3.0))
        return dict(
            raw=asdict(raw),
            profile=profile,
            alpha=alpha,
            gains=[alpha] * 10,
            mean_gain=gain,
            objective="distinct_boss_score",
            objective_score=design,
            status="fail" if alpha == 1.4375 else "pass",
        )

    outputs = SimpleNamespace(
        root=tmp_path, objective="distinct_boss_score", compare=compare
    )
    selected = calibrate_constant(
        outputs, {"steps": [{"prompt": "portrait"}]}, "Final Boss", raw
    )
    assert selected["alpha"] == 1.4375 and selected["status"] == "fail"
    assert (
        max(read(tmp_path / "strengths.json"), key=lambda r: r["mean_gain"])["alpha"]
        != selected["alpha"]
    )


def test_release_comparison_uses_the_pinned_new_objective_and_exact_image_panel():
    ref = json.loads((CONFIG / "final-boss-reference.json").read_bytes())
    cfg = json.loads((CONFIG / "adult-study.json").read_bytes())
    prompts = cfg["confirmation_cases"]["Final Boss"]
    expected = {r["prompt_sha256"]: r for r in ref["cases"]}
    pairs = []
    for prompt in prompts:
        r = expected[digest(prompt["prompt"])]
        case = dict(
            prompt=prompt["prompt"],
            seed=r["seed"],
            off=dict(
                output={"sha256": r["off_sha256"]}, judgment={"score": r["off_score"]}
            ),
        )
        pairs.append(
            dict(
                case=case,
                gain=1.0,
                on=dict(output={"sha256": "candidate"}, integrity={"status": "pass"}),
            )
        )
    baseline = ref["distinct_boss"]
    recomputed = summarize(
        {k: score_case(v) for k, v in baseline["rows"].items()}, baseline["pairs"]
    )
    assert recomputed == baseline["design"] and recomputed["score"] == baseline["score"]
    bound = dict(
        pairs=pairs,
        status="pass",
        objective=objectives.DISTINCT_BOSS,
        design_protocol=objectives.protocol_id(),
    )
    rubric = cfg["concepts"]["Final Boss"]["effect"]
    weak = compare_reference("Final Boss", dict(bound, objective_score=0.01), rubric)
    strong = compare_reference("Final Boss", dict(bound, objective_score=0.8), rubric)
    assert (
        weak["decision"] == "retain_reference"
        and strong["decision"] == "exceeds_reference"
    )
    assert strong["reference_score"] == baseline["score"]
    assert (
        compare_reference(
            "Final Boss",
            dict(bound, objective_score=1.0, design_protocol="other"),
            rubric,
        )["decision"]
        == "reference_unmeasured"
    )


def test_published_reference_binds_all_three_rubrics_to_the_original_images():
    import gzip
    from particle_sliders.alpha_tune_jev import _digest, _response_values
    from lumen_studio.study.design_score import STATE, QUESTIONS, damage
    from release_tools.verify_final_boss_designs import verify_orders

    ref = json.loads((CONFIG / "final-boss-reference.json").read_bytes())[
        "distinct_boss"
    ]
    packaged = (
        CONFIG.parents[1]
        / "docs/final-boss-raw192/vibe12/constant-reference-design-receipts.json.gz"
    )
    cache = json.loads(gzip.decompress(packaged.read_bytes()))
    protocol = objectives.protocol()
    for row in ref["rows"].values():
        off, on = row["off_sha256"], row["on_sha256"]
        for key, hashes, rubric, expected in (
            ("off", [off, off], protocol["boss"], row["original_boss_distinction"]),
            ("on", [off, on], protocol["boss"], row["boss_distinction"]),
            (
                "damage",
                [on],
                dict(state=STATE, questions=QUESTIONS),
                row["image_breakage"],
            ),
        ):
            receipt_key = row["receipts"][key]
            proof = cache[receipt_key]
            request = proof["request"]
            assert (
                _digest(request) == receipt_key
                and _digest(proof["response"]) == proof["response_sha256"]
            )
            assert request["model"] == protocol["model"] and proof["status"] == "ok"
            assert (
                request["state"] == rubric["state"]
                and request["questions"] == rubric["questions"]
            )
            assert [r["sha256"] for r in request["images"]] == hashes
            values = _response_values(proof["response"], rubric["questions"])
            assert (
                damage(values) if key == "damage" else values["boss_distinction"]
            ) == expected
    for pair in ref["pairs"]:
        verify_orders(
            dict(model=protocol["model"], **protocol["repeat"]),
            pair,
            [ref["rows"][c]["on_sha256"] for c in pair["cases"]],
            cache,
            "repeat",
        )


def test_heldout_replay_rejects_proxy_rank_checkpoint_or_interval_substitution(
    tmp_path,
):
    from copy import deepcopy
    from lumen_studio.study.training import validate_calibration
    from lumen_studio.contracts import atomic_json

    raw_path = tmp_path / "RAW"
    raw_path.write_bytes(b"fixed")
    raw = FileRecord.from_path(raw_path)
    cases = [{"prompt": "portrait", "seed": 13}]

    def compare(panel, variation, checkpoint, alpha, profile, *, owner):
        return dict(
            raw=asdict(raw),
            profile=profile,
            alpha=alpha,
            gains=[alpha] * 10,
            mean_gain=1.0,
            objective="distinct_boss_score",
            objective_score=1 / (1 + abs(alpha - 1.3)),
            pairs=[{"case": c} for c in panel],
            status="pass",
        )

    outputs = SimpleNamespace(
        root=tmp_path,
        objective="distinct_boss_score",
        compare=compare,
        with_objective=lambda row: row,
    )
    best = calibrate_constant(outputs, {"steps": cases}, "Final Boss", raw)
    validate_calibration(tmp_path, best, cases, outputs)
    with pytest.raises(ValueError, match="panel"):
        validate_calibration(
            tmp_path, best, [{"prompt": "heldout", "seed": 13}], outputs
        )
    with pytest.raises(ValueError, match="objective"):
        validate_calibration(
            tmp_path, dict(best, objective="matched_intended_effect"), cases, outputs
        )
    with pytest.raises(ValueError, match="strength"):
        validate_calibration(
            tmp_path, dict(best, gains=[0.0] + best["gains"][1:]), cases, outputs
        )
    trials = read(tmp_path / "strengths.json")
    altered = deepcopy(trials)
    altered[0]["raw"]["sha256"] = "other"
    atomic_json(tmp_path / "strengths.json", altered)
    with pytest.raises(ValueError, match="weights"):
        validate_calibration(tmp_path, best, cases, outputs)


def test_trainer_rates_and_raw_both_follow_the_design_objective(tmp_path, monkeypatch):
    from lumen_studio.contracts import atomic_json
    from lumen_studio.study import cli
    from lumen_studio.study.training import arm_path, select_checkpoint

    outputs = SimpleNamespace(
        root=tmp_path,
        objective=objectives.DISTINCT_BOSS,
        with_objective=lambda row: row,
    )
    first = dict(g=0.01, d=0.01, particle=0.03)
    better = dict(first, d=0.02)

    def observation(quality):
        return dict(
            status="pass",
            mean_gain=1 - quality,
            objective=objectives.DISTINCT_BOSS,
            objective_score=quality,
        )

    def train(outputs, cache, panels, rates, *args, **kwargs):
        return observation(0.8 if rates == better else 0.3)

    monkeypatch.setattr(cli, "train", train)
    monkeypatch.setattr(cli, "rate_candidates", lambda rates: [first, better])
    assert next(cli.screened_rates(outputs, None, None, first, 64)) == better
    for step, score in ((16, 0.2), (32, 0.8), (48, 0.4), (64, 0.1)):
        atomic_json(
            arm_path(tmp_path, better) / "fit" / f"{step}.json",
            dict(observation(score), step=step),
        )
    assert select_checkpoint(tmp_path, better, 64, outputs=outputs)["step"] == 32
