"""Saved-call lifecycle and resume behaviour of the named --vibe workflow. No network."""
from collections import Counter
import json
from pathlib import Path
import socket

import pytest

pytest.importorskip("particle_sliders_vibe")
pytest.importorskip("nanoodle")

from nanoodle.errors import NanoodleError
from particle_sliders_vibe import ProviderRejected, _call

from lumen_studio import vibe_cli, vibe_curation, vibe_observer, vibe_rewrite
from lumen_studio.vibe_calls import (
    CallLedger, CallState, InvalidReply, NotDispatched, ProviderUnavailable, UnresolvedCall, temporary,
)
from lumen_studio.vibe_observer import decision_values
from lumen_studio.vibe_profiles import FINAL_BOSS
from lumen_studio.vibe_teacher_checks import SINGLE_STATE

REAL_REWRITE = vibe_rewrite.rewrite_batch


def record(root, ledger, kind, inputs):
    return json.loads((root / "calls" / (ledger.key(kind, inputs) + ".json")).read_text())


def test_completed_call_runs_once_and_matches_the_shared_format(tmp_path):
    ledger = CallLedger(tmp_path)
    runs = []
    assert ledger.state("judge", {"n": 1}) is CallState.ABSENT
    for _ in range(2):
        assert ledger("judge", {"n": 1}, lambda key: runs.append(key) or {"score": 1}) == {"score": 1}
    assert len(runs) == 1 and ledger.state("judge", {"n": 1}) is CallState.DONE
    # Receipts are interchangeable with the shared add-on in both directions.
    assert _call(tmp_path, "judge", {"n": 1}, lambda key: pytest.fail("repeated")) == {"score": 1}
    assert _call(tmp_path, "judge", {"n": 2}, lambda key: {"score": 0}) == {"score": 0}
    assert ledger("judge", {"n": 2}, lambda key: pytest.fail("repeated")) == {"score": 0}
    assert record(tmp_path, ledger, "judge", {"n": 1}) == dict(
        status="done", kind="judge", inputs={"n": 1}, request_id=runs[0], result={"score": 1})


def test_interrupted_local_work_repeats_and_provider_calls_wait(tmp_path):
    ledger = CallLedger(tmp_path)

    def interrupted(key):
        raise KeyboardInterrupt

    for kind in ("render", "judge"):
        with pytest.raises(KeyboardInterrupt):
            ledger(kind, {}, interrupted)
        assert ledger.state(kind, {}) is CallState.PENDING
    assert ledger("render", {}, lambda key: "image") == "image"
    with pytest.raises(UnresolvedCall, match="unresolved call .* --retry-pending") as refused:
        ledger("judge", {}, lambda key: pytest.fail("an ambiguous paid call was repeated"))
    assert ledger.unresolved() == [refused.value.key] and temporary(refused.value)
    assert CallLedger(tmp_path, retry_pending=True)("judge", {}, lambda key: "verdict") == "verdict"
    assert ledger.unresolved() == []


@pytest.mark.parametrize("error,saved", [
    (ProviderRejected("NanoGPT", 429), dict(provider="NanoGPT", http_status=429)),
    (ProviderUnavailable("Jev", 503), dict(provider="Jev", http_status=503)),
    (NotDispatched("NANOGPT_API_KEY is required"), dict(reason="NANOGPT_API_KEY is required", dispatched=False)),
    (InvalidReply("not a decision", reply="<html>"), dict(reason="not a decision", reply="<html>")),
])
def test_known_failures_are_saved_and_tried_again(tmp_path, error, saved):
    ledger = CallLedger(tmp_path)

    def fail(key):
        raise error

    with pytest.raises(type(error)):
        ledger("judge", {}, fail)
    assert record(tmp_path, ledger, "judge", {}).items() >= dict(status="failed", **saved).items()
    assert ledger("judge", {}, lambda key: "verdict") == "verdict"


def test_unusable_reply_is_never_saved_as_complete(tmp_path):
    ledger = CallLedger(tmp_path)

    def whole(reply):
        if len(reply) != 2:
            raise InvalidReply("wrong count")

    with pytest.raises(InvalidReply):
        ledger("propose", {}, lambda key: ["only one"], check=whole)
    assert record(tmp_path, ledger, "propose", {}).items() >= dict(status="failed", reply=["only one"]).items()
    assert ledger("propose", {}, lambda key: ["a", "b"], check=whole) == ["a", "b"]
    # A reply that an earlier version saved as complete is asked again as well.
    _call(tmp_path, "propose", {"old": True}, lambda key: ["only one"])
    assert ledger("propose", {"old": True}, lambda key: ["a", "b"], check=whole) == ["a", "b"]


def test_archived_receipts_are_reused_and_changed_media_is_refused(tmp_path):
    source, derived = tmp_path / "source", tmp_path / "derived"
    image = tmp_path / "image.png"
    image.write_text("pixels")
    saved = CallLedger(source)("render", {"seed": 1}, lambda key: dict(path=str(image)))
    ledger = CallLedger(derived, archives=(source,))
    assert ledger("render", {"seed": 1}, lambda key: pytest.fail("repeated")) == saved
    assert not (derived / "calls").exists()
    image.write_text("changed")
    with pytest.raises(ValueError, match="saved output changed"):
        ledger("render", {"seed": 1}, lambda key: pytest.fail("repeated"))


def test_transport_and_server_failures_are_temporary():
    assert temporary(NanoodleError("could not reach the provider"))
    assert temporary(ProviderUnavailable("Jev", 502)) and temporary(ProviderRejected("NanoGPT", 429))
    assert not temporary(ProviderRejected("NanoGPT", 401)) and not temporary(ValueError("bad reading"))


@pytest.fixture
def world(tmp_path, monkeypatch, capsys):
    """A named Final Boss collection with local renders and a scripted judge."""
    monkeypatch.setenv("NANOGPT_API_KEY", "local-fixture-only")

    def no_network(*args, **kwargs):
        raise AssertionError("the offline fixture attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", no_network)
    world = dict(render=lambda prompt: None, judge=lambda images, state: None, propose=None, rewrite=None,
                 strong=lambda prompt: True, rendered=0, judged=0, out=tmp_path / "run")

    class Native:
        identity = {"model": "fixture-native", "product": "anima-particle-sliders"}

        def __init__(self, *args, **kwargs):
            world["loaded"] = True

        def propose(self, vibe, count, history, **kwargs):
            if world["propose"]:
                return world["propose"](count)
            return [dict(neutral=f"Scene {len(history)}-{i}", positive=f"Scene {len(history)}-{i}. Treated.")
                    for i in range(count)]

        def render(self, prompt, seed, directory, *, request_id):
            world["render"](prompt)
            world["rendered"] += 1
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "image.png"
            path.write_text(prompt)
            return dict(path=str(path), prompt=prompt, seed=seed, request_id=request_id)

        def close(self):
            pass

    def judge(provider, model, images, state, questions):
        world["judge"](images, state)
        world["judged"] += 1
        prompt = Path(images[0]).read_text().casefold()
        answers = {}
        for name, question in questions.items():
            if question["type"] == "score":
                top = 3 if world["strong"](prompt) else 2
                score = top if name in {"effect", "usable"} and "continuous shot" in prompt else 0
                answers[name] = dict(type="score", score=score, probabilities={str(i): int(i == score) for i in range(4)})
            else:
                family = name.removeprefix("family_")
                present = family in FINAL_BOSS.families and FINAL_BOSS.families[family].casefold() in prompt
                answers[name] = dict(type="noul", noul=.96 if present else .01)
        raw = {"answers": answers}
        return dict(values=decision_values(raw, questions), raw=raw)

    def rewrite(profile, scenes, *_):
        if world["rewrite"] is None:
            raise vibe_rewrite.RewriteUnavailable("no rewrite in this fixture")
        return world["rewrite"](scenes)

    monkeypatch.setattr(vibe_cli, "AnimaVibe", Native)
    monkeypatch.setattr(vibe_curation, "observe", judge)
    monkeypatch.setattr(vibe_rewrite, "rewrite_batch", rewrite)
    cli = ["--vibe", "final boss", "--out", str(world["out"]), "--model-dir", str(tmp_path), "--prompt-model",
           "fixture", "--judge-model", "fixture", "--pool-size", "3", "--replace", "1", "--best-of", "1"]

    def run(*extra):
        capsys.readouterr()
        assert vibe_cli.main(cli + list(extra)) == 0
        printed = capsys.readouterr()
        world["warning"] = printed.err
        return json.loads(printed.out.strip().splitlines()[-1])

    def calls():
        return Counter(row["status"] + " " + row["kind"]
                       for row in map(json.loads, (path.read_text() for path in world["out"].rglob("calls/*.json"))))

    return world | dict(run=run, calls=calls, cli=cli), world


def test_interrupted_render_resumes_without_retry_pending(world):
    api, state = world

    def stop_once(prompt):
        if "Scene 0-1" in prompt and not state.get("stopped"):
            state["stopped"] = True
            raise KeyboardInterrupt

    state["render"] = stop_once
    with pytest.raises(KeyboardInterrupt):
        vibe_cli.main(api["cli"])
    assert api["calls"]()["pending render"] == 1
    summary = api["run"]()
    assert summary["teacher_selection"] == "complete" and summary["qualified_contexts"] == 3
    assert not api["calls"]()["pending render"]


def test_missing_judge_key_refuses_before_model_or_saved_calls(world, monkeypatch, capsys):
    api, state = world
    monkeypatch.setattr(vibe_curation, "observe", vibe_observer.observe)
    monkeypatch.delenv("NANOGPT_API_KEY")
    with pytest.raises(SystemExit):
        vibe_cli.main(api["cli"])
    assert "NANOGPT_API_KEY is required" in capsys.readouterr().err
    assert "loaded" not in state and not state["out"].exists()
    monkeypatch.setenv("NANOGPT_API_KEY", "local-fixture-only")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        vibe_cli.main(api["cli"] + ["--judge-provider", "typesafe"])
    assert "TYPESAFE_API_KEY is required" in capsys.readouterr().err
    monkeypatch.delenv("NANOGPT_API_KEY")
    with pytest.raises(NotDispatched):
        vibe_observer.observe("nanogpt", "fixture", (), "state", {})


def test_unusable_proposal_is_asked_again_on_the_next_run(world):
    api, state = world
    state["propose"] = lambda count: [dict(neutral="Same scene", positive=f"Same scene. Treated {i}.")
                                      for i in range(count)]
    with pytest.raises(ValueError, match="duplicate neutral scenes"):
        vibe_cli.main(api["cli"])
    assert api["calls"]() == {"failed propose": 1} and state["rendered"] == 0
    state["propose"] = None
    assert api["run"]()["qualified_contexts"] == 3
    assert api["calls"]()["done propose"] == 1 and not api["calls"]()["failed propose"]


def test_server_error_during_second_read_is_reported_and_repeated(world):
    api, state = world

    def unavailable(images, state_text):
        if state_text == SINGLE_STATE:
            raise ProviderUnavailable("Jev", 503)

    state["judge"] = unavailable
    first = api["run"]()
    assert first["teacher_selection"] == "incomplete" and first["winning_policy"] == "incumbent"
    assert first["teacher_selection_issue"] == "Jev HTTP 503" and first["unresolved_calls"] == []
    assert first["unmeasured"] == ["second effect read"]
    assert first["next_step"].startswith("Rerun the same command")
    evidence = Path(first["selection_evidence"])
    assert sorted(path.name for path in evidence.glob("round-*")) == ["round-1.temporary.json"]
    assert not api["calls"]()["pending judge"]
    state["judge"] = lambda images, state_text: None
    before = state["judged"]
    second = api["run"]()
    assert second["teacher_selection"] == "complete" and state["judged"] > before
    assert "round-1.json" in {path.name for path in evidence.glob("round-*")}
    assert not (evidence / "round-1.temporary.json").exists()
    assert second["manifest"] != first["manifest"]


def test_lost_connection_during_second_read_names_the_unresolved_calls(world):
    api, state = world

    def lost(images, state_text):
        if state_text == SINGLE_STATE:
            raise NanoodleError("could not reach the judge (timed out)")

    state["judge"] = lost
    first = api["run"]()
    assert first["teacher_selection"] == "incomplete" and "could not reach" in first["teacher_selection_issue"]
    assert first["unresolved_calls"] and first["next_step"].startswith("Reconcile the recorded requests")
    state["judge"] = lambda images, state_text: None
    second = api["run"]()
    assert second["teacher_selection"] == "incomplete" and second["unresolved_calls"] == first["unresolved_calls"]
    assert second["teacher_selection_issue"].startswith("unresolved call ")
    assert "Teacher selection is incomplete" in state["warning"] and "--retry-pending" in state["warning"]
    third = api["run"]("--retry-pending")
    assert third["teacher_selection"] == "complete" and "unresolved_calls" not in third


def test_incomplete_rewrite_round_never_replaces_a_complete_selection(world):
    api, state = world
    state["strong"] = lambda prompt: "silk" in prompt or "sharpened" in prompt
    state["rewrite"] = lambda scenes: {row["id"]: row["neutral"] + ". Sharpened. A continuous shot." for row in scenes}

    def unavailable(images, state_text):
        if state_text == SINGLE_STATE and "sharpened" in Path(images[0]).read_text().casefold():
            raise ProviderUnavailable("Jev", 503)

    state["judge"] = unavailable
    summary = api["run"]()
    evidence = Path(summary["selection_evidence"])
    assert (evidence / "round-1.json").exists() and (evidence / "round-1-expanded.temporary.json").exists()
    assert summary["teacher_selection"] == "incomplete"
    ranked = json.loads((evidence / "round-1.json").read_text())["result"]
    assert "temporary_judge_failure" not in ranked["selection_policy"]
    # The exported teachers stay bound to the round that finished its checks.
    portfolio = json.loads((state["out"] / "portfolio.json").read_text())
    complete = vibe_curation.portfolio_manifest(portfolio | {"rounds": [ranked]}, .75)
    assert Path(summary["manifest"]).parent.name == complete["sha256"]
    state["judge"] = lambda images, state_text: None
    assert api["run"]()["teacher_selection"] == "complete"


def test_unusable_rewrite_reply_is_reported_and_asked_again(world, monkeypatch):
    from particle_sliders_vibe.nanoodle import NanoodleWorkflows

    api, state = world
    state["strong"] = lambda prompt: "silk" in prompt or "sharpened" in prompt
    replies = [[{"id": "scene-1", "positive": "too short"}]]

    def propose(self, context, count, history, *, request_id):
        if replies:
            return replies.pop()
        scenes = json.loads(context)["scenes"]
        return [dict(id=row["id"], positive=row["neutral"] + ". Sharpened. A continuous shot, " + "detail " * 12)
                for row in scenes]

    monkeypatch.setattr(NanoodleWorkflows, "propose", propose)
    monkeypatch.setattr(vibe_rewrite, "rewrite_batch", REAL_REWRITE)
    first = api["run"]()
    assert first["teacher_selection"] == "incomplete" and first["unmeasured"] == ["rewrite pass"]
    assert api["calls"]()["failed rewrite"] == 1 and first["unresolved_calls"] == []
    second = api["run"]()
    assert second["teacher_selection"] == "complete" and api["calls"]()["done rewrite"] == 1
    assert second["qualified_contexts"] > first["qualified_contexts"]
    assert not list(Path(second["selection_evidence"]).glob("*unavailable*"))
