"""Qualify seed-B candidates with the main selection's own machinery (no GPU), then gate the norms.

  qualify.py judge [--retry-pending]   after the assets stage: judge every candidate, write
                                       seed-b/judgments.json and one derived manifest per qualified candidate.
                                       exit 0 judged, 2 no admissible candidate (verdict written), 4 unmeasured.
  qualify.py gate                      after the targets stage: G3 with fallback, admission, writes
                                       seed-b/qualification.json. exit 0 passed, 2 no admissible candidate.

Per candidate (scene, seed), exactly as lumen_studio.vibe_curation.curate_round measures a case:
neutral image -> IMAGE_STATE + image_questions(FINAL_BOSS, candidate=False); positive image ->
image_questions(candidate=True); finish_gain = finish_score(on) - finish_score(off); integrity "pass" only
when max(catastrophic) <= 0.20. Qualified = vibe_teacher_selection.qualifies(row, 0.75) and the selector's
second read (SINGLE_STATE + single_question, recomputed finish gain > 0). Judge typesafe/jev-1.14 through
provider nanogpt (the portfolio identity). Every request/response is a particle_sliders_vibe._call receipt
under STORE/qualify/calls (resumable; finished calls are never repeated).

Run: PY (env-stock) with PYTHONPATH=STORE/source:STORE/vibe-deps and NANOGPT_API_KEY in the environment.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import seed_split as S  # noqa: E402

for _path in (S.VIBE_DEPS, S.SOURCE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

RULES = {
    "selection": "lumen_studio.vibe_teacher_selection.qualifies(row, 0.75): integrity pass, finish_gain > 0, "
                 "positive effect >= 0.75 and > neutral effect",
    "integrity": "pass only when max(neutral, positive) catastrophic <= 0.20 (vibe_curation)",
    "second_read": "vibe_teacher_checks.SINGLE_STATE/single_question effect, finish_score with the first-read "
                   "overbake/layout (select_collection.recheck); recomputed finish gain > 0",
    "order": "first qualified candidate in A+1, A+2, A+3 is seed B",
    "G3": {"bounds": list(S.G3_BOUNDS),
           "statistic": "amendment 1 (seed-b/gate-amendment-1.json): overall ||positive-neutral|| of the seed-B slots "
                        "over the seed-A slots they replace, both as the geometric mean of the pooled per-position "
                        "ratios and as the norm-weighted ratio; both must be in bounds. Pooled per-position ratios "
                        "are recorded as a diagnostic only",
           "fallback": "advance the scene whose own geometric-mean ratio is farthest from 1 to its next qualified "
                       "candidate; a scene without one keeps all 20 seed-A records"},
    "admission": f"at least {S.MINIMUM_SPLIT_SCENES} of 5 scenes with a seed B, else no admissible candidate",
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_stable(path, record):
    """Write a decision record, keeping the existing file when only its time differs.

    Later stages pin these files (targets pins judgments.json; the crossed cache identity pins
    qualification.json), so an idempotent rerun must not change their bytes.
    """
    if path.is_file():
        saved = S.read(path)
        if {k: v for k, v in saved.items() if k != "time"} == json.loads(json.dumps(
                {k: v for k, v in record.items() if k != "time"})):
            return saved
    S.write_json(path, record)
    return record


def selection_modules():
    from lumen_studio import vibe_observer, vibe_portfolio, vibe_teacher_checks, vibe_teacher_selection
    from lumen_studio.contracts import digest
    from lumen_studio.vibe_profiles import FINAL_BOSS, profile_for

    for module in (vibe_observer, vibe_portfolio, vibe_teacher_checks, vibe_teacher_selection):
        if not Path(module.__file__).resolve().is_relative_to(S.SOURCE):
            raise RuntimeError(f"{module.__name__} did not load from the pinned main checkout")
    return {"observer": vibe_observer, "portfolio": vibe_portfolio, "checks": vibe_teacher_checks,
            "selection": vibe_teacher_selection, "digest": digest, "profile": FINAL_BOSS, "profile_for": profile_for}


def judge_identity(manifest, modules):
    identity = manifest["collection_identity"]
    profile = modules["profile_for"](manifest["criterion"])
    if profile is not modules["profile"] or json.loads(json.dumps(asdict(profile))) != identity["profile"]:
        raise ValueError("The Final Boss profile differs from the portfolio identity")
    engine = {name: S.file_sha(S.SOURCE / "lumen_studio" / name) for name in identity["engine"]}
    if engine != identity["engine"]:
        raise ValueError("The pinned main vibe engine differs from the portfolio identity")
    return {"provider": identity["provider"], "model": identity["judge"], "profile": profile, "engine": engine}


class Reader:
    """select_collection.reading over particle_sliders_vibe._call: content-addressed, resumable receipts."""

    def __init__(self, root, provider, model, modules, observe=None, retry_pending=False):
        from functools import partial
        from particle_sliders_vibe import _call

        self.call = partial(_call, Path(root), retry_pending=retry_pending)
        self.provider, self.model, self.modules = provider, model, modules
        self.observe = observe or modules["observer"].observe

    def __call__(self, outputs, state, questions):
        inputs = {"provider": self.provider, "model": self.model, "state": state, "questions": questions,
                  "images": [output["sha256"] for output in outputs]}
        key = self.modules["digest"]({"kind": "judge", "inputs": inputs})
        paths = [output["path"] for output in outputs]
        if any(S.file_sha(path) != output["sha256"] for path, output in zip(paths, outputs, strict=True)):
            raise ValueError("Teacher candidate image hash changed")
        receipt = self.call("judge", inputs, lambda _: self.observe(self.provider, self.model, paths, state, questions))
        values = self.modules["observer"].decision_values(receipt["raw"], questions)
        if values != receipt["values"]:
            raise ValueError("Saved readings differ from the typed judge response")
        return values, key


def measure_candidate(reader, modules, profile, row_id, seed, outputs):
    """One candidate row in vibe_curation's case structure, its qualification and the second read."""
    observer, portfolio = modules["observer"], modules["portfolio"]
    neutral, off_key = reader((outputs["neutral"],), observer.IMAGE_STATE, observer.image_questions(profile, candidate=False))
    values, on_key = reader((outputs["positive"],), observer.IMAGE_STATE, observer.image_questions(profile, candidate=True))
    before = portfolio.finish_score(neutral["effect"], neutral["overbake"], neutral["layout"])
    on = portfolio.finish_score(values["effect"], values["overbake"], values["layout"])
    severity = max(neutral["catastrophic"], values["catastrophic"])
    integrity = "pass" if severity <= .20 else "fail" if severity >= .80 else "unmeasured"
    family_scores = {name: values[f"family_{name}"] for name in profile.families}
    family = portfolio.observed_family(family_scores, minimum=profile.family_minimum)
    row = {"row": row_id, "seed": seed, "case": modules["digest"](outputs["neutral"]["prompt"]),
           "method": f"seed-b:{seed}", "observed_family": family if family in profile.families else None,
           "integrity": integrity, "usable": values["usable"], "finish_gain": on - before,
           "neutral": {"output": outputs["neutral"], "judgment": {"score": neutral["effect"]},
                       "readings": neutral, "decision": off_key},
           "positive": {"output": outputs["positive"], "judgment": {"score": values["effect"]},
                        "readings": values, "decision": on_key}}
    row["qualifies"] = modules["selection"].qualifies(row, S.MINIMUM_SCORE)
    row["second_read"] = None
    if row["qualifies"]:
        second = {}
        for side in ("neutral", "positive"):
            alternative, key = reader((row[side]["output"],), modules["checks"].SINGLE_STATE,
                                      modules["checks"].single_question(profile))
            quality = row[side]["readings"]
            second[side] = {"effect": alternative["effect"], "decision": key,
                            "finish": portfolio.finish_score(alternative["effect"], quality["overbake"], quality["layout"])}
        gain = second["positive"]["finish"] - second["neutral"]["finish"]
        row["second_read"] = second | {"finish_gain": gain, "passed": gain > 0}
    row["qualified"] = bool(row["qualifies"] and row["second_read"] and row["second_read"]["passed"])
    return row


def judge_all(plan, candidates, reader, modules, profile, workers=4):
    """{row: [candidate rows in fixed order]} for every declared candidate."""
    jobs = [(row_id, seed) for row_id in plan["scene_order"] for seed in plan["candidates"][row_id]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {job: pool.submit(measure_candidate, reader, modules, profile, job[0], job[1],
                                    candidates[job[0]][str(job[1])]) for job in jobs}
        results = {job: future.result() for job, future in futures.items()}
    return {row_id: [results[(row_id, seed)] for seed in plan["candidates"][row_id]] for row_id in plan["scene_order"]}


def admitted_scenes(order, rows):
    """Scenes with at least one qualified candidate (admission needs MINIMUM_SPLIT_SCENES of them)."""
    return [row_id for row_id in order if any(item["qualified"] for item in rows[row_id])]


def stop_verdict(reason, extra):
    record = {"passed": False, "status": "stop", "verdict": "no admissible candidate", "reason": reason,
              "time": now(), "rules": RULES} | extra
    write_stable(S.QUALIFICATION, record)
    print(json.dumps({"status": "stop", "verdict": record["verdict"], "reason": reason}), flush=True)
    return 2


def judge(retry_pending=False, observe=None):
    manifest = S.checked_manifest()
    plan = S.seed_plan(manifest)
    assets = S.read(S.ASSETS_RECEIPT)
    if not assets.get("passed") or assets["seed_plan"] != S.pin(S.SEED_PLAN):
        raise RuntimeError("Run the assets stage (R0, R1 and candidate renders) first")
    modules = selection_modules()
    identity = judge_identity(manifest, modules)
    reader = Reader(S.QUALIFY_ROOT, identity["provider"], identity["model"], modules, observe, retry_pending)
    try:
        rows = judge_all(plan, assets["candidates"], reader, modules, identity["profile"])
    except ValueError:
        raise
    except Exception as error:  # provider/transport failure or an unresolved receipt: nothing is decided
        print(f"unmeasured: {type(error).__name__}: {error}; rerun (with --retry-pending only after reconciling a pending call)", flush=True)
        return 4
    for row_id, items in rows.items():
        for item in items:
            if item["qualified"]:
                path = S.candidate_manifest_path(row_id, item["seed"])
                derived = S.derived_manifest(manifest, row_id, item["seed"], item, modules["digest"])
                from lumen_studio.vibe import validate_manifest

                validate_manifest(derived)
                if path.exists() and S.read(path) != derived:
                    raise ValueError(f"A different derived manifest already exists: {path}")
                if not path.exists():
                    S.write_json(path, derived)
                item["derived_manifest"] = S.pin(path)
            print(row_id[:13], item["seed"], "qualifies", item["qualifies"], "second read",
                  None if item["second_read"] is None else round(item["second_read"]["finish_gain"], 4),
                  "qualified", item["qualified"], flush=True)
    scenes = admitted_scenes(plan["scene_order"], rows)
    record = {"status": "judged", "time": now(), "scene_order": plan["scene_order"], "candidates": rows,
              "scenes_with_qualified_candidate": scenes, "rules": RULES,
              "judge": {"provider": identity["provider"], "model": identity["model"], "engine": identity["engine"]},
              "assets": S.pin(S.ASSETS_RECEIPT), "seed_plan": S.pin(S.SEED_PLAN), "manifest": S.pin(S.MANIFEST),
              "source_commit": S.source_head(), "implementation": S.pin(Path(__file__).resolve()),
              "calls": str(S.QUALIFY_ROOT / "calls")}
    write_stable(S.JUDGMENTS, record)
    if len(scenes) < S.MINIMUM_SPLIT_SCENES:
        return stop_verdict(f"only {len(scenes)} scenes have a qualified candidate",
                            {"judgments": S.pin(S.JUDGMENTS), "seed_b": {row: None for row in plan["scene_order"]}})
    print(f"judged: {len(scenes)} scenes have a qualified candidate; run the targets stage", flush=True)
    return 0


# --- G3 ------------------------------------------------------------------------------------------

def edit_norms(directory):
    """{row: {(trajectory, position): ||positive - neutral||}} from a cache on disk (float64 on CPU)."""
    import torch

    index = S.read(Path(directory) / "index.json")
    norms = {}
    for item in index["shards"]:
        shard = torch.load(Path(directory) / item["path"], map_location="cpu", weights_only=True)
        norms[item["row"]] = {(r["trajectory"], r["position"]): (r["positive"].double() - r["neutral"].double()).norm().item()
                              for r in shard["records"]}
        del shard
    return norms


def g3_select(order, qualified, a_norms, b_norms):
    """qualified: {row: [qualified seeds in the fixed order]}. Pure decision with its trail."""
    current = {row: (qualified[row][0] if qualified[row] else None) for row in order}
    trail = []
    while True:
        split = {row: current[row] for row in order if current[row] is not None}
        if len(split) < S.MINIMUM_SPLIT_SCENES:
            trail.append({"split": split, "status": "stop", "reason": f"{len(split)} split scenes"})
            return {"status": "stop", "seed_b": current, "trail": trail, "pooled": None,
                    "verdict": "no admissible candidate"}
        import math

        pooled, individual = S.g3_ratios(split, a_norms, b_norms)
        overall = S.g3_overall(split, a_norms, b_norms)
        outside = [k for k, ratio in enumerate(pooled) if not S.G3_BOUNDS[0] <= ratio <= S.G3_BOUNDS[1]]
        step = {"split": split, "overall": overall, "pooled": pooled, "per_scene": individual,
                "positions_outside_bounds_diagnostic": outside}
        if S.g3_in_bounds(overall):
            trail.append(step | {"status": "pass"})
            return {"status": "pass", "seed_b": current, "trail": trail, "pooled": pooled, "per_scene": individual,
                    "overall": overall, "verdict": None}
        own = overall["per_scene_geometric_mean"]
        advance = [sorted(split, key=lambda row: (-abs(math.log(own[row])), order.index(row)))[0]]
        moves = {}
        for row in advance:
            remaining = qualified[row][qualified[row].index(current[row]) + 1:]
            moves[row] = {"from": current[row], "to": remaining[0] if remaining else None}
            current[row] = remaining[0] if remaining else None
        trail.append(step | {"status": "advance", "moves": moves})


def gate():
    manifest = S.checked_manifest()
    plan = S.seed_plan(manifest)
    judgments = S.read(S.JUDGMENTS)
    targets = S.read(S.TARGETS_RECEIPT)
    if not targets.get("passed") or targets["judgments"] != S.pin(S.JUDGMENTS):
        raise RuntimeError("Run the targets stage on the current judgments first")
    caches = {(item["row"], item["seed"]): item for item in targets["caches"]}
    qualified = {row: [item["seed"] for item in judgments["candidates"][row] if item["qualified"]]
                 for row in plan["scene_order"]}
    if set(caches) != {(row, seed) for row, seeds in qualified.items() for seed in seeds}:
        raise ValueError("The candidate caches do not cover exactly the qualified candidates")
    for key, item in caches.items():
        if S.pin(Path(item["directory"]) / "index.json") != item["index"]:
            raise ValueError(f"Candidate cache changed: {key}")
    a_norms = edit_norms(S.SHARED_CACHE)
    b_norms = {}
    for (row, seed), item in caches.items():
        b_norms[(row, seed)] = edit_norms(item["directory"])[row]
    decision = g3_select(plan["scene_order"], qualified, a_norms, b_norms)
    readings = {row: [{"seed": item["seed"], "integrity": item["integrity"], "finish_gain": item["finish_gain"],
                       "neutral_effect": item["neutral"]["judgment"]["score"],
                       "positive_effect": item["positive"]["judgment"]["score"],
                       "qualifies": item["qualifies"],
                       "second_read_gain": None if item["second_read"] is None else item["second_read"]["finish_gain"],
                       "qualified": item["qualified"]} for item in judgments["candidates"][row]]
                for row in plan["scene_order"]}
    record = {"passed": decision["status"] == "pass", "status": decision["status"], "verdict": decision["verdict"],
              "time": now(), "seed_a": plan["seed_a"], "seed_b": decision["seed_b"],
              "split_scenes": [row for row in plan["scene_order"] if decision["seed_b"][row] is not None],
              "pooled": decision["pooled"], "overall": decision.get("overall"), "trail": decision["trail"],
              "readings": readings, "rules": RULES,
              "judgments": S.pin(S.JUDGMENTS), "targets": S.pin(S.TARGETS_RECEIPT), "seed_plan": S.pin(S.SEED_PLAN),
              "shared_cache": S.shared_receipt()["cache"], "implementation": S.pin(Path(__file__).resolve())}
    write_stable(S.QUALIFICATION, record)
    print(json.dumps({"status": record["status"], "seed_b": record["seed_b"], "overall": record["overall"],
                      "pooled": record["pooled"]}, indent=2),
          flush=True)
    return 0 if record["passed"] else 2


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("judge", "gate"))
    parser.add_argument("--retry-pending", action="store_true")
    args = parser.parse_args(argv)
    return judge(args.retry_pending) if args.stage == "judge" else gate()


if __name__ == "__main__":
    sys.exit(main())
