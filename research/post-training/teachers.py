"""Measure the current Boss teachers and choose cleaner ones for the most damaged scenes."""

from dataclasses import asdict

import torch
from particle_sliders.alpha_tune_jev import _Image

from lumen_studio.cache import TargetCache
from lumen_studio.contracts import atomic_json, file_hash
from lumen_studio.study.boss_targets import replace_boss_teachers
from lumen_studio.study.design_score import QUESTIONS, STATE, damage
from lumen_studio.study.objectives import protocol
from lumen_studio.study.outputs import Outputs, read

from campaign import CODE, PARENT_STUDY, POLICY, ROOT, open_runtime

root = ROOT / "teachers"
root.mkdir(exist_ok=True)
cache = TargetCache(PARENT_STUDY / POLICY["parent"]["cache"], verify=True)
assert cache.index["fingerprint"] == POLICY["parent"]["cache_fingerprint"]
rt = open_runtime()
try:
    out = Outputs(root, rt, "Final Boss", {})
    boss = protocol()["boss"]

    def measure(neutral, image):
        off, on = _Image.read(neutral), _Image.read(image)
        before = out.judge._decide((off, off), **boss)
        after = out.judge._decide((off, on), **boss)
        accident = out.judge._decide((on,), STATE, QUESTIONS)
        pair = out.boss_teacher_pair(dict(output=asdict(neutral)), dict(output=asdict(image)))
        if None in (before, after, accident):
            return dict(image=asdict(image), gain=None, damage=None, separate=pair["status"])
        return dict(
            image=asdict(image),
            boss=after["boss_distinction"],
            original=before["boss_distinction"],
            gain=after["boss_distinction"] - before["boss_distinction"],
            damage=damage(accident),
            separate=pair["status"],
        )

    scenes = []
    for shard in cache.index["shards"]:
        row = torch.load(cache.directory / shard["path"], map_location="cpu", weights_only=True)["row"]
        neutral = out.image(row["neutral"], shard["seed"])
        current = out.image(row["positive"], shard["seed"])
        archived = row["selection"]
        scenes.append(
            dict(
                scene=shard["row"],
                seed=shard["seed"],
                neutral=row["neutral"],
                positive=row["positive"],
                neutral_image=neutral,
                replayed=[
                    neutral.sha256 == archived["neutral"]["output"]["sha256"],
                    current.sha256 == archived["positive"]["output"]["sha256"],
                ],
                current=measure(neutral, current),
            )
        )
    if any(s["current"]["damage"] is None for s in scenes):
        raise SystemExit("A current teacher is unmeasured; rerun to retry the judge")
    ranked = sorted(scenes, key=lambda s: -s["current"]["damage"])
    chosen = [s["scene"] for s in ranked[: POLICY["teachers"]["replace"]]]
    atomic_json(
        root / "current.json",
        dict(
            cache=cache.index["fingerprint"],
            scenes=[dict(s, neutral_image=asdict(s["neutral_image"])) for s in scenes],
            chosen=chosen,
        ),
    )
    for s in scenes:
        c = s["current"]
        print(
            f"{s['scene'][-8:]} {'replace' if s['scene'] in chosen else 'anchor '} "
            f"gain {c['gain']:+.3f} damage {c['damage']:.3f} separate {c['separate']} "
            f"replay {s['replayed']}",
            flush=True,
        )
    proposals_path = CODE / "alternatives.json"
    if not proposals_path.exists():
        raise SystemExit("Write alternatives.json for the chosen scenes, then rerun")
    proposals = read(proposals_path)
    if set(proposals) != set(chosen):
        raise SystemExit("alternatives.json must cover exactly the chosen scenes")
    measured = {}
    for s in scenes:
        if s["scene"] in chosen:
            measured[s["scene"]] = [
                dict(p, **measure(s["neutral_image"], out.image(p["positive"], s["seed"])))
                for p in proposals[s["scene"]]
            ]
    built = {}
    for arm, rule in POLICY["teachers"]["rules"].items():
        replacements, decisions = {}, {}
        for s in scenes:
            if s["scene"] not in chosen:
                continue
            current = s["current"]
            eligible = [
                o
                for o in measured[s["scene"]]
                if o["gain"] is not None
                and o["separate"] == "pass"
                and o["damage"] < current["damage"]
                and (not rule["protect_gain"] or o["gain"] >= current["gain"])
            ]
            best = min(eligible, key=lambda o: (o["damage"], -o["gain"])) if eligible else None
            decisions[s["scene"]] = dict(
                current=current, proposals=measured[s["scene"]], selected=best and best["name"]
            )
            if best:
                replacements[s["scene"]] = dict(
                    positive=best["positive"],
                    explicit=best["explicit"],
                    selected=dict(
                        output=dict(
                            prompt=best["positive"],
                            seed=s["seed"],
                            path=best["image"]["path"],
                            sha256=best["image"]["sha256"],
                        ),
                        readings={k: best[k] for k in ("boss", "original", "gain", "damage")},
                    ),
                )
        atomic_json(
            root / f"selection-{arm}.json",
            dict(rule=rule, proposals=file_hash(proposals_path), decisions=decisions,
                 replaced=sorted(replacements)),
        )
        if not replacements:
            print(arm, "has no qualified proposal; improve the proposals before training", flush=True)
            continue
        cache_ = replace_boss_teachers(rt, cache, ROOT / "targets" / arm, replacements)
        built[arm] = dict(
            fingerprint=cache_.index["fingerprint"],
            replaced=cache_.index["identity"]["teacher_replacement"]["rows"],
        )
        print(arm, "targets", cache_.index["fingerprint"], "replaced",
              {k[-8:]: decisions[k]["selected"] for k in replacements}, flush=True)
    atomic_json(root / "targets.json", dict(parent=cache.index["fingerprint"], arms=built))
finally:
    rt.close()
