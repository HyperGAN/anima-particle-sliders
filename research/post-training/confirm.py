"""Rank chosen RAWs at their frozen alphas on characters none of them was trained or selected on.

The first panel is the sixteen later portrait scenes. The second is the sixteen held-back
scenes of the fixed vibe grid, in five framings. A RAW taught on a panel's scenes is refused.
Each image is read exactly as on the board, and every pair of characters is compared for
a repeated design in both orders.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import combinations
import os
import sys

from lumen_studio.cache import TargetCache
from lumen_studio.contracts import atomic_json
from lumen_studio.particles import ParticleAdapter
from lumen_studio.study import working_score as ws
from lumen_studio.study.design_score import summarize
from lumen_studio.study.outputs import Outputs, read

from campaign import CODE, POLICY, ROOT, open_runtime
from teacher_score import measure, repeat

second = os.environ.get("ANIMA_CONFIRMATION_PANEL") == "2"
panel = read(CODE / ("confirmation_panel_2.json" if second else "portrait_scenes_more.json"))
home = ROOT / ("confirmation-2" if second else "confirmation")
rt = open_runtime()
try:
    out = Outputs(home, rt, "Final Boss", {})
    off = {scene: asdict(out.image(entry["neutral"], entry["seed"])) for scene, entry in panel.items()}
    with ThreadPoolExecutor(24) as pool:
        for name in sys.argv[1:]:
            result = read(ROOT / "working-search" / name / "result.json")
            arm = name.rsplit("-", 1)[0]
            if arm not in ("parent", "continued"):
                taught = TargetCache(ROOT / "targets" / POLICY["arms"][arm]["targets"], verify=False)
                if {shard["row"] for shard in taught.index["shards"]} & set(panel):
                    raise SystemExit(f"{name} was taught on the confirmation scenes")
            raw, alpha = result["raw"], result["alpha"]
            adapter = ParticleAdapter(rt.transformer).to(rt.device)
            meta = adapter.load_export(raw["path"], model_identity=rt.identity)
            assert meta["step"] == result["step"] and meta["weights"] == "live"
            adapter.requires_grad_(False).eval()
            key = "confirm-" + name
            rt.mixer.add(key, adapter)
            try:
                on = {scene: asdict(out.image(entry["neutral"], entry["seed"], raw=raw["sha256"],
                                              variation=key, gains=[alpha] * 10))
                      for scene, entry in panel.items()}
            finally:
                rt.mixer.close()
            rows = {scene: job.result() for scene, job in
                    {scene: pool.submit(measure, off[scene], on[scene]) for scene in panel}.items()}
            pairs = {(a, b): pool.submit(repeat, on[a], on[b]) for a, b in combinations(panel, 2)}
            pairs = [dict(cases=[a, b], repeat=job.result()) for (a, b), job in pairs.items()]
            cases = {s: r["case"] for s, r in rows.items()}
            refused = [p["cases"] for p in pairs if p["repeat"] is None]
            complete = None not in cases.values() and not refused
            score = summarize(cases, pairs) if complete else None
            # The judge refuses a few image pairs every time. The score then lies between the
            # values it takes when each refused pair is a fully distinct or a fully repeated design.
            bounds = None
            if None not in cases.values() and 0 < len(refused) <= len(pairs) // 50:
                bounds = [summarize(cases, [dict(p, repeat=fill if p["repeat"] is None else p["repeat"])
                                            for p in pairs])["score"] for fill in (1.0, 0.0)]
            atomic_json(home / f"{name}.json", dict(name=name, raw=raw, alpha=alpha, version=ws.VERSION,
                                                    board_score=result["score"], rows=rows, pairs=pairs,
                                                    summary=score, refused_pairs=refused, bounds=bounds))
            print(name, alpha, "board", round(100 * result["score"], 2), "confirmation",
                  None if score is None else round(100 * score["score"], 2),
                  *(["between", *(round(100 * b, 2) for b in bounds)] if bounds else []), flush=True)
finally:
    rt.close()
