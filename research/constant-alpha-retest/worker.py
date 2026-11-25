"""Replay each saved Vibe12 RAW on the PG 0.12 runtime and prove archived output parity."""

from importlib.metadata import version
import sys
import torch
from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.contracts import atomic_json, file_hash
from lumen_studio.particles import ParticleAdapter
from lumen_studio.study.outputs import Outputs, read

from locations import RESULTS as here

campaign = here.parent.parent / "anima-final-boss-vibe12-20261124"
study = read(here.parent / "product/docs/final-boss-raw192/vibe12/standings.json")
source = study["teacher"]["model"]


def run(stage, shard, parts, objective=None):
    assert version("particlegan") == "0.12.0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, 19 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    )
    rt = TurboRuntime(
        "/mnt/nvme/sliders/qualification/models/anima-turbo-v11",
        device="cuda:0",
        verify=True,
    )
    try:
        assert all(
            rt.identity[key] == source[key]
            for key in ("identity_schema", "model", "sha256")
        )
        for name in sorted(study["methods"])[int(shard) :: int(parts)]:
            root = here / "methods" / name
            training = read(campaign / "cohorts" / name / "result.json")
            raw = training["raw"]
            assert training["step"] == 192 and training["ema_decay"] is None
            assert (
                file_hash(raw["path"])
                == raw["sha256"]
                == study["methods"][name]["raw_sha256"]
            )
            assert training["cache"] == study["methods"][name]["training_cache_sha256"]
            adapter = ParticleAdapter(rt.transformer).to(rt.device)
            info = adapter.load_export(raw["path"], model_identity=source)
            assert (
                info["weights"] == "live"
                and info["step"] == 192
                and info["model_identity"] == source
            )
            adapter.requires_grad_(False).eval()
            rt.mixer.add(name, adapter)
            outputs = Outputs(root, rt, "Final Boss", {})
            proof = root / "pg012-parity.json"
            expected = next(
                r
                for r in study["settings"]
                if r["method"] == name
                and r["profile"] == "constant"
                and r["alpha"] == 0.6375
            )
            if stage == "qualify-board" or not proof.is_file():
                records = []
                for c in study["cases"]:
                    image = outputs.image(
                        **study["evaluation_requests"][c],
                        raw=raw["sha256"],
                        variation=name,
                        gains=expected["gains"],
                    )
                    records.append(
                        dict(
                            case=c,
                            expected=expected["cases"][c]["on_sha256"],
                            actual=image.sha256,
                        )
                    )
                proof_data = dict(
                    method=name,
                    raw_sha256=raw["sha256"],
                    replay_pg=version("particlegan"),
                    training_pg="0.11.1",
                    training_model=source,
                    replay_model=rt.identity,
                    alpha=0.6375,
                    cases=records,
                )
                atomic_json(proof, proof_data)
            q = read(proof)
            assert q["raw_sha256"] == raw["sha256"] and q["replay_model"] == rt.identity
            assert len(q["cases"]) == 8 and all(
                p["expected"]
                == p["actual"]
                == expected["cases"][p["case"]]["on_sha256"]
                for p in q["cases"]
            )
            print(stage, name, flush=True)
            if stage == "retest-board":
                from evaluate import retest, independent_effect_outputs

                if objective == "matched_boss_gain":
                    outputs = independent_effect_outputs(outputs)
                retest(name, outputs, raw, objective)
            rt.mixer.close()
            del adapter
    finally:
        rt.close()


if __name__ == "__main__":
    run(*sys.argv[1:])
