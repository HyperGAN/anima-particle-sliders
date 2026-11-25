"""Apply the identical Final Boss protocol to saved native PG 0.12 fits and diagnostic arms."""

from importlib.metadata import version
import sys
import torch
from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.particles import ParticleAdapter
from lumen_studio.contracts import atomic_json, file_hash
from lumen_studio.study.outputs import Outputs, read
from evaluate import retest, cases, independent_effect_outputs
from locations import RESULTS as here

entries = read(here / "inventory.json")
legacy = read(here.parent / "product/docs/final-boss-raw192/vibe12/standings.json")


def run(shard, parts, metric):
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
    methods = sorted(
        n
        for n, r in entries.items()
        if n not in legacy["methods"] and r["run"] not in ("anime", "appeal-v2")
    )
    try:
        for n in methods[int(shard) :: int(parts)]:
            r = entries[n]
            root = here / "methods" / n
            raw = r["raw"]
            assert file_hash(raw["path"]) == raw["sha256"]
            old = r["pg"] == "0.11.1"
            model = legacy["teacher"]["model"] if old else rt.identity
            adapter = ParticleAdapter(rt.transformer).to(rt.device)
            info = adapter.load_export(raw["path"], model_identity=model)
            assert (
                info["step"] == r["step"]
                and info["weights"] == "live"
                and info["model_identity"] == model
            )
            rt.mixer.add(n, adapter.requires_grad_(False).eval())
            outputs = Outputs(root, rt, "Final Boss", {})
            if metric == "matched_boss_gain":
                outputs = independent_effect_outputs(outputs)
            proof = read(r["parity"])
            panel = (
                [
                    (cases[p["case"]]["request"], p["on"]["sha256"])
                    for p in proof["cases"]
                ]
                if old
                else [
                    (
                        dict(prompt=p["case"]["prompt"], seed=p["case"]["seed"]),
                        p["on"]["output"]["sha256"],
                    )
                    for p in proof["pairs"]
                ]
            )
            verified = []
            for request, expected in panel:
                img = outputs.image(
                    **request, raw=raw["sha256"], variation=n, gains=proof["gains"]
                )
                assert img.sha256 == expected, (n, img.sha256, expected)
                verified.append(
                    dict(seed=request["seed"], actual=img.sha256, expected=expected)
                )
            atomic_json(
                root / "pg012-parity.json",
                dict(
                    method=n,
                    raw_sha256=raw["sha256"],
                    replay_pg=version("particlegan"),
                    training_pg=r["pg"],
                    training_model=model,
                    replay_model=rt.identity,
                    gains=proof["gains"],
                    cases=verified,
                ),
            )
            retest(n, outputs, raw, metric, archive=False)
            rt.mixer.close()
            del adapter
    finally:
        rt.close()


if __name__ == "__main__":
    run(*sys.argv[1:])
