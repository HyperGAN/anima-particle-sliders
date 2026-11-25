"""Prepare immutable training caches and the common historical reference."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent / "anima-character-final-boss-20261122"
FORE = ROOT / "cohorts/foreground"
sys.path.insert(0, str(FORE))
from design import save
from judge import compare
from native import MODEL_DIR, guarded, pin, read, render
from train import schedules

VARIATION = "character-final-boss"
UNIFORM = (0, 2, 4, 5, 7, 9)


def source_index():
    return read(EXPERIMENT / "target-cache/index.json")


def write_cache(key, teachers, records, identity_fields):
    import torch
    from lumen_studio.cache import save_tensor_file
    from lumen_studio.contracts import digest

    root = ROOT / "caches" / key
    root.mkdir(parents=True, exist_ok=True)
    if (root / "index.json").exists():
        return read(root / "index.json")
    identity = {**source_index()["identity"], "layout": key, **identity_fields, "teachers": teachers}
    identity["manifest_sha256"] = digest([item["row"] for item in teachers])
    fingerprint = digest(identity)
    shards = []
    for example in records:
        example["fingerprint"] = fingerprint
        path = root / f"{example['row']['id']}-{example['seed']}.pt"
        if path.exists():
            previous = torch.load(path, map_location="cpu", weights_only=True)
            if (previous["fingerprint"] != fingerprint or previous["row"] != example["row"]
                    or len(previous["records"]) != len(example["records"])
                    or any(not torch.equal(a[field], b[field])
                           for a, b in zip(previous["records"], example["records"], strict=True)
                           for field in ("latent", "neutral", "positive"))):
                raise ValueError("Previously captured cache field differs")
        else:
            save_tensor_file(path, example)
        shards.append(dict(path=path.name, sha256=pin(path)["sha256"], count=len(example["records"]),
                           row=example["row"]["id"], seed=example["seed"]))
    index = dict(identity=identity, input_fingerprint=fingerprint,
                 fingerprint=digest(dict(input=fingerprint, shards=shards)), shards=shards)
    save(root / "index.json", index)
    return index


def derive():
    import torch

    torch.set_num_threads(2)
    index = source_index()
    teachers = index["identity"]["teachers"]
    parent = EXPERIMENT / "target-cache"
    source = [torch.load(parent / shard["path"], map_location="cpu", weights_only=True)
              for shard in index["shards"]]
    for key in ("neutral-trajectory", "mixed-trajectory-ordered", "uniform-steps", "divergence"):
        subsets, schedule = [], {}
        for scene, sample in enumerate(source):
            rows = sample["records"]
            tag = sample["row"]["id"]
            if key == "neutral-trajectory":
                subset = [r for r in rows if r["trajectory"] == "neutral"]
            elif key == "mixed-trajectory-ordered":
                tracks = {position: ("neutral", "positive")[(position + scene) % 2]
                          for position in range(10)}
                subset = [r for r in rows if r["trajectory"] == tracks[r["position"]]]
                subset.sort(key=lambda r: r["position"])
                if [r["position"] for r in subset] != list(range(10)):
                    raise ValueError("Expected one mixed-path field per timestep")
                schedule[tag] = tracks
            else:
                strength = {i: sum(float((r["positive"]-r["neutral"]).square().mean())
                                       for r in rows if r["position"] == i) for i in range(10)}
                positions = UNIFORM if key == "uniform-steps" else tuple(
                    sorted(range(10), key=lambda i: (-strength[i], i))[:6])
                schedule[tag] = dict(positions=sorted(positions), residual_mse_by_position=strength)
                subset = [r for r in rows if r["position"] in positions]
            subsets.append(dict(sample, records=subset))
        result = write_cache(key, teachers, subsets, dict(parent=index["fingerprint"], positions=schedule))
        print(f"{key}: {sum(item['count'] for item in result['shards'])} verified fields", flush=True)


def capture_teacher(runtime, teacher, key, full_mask):
    import torch
    from fit import mask_for
    from lumen_studio.cache import save_tensor_file

    row = dict(teacher["row"])
    mask = Path(teacher["mask"]["path"])
    if key == "prompt-only":
        mask = full_mask
    else:
        common = ("Regal plate cuirass, large angular shoulders, high dark collar, "
                  "ornate fitted sash and a stern commanding expression.")
        row["positive"] = row["positive"].replace(row["boss"], common, 1)
        if row["positive"] == teacher["row"]["positive"]:
            raise ValueError("Original costume clause missing")
        row["boss"] = common
    folder = ROOT / "captured" / key / row["id"]
    folder.mkdir(parents=True, exist_ok=True)
    evidence = folder / "target.json"
    payload = folder / "fields.pt"
    image_file = folder / "image.png"
    identity = dict(row=row, model=runtime.identity, mask=pin(mask), neutral=teacher["neutral"],
                    method=key)
    if evidence.exists():
        receipt = read(evidence)
        if receipt != identity | {"image": pin(image_file), "fields": pin(payload)}:
            raise ValueError("Captured teacher bytes changed")
        return receipt, torch.load(payload, map_location="cpu", weights_only=True)
    with torch.no_grad(), runtime.mixer.scales({}):
        neutral, positive = runtime.encode(row["neutral"]), runtime.encode(row["positive"])
        n = p = runtime.noise(row["seed"], 512, 512)
        matte = mask_for(mask, n.shape[-2:], runtime.device)
        n_scheduler, p_scheduler = runtime.scheduler(10), runtime.scheduler(10)
        records = {"neutral": [], "positive": []}
        for i, ts in enumerate(n_scheduler.timesteps):
            guarded()
            vn, vp = runtime.predict(n, ts, neutral), runtime.predict(n, ts, positive)
            vnp, vpp = (vn, vp) if i == 0 else (runtime.predict(p, ts, neutral), runtime.predict(p, ts, positive))
            n_next = runtime.step(n_scheduler, vn, ts, n)
            p_next = runtime.step(p_scheduler, vnp + matte * (vpp - vnp), ts, p)
            for name, latent, base, target in (("neutral", n, vn, vp), ("positive", p, vnp, vpp)):
                edited = base + matte * (target - base)
                records[name].append(dict(position=i, timestep=float(ts), trajectory=name,
                    latent=latent.cpu().clone(), neutral=base.cpu().clone(), positive=edited.cpu().clone()))
            n, p = n_next, torch.where(matte > 0, p_next, n_next)
        image = runtime.decode(p).convert("RGB")
    tmp = folder / "image.png.partial"
    image.save(tmp, format="PNG")
    tmp.replace(image_file)
    data = dict(row=dict(id=row["id"], variation=VARIATION, split="train",
                         neutral=row["neutral"], positive=row["positive"], seeds=[row["seed"]]),
                seed=row["seed"], embedding=neutral.cpu(), positive_embedding=positive.cpu(),
                records=records["neutral"]+records["positive"])
    save_tensor_file(payload, data)
    receipt = identity | {"image": pin(image_file), "fields": pin(payload)}
    save(evidence, receipt)
    return receipt, data


def prepare_panel(runtime):
    from lumen_studio.particles import ParticleAdapter

    panel = read(ROOT / "panel.json")
    historical = read(EXPERIMENT / "baseline-result.json")
    retain = {item["row"]["label"]: item for item in historical["cases"]}
    path = Path(historical["raw"]["path"])
    if pin(path)["sha256"] != historical["raw"]["sha256"]:
        raise ValueError("Historical RAW192 changed")
    adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
    meta = adapter.load_export(path, model_identity=runtime.identity)
    if meta["step"] != 192 or meta["weights"] != "live":
        raise ValueError("Historical reference is not RAW192")
    runtime.mixer.add("archived-final-boss", adapter)
    gains = schedules(runtime)["early"]
    jobs = {}
    for row in panel:
        label = row["label"]
        if label in retain:
            continue
        off = render(runtime, row["prompt"], row["seed"], ROOT / "shared-off" / label)
        on = render(runtime, row["prompt"], row["seed"], ROOT / "shared-baseline" / label,
                    gains=gains, variation="archived-final-boss")
        jobs[label] = (row, off, on)
        print(f"Rendered shared held-out pair {label}", flush=True)
    runtime.mixer.close()
    return panel, retain, jobs, gains


def score_panel(panel, retain, jobs, gains):
    def measure(item):
        row, off, on = item
        visual = compare(f"benchmark-reference-{row['label']}-v1",
            [Path(off["path"]), Path(on["path"])], row["background"], row["lighting"],
            character=row["person"].removeprefix("adult "))
        return dict(row=row, off=off, on=on, result=visual)
    with ThreadPoolExecutor(2) as pool:
        new = list(pool.map(measure, jobs.values()))
    by_label = retain | {item["row"]["label"]:item for item in new}
    cases = [by_label[row["label"]] for row in panel]
    flags = [score for row in cases for score in row["result"]["catastrophic_probability"]]
    status = "fail" if max(flags) >= .8 else "unmeasured" if max(flags) > .2 else "pass"
    save(ROOT / "historical-reference.json", dict(step=192, cases=cases, gains=gains, status=status))


def main():
    if sys.argv[1] == "derive":
        derive()
        return
    guarded()
    import torch
    from PIL import Image
    from lumen_studio.backends.anima import TurboRuntime
    from lumen_studio.core_source import verify_core

    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(19 * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    verify_core()
    full_mask = ROOT / "whole-frame-mask.png"
    if not full_mask.exists():
        Image.new("L", (512, 512), 255).save(full_mask)
    runtime = TurboRuntime(MODEL_DIR, device="cuda:0", verify=True)
    try:
        panel = prepare_panel(runtime)
        teachers = source_index()["identity"]["teachers"]
        for key in ("prompt-only", "consistent-cue"):
            pairs = [capture_teacher(runtime, teacher, key, full_mask) for teacher in teachers]
            result = write_cache(key, [x[0] for x in pairs], [x[1] for x in pairs],
                                 dict(parent=source_index()["fingerprint"]))
            print(f"{key}: {sum(item['count'] for item in result['shards'])} captured fields", flush=True)
    finally:
        runtime.close()
    if not (ROOT / "historical-reference.json").exists():
        score_panel(*panel)


if __name__ == "__main__":
    main()
