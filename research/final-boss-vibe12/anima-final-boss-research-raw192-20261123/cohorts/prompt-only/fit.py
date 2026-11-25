"""Localized velocity targets with the native background trajectory as reference."""

import json
from pathlib import Path

from design import ROOT, save


def read(path):
    return json.loads(path.read_text())


def mask_for(path, shape, device):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255
    matte = torch.as_tensor(values, device=device)[None, None]
    matte = torch.nn.functional.interpolate(matte, size=shape, mode="area")
    return matte.unsqueeze(2)


def localized(runtime):
    import torch
    from lumen_studio.cache import save_tensor_file
    from native import guarded, pin

    output = ROOT / "localized"
    design = read(ROOT / "design.json")
    for row in design["rows"]:
        guarded()
        folder = output / row["id"]
        folder.mkdir(parents=True, exist_ok=True)
        receipt, image_file, data_file = (folder / x for x in ("target.json", "image.png", "fields.pt"))
        neutral_folder = ROOT / "images" / row["id"] / "neutral"
        mask_folder = ROOT / "masks" / row["id"]
        mask_file = mask_folder / "mask.png"
        mask_pin = read(mask_folder / "mask.json")
        neutral_pin = read(neutral_folder / "render.json")["image"]
        if pin(mask_file)["sha256"] != mask_pin["sha256"] or mask_pin["neutral_sha256"] != neutral_pin["sha256"]:
            raise RuntimeError("Mask no longer matches its neutral image")
        ident = dict(model=runtime.identity, row=row, mask=pin(mask_file), neutral=neutral_pin,
                     method="neutral-inpainted-foreground-velocity-v1")
        if receipt.exists():
            saved = read(receipt)
            if saved != dict(ident, image=pin(image_file), fields=pin(data_file)):
                raise RuntimeError("Localized teacher provenance changed")
            continue
        if image_file.exists() or data_file.exists():
            raise RuntimeError(f"Unreconciled localized work in {folder}")
        with torch.no_grad(), runtime.mixer.scales({}):
            neutral = runtime.encode(row["neutral"])
            positive = runtime.encode(row["positive"])
            initial = runtime.noise(row["seed"], 512, 512)
            matte = mask_for(mask_file, initial.shape[-2:], runtime.device)
            schedule_n, schedule_p = runtime.scheduler(10), runtime.scheduler(10)
            paths = {"neutral": [], "positive": []}
            field_n = field_p = initial
            for index, time in enumerate(schedule_n.timesteps):
                guarded()
                vn = runtime.predict(field_n, time, neutral)
                vp = runtime.predict(field_n, time, positive)
                next_n = runtime.step(schedule_n, vn, time, field_n)
                pairs = {"neutral": (field_n, vn, vp)}
                if index == 0:
                    vnp, vpp = vn, vp
                else:
                    vnp = runtime.predict(field_p, time, neutral)
                    vpp = runtime.predict(field_p, time, positive)
                pairs["positive"] = (field_p, vnp, vpp)
                vp_local = vnp + matte * (vpp - vnp)
                next_p = runtime.step(schedule_p, vp_local, time, field_p)
                next_p = torch.where(matte > 0, next_p, next_n)
                for track, (latent, base, target) in pairs.items():
                    edited = base + matte * (target - base)
                    paths[track].append(dict(position=index, timestep=float(time), trajectory=track,
                                             latent=latent.cpu().clone(), neutral=base.cpu().clone(),
                                             positive=edited.cpu().clone()))
                field_n, field_p = next_n, next_p
            image = runtime.decode(field_p).convert("RGB")
        temp_image = folder / "teacher.png.partial"
        image.save(temp_image, format="PNG")
        temp_image.replace(image_file)
        row_payload = dict(id=row["id"], variation="character-final-boss", split="train",
                           neutral=row["neutral"], positive=row["positive"], seeds=[row["seed"]])
        save_tensor_file(data_file, dict(row=row_payload, seed=row["seed"], embedding=neutral.cpu(),
                                        positive_embedding=positive.cpu(), mask=matte.cpu(),
                                        records=paths["neutral"] + paths["positive"]))
        save(receipt, dict(ident, image=pin(image_file), fields=pin(data_file)))
        print(f"Localized teacher for {row['id']}", flush=True)


def fit(runtime, stage, attempt):
    if stage == "localized":
        localized(runtime)
    else:
        from train import run
        run(runtime, stage, attempt)
