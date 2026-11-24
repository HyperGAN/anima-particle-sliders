"""Apply the incumbent foreground preprocessing to both frozen Anima endpoints."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def foreground(net, path):
    with Image.open(path) as image:
        if image.size != (512, 512):
            raise ValueError(f"The endpoint resolution changed for {path}")
        rgb = image.convert("RGB")
        small = np.asarray(rgb.resize((320, 320), Image.Resampling.LANCZOS), np.float32)
    mean, deviation = (np.array(values, np.float32) for values in ((.485, .456, .406), (.229, .224, .225)))
    normalized = (small / 255 - mean) / deviation
    net.setInput(normalized.transpose(2, 0, 1)[None])
    prediction = net.forward(net.getUnconnectedOutLayersNames()[0])[0, 0]
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite foreground mask for {path}")
    labels_at_size = cv2.resize((prediction >= .55).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(labels_at_size)
    if count < 2:
        raise ValueError(f"No character region for {path}")
    largest = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    coverage = float(largest.mean())
    if not .07 <= coverage <= .78:
        raise ValueError(f"Unusable foreground region {coverage:.3f} for {path}")
    grown = cv2.dilate(largest, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)))
    soft = cv2.GaussianBlur(grown.astype(np.float32), (31, 31), 6)
    soft[soft < .025] = 0
    return rgb, np.round(soft * 255).astype(np.uint8), largest.astype(bool)


def checked_image(folder, row, endpoint):
    path = folder / endpoint / "image.png"
    provenance = read(path.with_name("render.json"))
    if (digest(path) != provenance["image"]["sha256"] or provenance["prompt"] != row[endpoint]
            or provenance["seed"] != row["seed"] or provenance["variation"] is not None or provenance["gains"] is not None):
        raise ValueError(f"The pinned endpoint does not match {row['id']}/{endpoint}")
    return path, {"sha256": provenance["image"]["sha256"], "model": provenance["model"]}


def overlay(rgb, matte, color):
    alpha = Image.fromarray(np.round(matte.astype(np.float32) * 0.33).astype(np.uint8))
    return Image.composite(Image.new("RGB", rgb.size, color), rgb, alpha)


def run(args):
    cv2.setNumThreads(2)
    net_identity = digest(args.model)
    net = cv2.dnn.readNetFromONNX(str(args.model))
    rows = read(args.fixture)["pairs"]
    args.out.mkdir(parents=True, exist_ok=False)
    tile = 228
    spacing = 10
    row_height = tile + 35
    width = (tile + spacing) * 3 + spacing
    sheet = Image.new("RGB", (width, 45 + row_height * len(rows)), "#171b22")
    draw = ImageDraw.Draw(sheet)
    regular = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    heading = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    for column, name in enumerate(("Original foreground", "Boss foreground", "Added subject area")):
        draw.text((spacing + column * (tile + spacing), 12), name, fill="#f0f2f8", font=heading)

    evidence = {}
    for index, row in enumerate(rows):
        name = row["id"]
        folder = args.archive / "images" / name
        captured = read(args.archive / "masks" / name / "mask.json")
        original = args.fixture.parent / row["matte"]["file"]
        if digest(original) != row["matte"]["sha256"] or captured["model_sha256"] != net_identity:
            raise ValueError(f"The original matte or segmenter identity changed for {name}")
        source, source_info = checked_image(folder, row, "neutral")
        target, target_info = checked_image(folder, row, "positive")
        if source_info["model"] != target_info["model"] or source_info["sha256"] != captured["neutral_sha256"]:
            raise ValueError(f"The trained endpoint inputs differ for {name}")
        source_rgb, source_matte, source_core = foreground(net, source)
        target_rgb, target_matte, target_core = foreground(net, target)
        with Image.open(original) as image:
            archived = np.asarray(image.convert("L"))
        if not np.array_equal(source_matte, archived):
            raise ValueError(f"The exact original foreground preprocessing could not be reproduced for {name}")
        union = np.maximum(source_matte, target_matte)
        target_mask = args.out / f"{name}-positive.png"
        Image.fromarray(target_matte).save(target_mask)
        extra = union - source_matte
        target_only = target_core & ~source_core
        evidence[name] = dict(neutral_image=source_info, positive_image=target_info,
                              original_matte_sha256=row["matte"]["sha256"], positive_matte_sha256=digest(target_mask),
                              original_coverage=float(source_matte.mean()) / 255,
                              positive_coverage=float(target_matte.mean()) / 255,
                              union_coverage=float(union.mean()) / 255,
                              new_soft_mass=float(extra.mean()) / 255,
                              changed_core_fraction=float(target_only.sum() / target_core.sum()),
                              core_iou=float((source_core & target_core).sum() / (source_core | target_core).sum()))
        y = 65 + row_height * index
        draw.text((spacing, y - 21), name.capitalize(), fill="#acb6c6", font=regular)
        cells = (overlay(source_rgb, source_matte, "#3adbb4"), overlay(target_rgb, target_matte, "#3adbb4"),
                 overlay(target_rgb, extra, "#fd924a"))
        for column, cell in enumerate(cells):
            sheet.paste(cell.resize((tile, tile), Image.Resampling.LANCZOS), (spacing + (tile + spacing) * column, y))
        print(name, {key: round(evidence[name][key], 4) for key in ("new_soft_mass", "changed_core_fraction", "core_iou")}, flush=True)

    sheet.save(args.out / "paired-foregrounds.jpg", quality=92, optimize=True)
    (args.out / "mask-evidence.json").write_text(json.dumps({"segmentation_model_sha256": net_identity,
            "segmentation_source_sha256": digest(Path(__file__)), "rows": evidence}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "fixture", "model", "out"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    run(parser.parse_args())
