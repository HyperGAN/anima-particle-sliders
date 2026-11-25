"""Jev target, paired preservation and catastrophic evidence for this experiment."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from design import ROOT, measure, save

BOSS_CRITERIA = [
    "No visible final boss character design; only an ordinary character or ordinary object.",
    "One weak visual cue of a formidable antagonist on the main subject.",
    "Several visible design cues on the main character establish a distinctive final boss.",
    "The main character is unmistakably a formidable final boss based on the character itself.",
]


def binary(instructions):
    return {"type": "noul", "instructions": instructions}


def catastrophic(image):
    return binary(
        f"Is {image} obviously catastrophically broken as a finished image: blank, severely corrupted, "
        "unintelligible noise, or lacking any coherent scene? Coherent artistic, lighting, subject and "
        "composition changes never count as a catastrophic failure."
    )


def validate_number(answer, key, upper):
    number = answer[key]
    if type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= upper:
        raise ValueError(f"Invalid Jev {key}")
    return float(number)


def compare(tag, paths, background, lighting, *, character=None):
    # Shuffle each matched pair. Bind the order independently of the returned scores.
    reverse = bool(hashlib.sha256(tag.encode()).digest()[0] & 1)
    order = list(reversed(paths)) if reverse else list(paths)
    questions = {}
    for position in (1, 2):
        img = f"image_{position}"
        questions[f"boss_{position}"] = {
            "type": "score",
            "instructions": f"Rate how strongly the main subject in {img} reads as a formidable final video-game boss solely from the subject's own outfit, accessories, expression or iconic silhouette. Ignore setting, lighting and theatrical effects.",
            "criteria": BOSS_CRITERIA,
        }
        questions[f"broken_{position}"] = catastrophic(img)
    questions["background"] = binary(
        f"Do both images preserve the same recognizable setting and the main background details of this description: {background}? "
        "Ignore the foreground character and small shifts in the exact arrangement. Answer no for a replaced setting or a clear loss of a main background detail visible in one image."
    )
    questions["lighting"] = binary(
        f"Do both images show the same overall background illumination and time of day expected from this description: {lighting}? "
        "Ignore shadows or reflections limited to the character and differences in costume colors. Answer no for an evident change of background light direction, color, brightness regime, or day/night."
    )
    if character is not None:
        questions["identity"] = binary(
            f"Do both images depict a single apparently similar {character}, ignoring changes in costume and expression? Answer no if gender, apparent age group or hair identity clearly changes."
        )
    state = "Compare the two labeled finished anime images. Judge from visible image evidence. No image has been designated as preferred."
    answers = measure(tag, order, state, questions)
    normal_order = (2, 1) if reverse else (1, 2)
    scores = [validate_number(answers[f"boss_{position}"], "score", 3) / 3 for position in normal_order]
    failures = [validate_number(answers[f"broken_{position}"], "noul", 1) for position in normal_order]
    return dict(scores=scores, gain=scores[1] - scores[0], catastrophic_probability=failures,
                background_same=validate_number(answers["background"], "noul", 1),
                lighting_same=validate_number(answers["lighting"], "noul", 1),
                identity_same=None if character is None else validate_number(answers["identity"], "noul", 1),
                reversed=reverse)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preserved(result):
    return (min(result["background_same"], result["lighting_same"], result["identity_same"]) >= 0.60
            and max(result["catastrophic_probability"]) <= 0.20
            and result["scores"][1] >= 0.65 and result["gain"] >= 0.18)


def wait_files(paths, timeout_s=210):
    end = time.monotonic() + timeout_s
    while not all(path.exists() for path in paths):
        if time.monotonic() > end:
            return False
        time.sleep(2)
    return True


def score_collection(localized=False):
    design = json.loads((ROOT / "design.json").read_text())
    def score(row):
        folders = [ROOT / "images" / row["id"] / side for side in ("neutral", "positive")]
        receipts = [folder / "render.json" for folder in folders]
        if localized:
            folders[1] = ROOT / "localized" / row["id"]
            receipts[1] = folders[1] / "target.json"
        if not wait_files(receipts):
            raise RuntimeError(f"Still waiting for committed renders: {row['id']}")
        paths = [folder / "image.png" for folder in folders]
        pins = [json.loads(receipt.read_text())["image"] for receipt in receipts]
        if any(file_hash(path) != pin["sha256"] for path, pin in zip(paths, pins, strict=True)):
            raise ValueError("Image identity differs from native receipt")
        tag = f"{'localized' if localized else 'pair'}-{row['id']}-v1"
        try:
            result = compare(tag, paths, row["background"], row["lighting"],
                             character=row["person"].removeprefix("adult "))
        except (OSError, RuntimeError) as error:
            return dict(row=row, images=pins, unmeasured=str(error))
        return dict(row=row, result=result, images=pins, tag=tag)

    with ThreadPoolExecutor(2) as pool:
        judged = []
        for entry in pool.map(score, design["rows"]):
            judged.append(entry)
            print(entry["row"]["id"], entry.get("result", entry.get("unmeasured")),
                  "qualified", "result" in entry and preserved(entry["result"]), flush=True)
    candidates = [entry for entry in judged if "result" in entry and preserved(entry["result"])]
    # Once the target and preservation checks are satisfied, rank by target effect only.
    selected = sorted(candidates, key=lambda entry: (-entry["result"]["gain"], entry["row"]["id"]))[:8]
    label = "localized-judgments.json" if localized else "collection-judgments.json"
    save(ROOT / label, dict(entries=judged, selected=selected,
        rules=dict(minimum_trait=0.65, minimum_gain=0.18, minimum_preservation=0.60,
                   maximum_catastrophic_probability=0.20, maximum_rows=8)))
    print(f"{len(selected)} selected from {len(judged)} candidates", flush=True)


def fit_observer():
    from PIL import Image
    archive = ROOT.parent / "lr-work/anima-strength-capability-224-20261121"
    fixtures = json.loads((archive / "independent-image-review-result.json").read_text())["fixture_results"]
    results = []
    for group_start in (0, 2, 4):
        group = fixtures[group_start:group_start + 2]
        original = [archive / "independent-image-review/qualification" / (row["opaque_id"] + ".png") for row in group]
        if any(file_hash(path) != row["image_sha256"] for path, row in zip(original, group, strict=True)):
            raise RuntimeError("Catastrophic qualification fixture changed")
        images = []
        for path in original:
            resized = ROOT / "calibration" / path.name
            if not resized.exists():
                resized.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(path) as image:
                    image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS).save(resized)
            images.append(resized)
        answers = measure(f"jev-integrity-calibration-{group_start}-v2", images,
                          "Judge catastrophic image integrity for each labeled image only.",
                          {f"broken_{i}": catastrophic(f"image_{i}") for i in range(1, len(images) + 1)})
        for i, row in enumerate(group, 1):
            probability = validate_number(answers[f"broken_{i}"], "noul", 1)
            decision = "fail" if probability >= 0.8 else "pass" if probability <= 0.2 else "unknown"
            results.append(dict(asset=row["opaque_id"], expected=row["status"],
                                decision=decision, probability=probability))
    qualified = all(row["decision"] == row["expected"] for row in results)
    save(ROOT / "jev-integrity-calibration.json", dict(qualified=qualified, results=results))
    print(json.dumps(results))


if __name__ == "__main__":
    {"collection": score_collection, "localized": lambda: score_collection(True),
     "calibrate": fit_observer}[sys.argv[1]]()
