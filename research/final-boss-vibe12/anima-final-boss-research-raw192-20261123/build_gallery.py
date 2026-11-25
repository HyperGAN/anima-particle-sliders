"""Package the scored RAW192 renders for the local sample gallery."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "sample-gallery"


def read(name):
    return json.loads((ROOT / name).read_text())


def copy_png(receipt, name):
    source = Path(receipt["path"])
    image = source.read_bytes()
    if not image.startswith(b"\x89PNG\r\n\x1a\n") or hashlib.sha256(image).hexdigest() != receipt["sha256"]:
        raise ValueError(f"Scored render changed: {source}")
    target = PUBLIC / "images" / name
    temporary = target.with_suffix(".part")
    temporary.write_bytes(image)
    temporary.replace(target)
    return f"images/{name}"


def add_calibrated(cases, variants):
    root = ROOT / "alpha-calibration-v1"
    reports = [root / item["id"] / "selected.json" for item in variants]
    if not all(path.is_file() for path in reports):
        return False
    for path in reports:
        report = json.loads(path.read_text())
        key = report["key"]
        selected = report["selected"][0]
        setting = next(item for item in report["trials"]
                       if item["alpha"] == selected["alpha"] and item["profile"] == selected["profile"])
        for entry in setting["cases"] + selected["cases"]:
            case_id = entry["row"]["label"]
            filename = f"{case_id}-{key}-calibrated.png"
            cases[case_id]["outputs"][key]["calibrated"] = {
                "image": copy_png(entry["on"], filename),
                "alpha": selected["alpha"],
                "profile": selected["profile"],
                **entry["result"],
            }
    return True


def add_shared_search(cases, variants):
    root = ROOT / "alpha-calibration-v1"
    if not all((root / "shared-api" / f"{key}.json").is_file()
               for key in ("prompt-only", "foreground-rms", "consistent-cue")):
        return False
    for variant in variants:
        key = variant["id"]
        for case in cases.values():
            outputs = case["outputs"][key]
            outputs["shared_search"] = outputs["calibrated"]
        path = root / "shared-api" / f"{key}.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        chosen = next((entry for entry in report["reserved"] if entry["status"] == "pass"), None)
        if chosen is None:
            continue
        if all(case["outputs"][key]["calibrated"]["alpha"] == chosen["alpha"]
               and case["outputs"][key]["calibrated"]["profile"] == chosen["profile"] for case in cases.values()):
            continue
        initial = json.loads((root / key / "selected.json").read_text())
        screen = next((item for item in initial["trials"]
                       if item["alpha"] == chosen["alpha"] and item["profile"] == chosen["profile"]), None)
        if screen is None:
            screen = next(item for item in (json.loads(p.read_text()) for p in (root / key).glob("*/screen.json"))
                          if item["alpha"] == chosen["alpha"] and item["profile"] == chosen["profile"])
        for entry in screen["cases"] + chosen["cases"]:
            case_id = entry["row"]["label"]
            filename = f"{case_id}-{key}-shared-search.png"
            cases[case_id]["outputs"][key]["shared_search"] = {
                "image": copy_png(entry["on"], filename),
                "alpha": chosen["alpha"],
                "profile": chosen["profile"],
                **entry["result"],
            }
    return True


def add_jev_command(cases, preset, view):
    root = ROOT / "prompt-only-alpha-tune" / f"{preset}.alpha-tune"
    tuning = root / "tuning.json"
    if not tuning.is_file():
        return False
    alpha = json.loads(tuning.read_text())["selected"][0]
    folder = root / "renders" / alpha["profile"] / float(alpha["alpha"]).hex()
    for phase in ("search", "reserved"):
        evidence = json.loads((folder / f"{phase}.json").read_text())
        for key, entry in evidence["cases"].items():
            score = entry["observation"]
            cases[key]["outputs"]["prompt-only"][view] = {
                "image": copy_png(entry["on"], f"{key}-prompt-only-{view.replace('_', '-')}.png"),
                "alpha": alpha["alpha"],
                "profile": "constant_dose" if alpha["profile"] == "constant" else alpha["profile"],
                "gain": score["on"] - score["off"],
                "usable": score["integrity"],
            }
    return True


def main():
    board = read("leaderboard.json")
    diagnostics = read("matched-diagnostics.json")
    reference = read("historical-reference.json")
    panel = read("panel.json")
    if not board["update"] == diagnostics["step"] == reference["step"] == 192:
        raise ValueError("Expected matching RAW192 results")

    (PUBLIC / "images").mkdir(parents=True, exist_ok=True)
    reference_cases = {case["row"]["label"]: case for case in reference["cases"]}
    cases = {}
    for row in panel:
        case_id = row["label"]
        receipt = reference_cases[case_id]
        if receipt["row"] != row:
            raise ValueError(f"Historical case differs from the panel: {case_id}")
        cases[case_id] = {
            "id": case_id, "scene": row["background"], "lighting": row["lighting"],
            "prompt": row["prompt"], "seed": row["seed"],
            "off": copy_png(receipt["off"], f"{case_id}-off.png"),
            "outputs": {"archived": {"early": {
                "image": copy_png(receipt["on"], f"{case_id}-archived-early.png"),
                **receipt["result"],
            }}},
        }

    variants = []
    for row in board["variants"]:
        key = row["key"]
        variants.append({"id": key, "label": row["label"]})
        result = diagnostics["results"][key]
        for profile, arm in result["arms"].items():
            for receipt in arm["cases"]:
                source_row = receipt["row"]
                case_id = source_row["label"]
                if (source_row != reference_cases[case_id]["row"]
                        or receipt["off"]["sha256"] != reference_cases[case_id]["off"]["sha256"]):
                    raise ValueError(f"Comparison inputs differ for {case_id} {key} {profile}")
                cases[case_id]["outputs"].setdefault(key, {})[profile] = {
                    "image": copy_png(receipt["on"], f"{case_id}-{key}-{profile}.png"),
                    **receipt["result"],
                }
    calibrated = add_calibrated(cases, variants)
    shared_search = calibrated and add_shared_search(cases, variants)
    jev_command = add_jev_command(cases, "prompt-only-tuned", "jev_command")
    jev_wording = add_jev_command(cases, "prompt-only-readable-jev", "jev_wording")
    variants.append({"id": "archived", "label": "Archived broad slider"})

    data = json.dumps({"variants": variants, "cases": list(cases.values()),
                       "calibrated": calibrated, "shared_search": shared_search,
                       "jev_command": jev_command, "jev_wording": jev_wording}).replace("<", "\\u003c")
    template = (ROOT / "gallery.html").read_text()
    temporary = PUBLIC / "index.part"
    temporary.write_text(template.replace("<!-- SAMPLE DATA -->", data))
    temporary.replace(PUBLIC / "index.html")
    print(f"Published {len(cases)} cases, {len(variants)} comparisons and {len(list((PUBLIC / 'images').glob('*.png')))} PNGs")


if __name__ == "__main__":
    main()
