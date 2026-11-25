"""Export a portable, image-backed snapshot of the live Vibe12 research leaderboard."""

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from os import fstat
from pathlib import Path

from PIL import Image

from release_tools.final_boss_damage_search import panel_scores, rank_snapshot
from release_tools.final_boss_scoring import BOSS, DAMAGE, GAIN, ORIGINAL, SCORE, VERSION as LEGACY
from release_tools.final_boss_visible_damage import VERSION


def read(path):
    return json.loads(path.read_text())


def write_sheet(records, folder):
    """One matched strip, named by its ordered full-resolution source hashes."""
    digest = sha256("".join(record["sha256"] for record in records).encode()).hexdigest()
    name = f"samples/{digest}.webp"
    destination = folder / name
    if destination.is_file():
        return name
    destination.parent.mkdir(exist_ok=True, parents=True)
    side = 144
    strip = Image.new("RGB", (len(records) * side, side))
    for index, record in enumerate(records):
        image_path = Path(record["path"])
        if sha256(image_path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Changed source sample at {image_path}")
        with Image.open(image_path) as original:
            image = original.convert("RGB")
            image.thumbnail((side, side), Image.Resampling.LANCZOS)
            strip.paste(image, (index * side + (side - image.width) // 2, (side - image.height) // 2))
    strip.save(destination, "WEBP", quality=85)
    return name


def export(campaign, folder):
    source = campaign / "storage/alpha-standings.json"
    with source.open() as stream:
        timestamp = datetime.fromtimestamp(fstat(stream.fileno()).st_mtime, timezone.utc)
        board = json.load(stream)
    version = board["scoring_version"]
    if board.get("quality_metric") != DAMAGE or version not in (LEGACY, VERSION):
        raise ValueError("Expected the damage-adjusted board")
    folder.mkdir(parents=True, exist_ok=True)
    search = campaign / "strength-search"
    run = "leaderboard-retune-visible" if version == VERSION else "leaderboard-retune-damage"
    queue = read(search / run / "queue.json")
    teacher = read(campaign / "storage/targets/receipt.json")
    labels = read(campaign / "recipes.json")
    cases = board["cases"]
    originals = [board["rows"][0]["cases"][case]["off"] for case in cases]
    declared = read(search / "shared-best-stock.json")
    snapshot = {"as_of_utc": timestamp.strftime("%Y-%m-%d %H:%M UTC"), "training_updates": 192,
                "weights": "raw", "ema_decay": None, "scoring_version": version, "quality_metric": DAMAGE,
                "teacher": {key: teacher[key] for key in ("cache", "cache_index_sha256", "manifest", "targets", "scenes", "model")},
                "cases": cases, "evaluation_requests": {case["id"]: case["request"] for case in declared["cases"]},
                "originals": dict(zip(cases, [record["sha256"] for record in originals])),
                "original_samples": write_sheet(originals, folder), "training_state": queue["training_state"],
                "damage_search_complete": queue["complete"], "methods": {}, "settings": []}
    if version == VERSION:
        snapshot["jev_binding"] = board["jev_binding"]
    for row in board["rows"]:
        method = row["method"]
        if method not in snapshot["methods"]:
            trained = read(campaign / "cohorts" / method / "result.json")
            snapshot["methods"][method] = {"label": labels[method], "raw_sha256": trained["raw"]["sha256"],
                                           "training_cache_sha256": trained["cache"]}
        if [row["cases"][case]["off"]["sha256"] for case in cases] != list(snapshot["originals"].values()):
            raise ValueError("Matched originals changed")
        ratings = {case: {key: row["cases"][case][key] for key in (BOSS, ORIGINAL, DAMAGE)}
                   | {"on_sha256": row["cases"][case]["on"]["sha256"]} for case in cases}
        scores = panel_scores(ratings, cases, version)
        if any(row["means"][metric] != scores[metric] for metric in (SCORE, GAIN, DAMAGE)):
            raise ValueError("Published scores do not reproduce")
        snapshot["settings"].append({key: row[key] for key in ("method", "profile", "alpha", "gains", "runs")}
                                    | {"cases": ratings, "means": scores, "receipt_sha256": row["source"]["sha256"],
                                       "samples": write_sheet([row["cases"][case]["on"] for case in cases], folder)})
    snapshot["search_paths"] = {}
    for method in snapshot["damage_search_complete"]:
        location = search / run / method
        result = read(location / "result.json")
        if result["checkpoint"]["sha256"] != snapshot["methods"][method]["raw_sha256"]:
            raise ValueError("Search path and displayed adapter disagree")
        snapshot["search_paths"][method] = read(location / "search-path.json")
    (folder / "standings.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    winners = rank_snapshot(snapshot)
    by_setting = {(s["method"], s["profile"], s["alpha"]): s for s in snapshot["settings"]}
    lines = ["# Twelve-scene Final Boss development leaderboard", "", snapshot["as_of_utc"], "",
             f"{len(snapshot['settings'])} settings from {len(winners)} trained recipes have all eight measurements. "
             f"{len(snapshot['damage_search_complete'])} recipes have finished the new search. "
             "Each row shows that recipe’s highest measured damage-adjusted Final Boss score across available strengths. "
             "Recipes with additional strengths still queued may move.", "",
             "The sample strips show these cases from left to right — " + ", ".join(cases) + ".", "",
             f"![Slider off]({snapshot['original_samples']})", "",
             "| Rank | Training recipe | Schedule and α | Final Boss score ↑ | Boss gain ↑ | Damage ↓ | Matched samples |",
             "| ---: | --- | --- | ---: | ---: | ---: | --- |"]
    if version == VERSION:
        reference = by_setting.get(("shared-best-opening", "early", 1.15625))
        if reference:
            lines[4:4] = ["**Opening-four exposure at early α 1.15625** is the visual comparison reference. "
                          "The table below ranks all recipes using the same formula.",
                          "", f"![Opening-four reference]({reference['samples']})", ""]
    for rank, winner in enumerate(winners, 1):
        method, profile, alpha = (winner[key] for key in ("method", "profile", "alpha"))
        samples = by_setting[method, profile, alpha]["samples"]
        lines.append(f"| {rank} | {labels[method]} | {profile} {alpha:.6g} | {100*winner[SCORE]:.2f} | "
                     f"{100*winner[GAIN]:.2f} | {100*winner[DAMAGE]:.2f} | [![Eight edits]({samples})]({samples}) |")
    lines.extend(["", "The JSON includes every measured setting, original and edited image hashes, per-case Jev ratings, "
                  "sample strips, native gains, RAW checkpoint hashes and completed search paths. "
                  "[All measurements](standings.json) · [Post-training handoff](../vibe12-post-training.md)", "",
                  "Reproduce the ranking from the repository root.", "",
                  "```sh", "python -m release_tools.final_boss_damage_search docs/final-boss-raw192/vibe12/standings.json", "```", ""])
    (folder / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.campaign, args.destination)
