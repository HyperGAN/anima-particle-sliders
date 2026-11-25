"""Rank Vibe12 training recipes at one declared constant alpha on matched cases."""
import argparse
import json
import math
from pathlib import Path
from statistics import fmean


def rank_training(board, alpha):
    cases = board["cases"]
    methods = board["methods"]
    if not cases or len(set(cases)) != len(cases) or not methods:
        raise ValueError("A training comparison requires distinct cases and methods")
    if board["weights"].lower() != "raw" or board["ema_decay"] is not None:
        raise ValueError("The training leaderboard requires RAW weights")
    sigmas = next(iter(methods.values()))["sigmas"]
    if not math.isfinite(alpha) or len(sigmas) < 2 or any(m["sigmas"] != sigmas for m in methods.values()):
        raise ValueError("A matched comparison requires a finite alpha and identical sampler intervals")
    rankings = {}
    baselines = {}
    for row in board["settings"]:
        if row["profile"] != "constant" or row["alpha"] != alpha:
            continue
        name = row["method"]
        if name not in methods or name in rankings or set(row["cases"]) != set(cases):
            raise ValueError(f"Incomplete, unknown or duplicate training row: {name}")
        steps = len(sigmas) - 1
        if row["gains"] != [alpha] * steps:
            raise ValueError(f"Unequal inference strengths for {name}")
        gains = []
        for case in cases:
            result = row["cases"][case]
            before = result["original_boss_distinction"]
            after = result["boss_distinction"]
            if not all(math.isfinite(score) for score in (before, after)):
                raise ValueError(f"Missing effect for {name}/{case}")
            if before != baselines.setdefault(case, before):
                raise ValueError(f"Different OFF scores for {case}")
            gains.append(after - before)
        rankings[name] = dict(method=name, raw_sha256=methods[name]["raw_sha256"],
                              mean_gain=fmean(gains))
    if set(rankings) != set(methods):
        raise ValueError("Every method must have a complete measurement at the same alpha")
    # Equal recorded means share a rank. The name only stabilizes display order.
    ordered = sorted(rankings.values(), key=lambda r: (-round(r["mean_gain"], 12), r["method"]))
    previous = None
    for index, row in enumerate(ordered, start=1):
        mean = round(row["mean_gain"], 12)
        if mean != previous:
            rank = index
        row["rank"] = rank
        previous = mean
    return ordered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("standings", type=Path)
    parser.add_argument("--alpha", required=True, type=float)
    args = parser.parse_args()
    print(json.dumps(rank_training(json.loads(args.standings.read_text()), args.alpha), indent=2))
