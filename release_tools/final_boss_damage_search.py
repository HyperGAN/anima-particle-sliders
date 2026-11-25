"""Tune Final Boss alpha or rank saved Jev measurements without rendering any images."""

import argparse
import json
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import fmean

from release_tools.final_boss_scoring import BOSS, DAMAGE, GAIN, ORIGINAL, SCORE, VERSION as LEGACY, score_case
from release_tools import final_boss_visible_damage as visible_damage


START = 1.0
INITIAL_STEP = .25
LOWER, UPPER = .25, 1.75
MIN_STEP = .03125
TRIAL_BUDGET = 9


def search(evaluate):
    """Measure both sides of a bracket, retaining the strongest measured score."""
    trials = {}
    steps = []

    def probe(alpha):
        if alpha not in trials and len(trials) < TRIAL_BUDGET:
            score = evaluate(alpha)
            if score is not None and not isfinite(score):
                raise ValueError('Alpha score must be finite')
            trials[alpha] = score

    anchor, step = START, INITIAL_STEP
    probe(anchor)
    while len(trials) < TRIAL_BUDGET and step >= MIN_STEP:
        above, below = min(UPPER, anchor + step), max(LOWER, anchor - step)
        probe(above)
        probe(below)
        old, old_step = anchor, step
        available = [alpha for alpha in (anchor, above, below) if trials.get(alpha) is not None]
        if available:
            anchor = max(available, key=trials.__getitem__)
            step *= 1.5 if anchor != old else .5
        elif step < UPPER - LOWER:
            step *= 1.5
        else:
            break
        steps.append({'anchor': old, 'step': old_step, 'above': above, 'below': below,
                      'next_anchor': anchor, 'next_step': step})
    return trials, steps


def panel_scores(cases, case_names, version=LEGACY):
    """Mean raw ratings and damage-adjusted score on the declared matched panel."""
    if not case_names or len(case_names) != len(set(case_names)) or set(cases) != set(case_names):
        raise ValueError("Each setting requires the same complete, uniquely named panel")
    score_for_version = {LEGACY: score_case, visible_damage.VERSION: visible_damage.score_case}
    if version not in score_for_version:
        raise ValueError(f"Unknown scoring version {version}")
    ratings = [cases[name] for name in case_names]
    if any(row.get(key) is None for row in ratings for key in (BOSS, ORIGINAL, DAMAGE)):
        return None
    return {
        SCORE: fmean(score_for_version[version](row) for row in ratings),
        GAIN: fmean(max(0, row[BOSS] - row[ORIGINAL]) for row in ratings),
        DAMAGE: fmean(row[DAMAGE] for row in ratings),
        BOSS: fmean(row[BOSS] for row in ratings),
        ORIGINAL: fmean(row[ORIGINAL] for row in ratings),
    }


def best_setting(settings, case_names, version=LEGACY):
    """Select by panel score, retaining the first measured setting on an exact tie."""
    best = None
    for setting in settings:
        scores = panel_scores(setting["cases"], case_names, version)
        if scores is not None and (best is None or scores[SCORE] > best[SCORE]):
            best = {key: setting[key] for key in ("method", "profile", "alpha")} | scores
    return best


def rank_snapshot(snapshot):
    """Reproduce leaderboard tie order by method, profile and then numeric alpha."""
    groups = defaultdict(list)
    for setting in snapshot["settings"]:
        groups[setting["method"]].append(setting)
    winners = []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: (row["profile"], row["alpha"]))
        winner = best_setting(ordered, snapshot["cases"], snapshot.get("scoring_version", LEGACY))
        if winner is not None:
            winners.append(winner)
    return sorted(winners, key=lambda row: (-row[SCORE], row["method"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="JSON snapshot with cases and measured settings")
    args = parser.parse_args(argv)
    with args.snapshot.open() as stream:
        standings = rank_snapshot(json.load(stream))
    print("Rank\tRecipe\tProfile\tAlpha\tFinal Boss score\tBoss gain\tImage damage")
    for rank, row in enumerate(standings, 1):
        print(f"{rank}\t{row['method']}\t{row['profile']}\t{row['alpha']:.7g}\t"
              f"{100 * row[SCORE]:.2f}\t{100 * row[GAIN]:.2f}\t{100 * row[DAMAGE]:.2f}")


if __name__ == "__main__":
    main()
