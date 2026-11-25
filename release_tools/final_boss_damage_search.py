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
LOWER = .25
MIN_STEP = .03125
TRIAL_BUDGET = 9
# With no measured probe, the bracket widens until its step reaches this.
UNMEASURED_REACH = 1.5
# The first published searches clamped every probe to this alpha and never climbed. Replay only.
FIRST_PROTOCOL_CEILING = 1.75
# With no product limit given, a score still rising here is a scoring fault.
RUNAWAY = 16.0


def search(evaluate, strongest=None, climb=True):
    """Bracket the best alpha from both sides, and never stop with a usable stronger side unmeasured.

    The trial budget limits refinement. While the best measured alpha is also the
    strongest one tried, the next stronger probe is taken even after the budget is
    spent. ``strongest`` is the largest alpha the product can apply for this
    schedule; nothing beyond it is probed. ``climb=False`` with the old ceiling
    replays the first published protocol, which always stopped on the budget.
    """
    trials = {}
    steps = []

    def probe(alpha, required=False):
        if alpha in trials or not (required or len(trials) < TRIAL_BUDGET):
            return
        if strongest is None and alpha > RUNAWAY:
            raise ValueError(f'Alpha score is still rising at {max(trials):g}; the score is not penalizing damage')
        score = evaluate(alpha)
        if score is not None and not isfinite(score):
            raise ValueError('Alpha score must be finite')
        trials[alpha] = score

    anchor, step = START, INITIAL_STEP
    probe(anchor)
    while step >= MIN_STEP:
        measured = [alpha for alpha, score in trials.items() if score is not None]
        climbing = climb and bool(measured) and anchor == max(measured) and anchor != strongest
        if len(trials) >= TRIAL_BUDGET and not climbing:
            break
        above = anchor + step if strongest is None else min(strongest, anchor + step)
        below = max(LOWER, anchor - step)
        probe(above, required=climbing)
        probe(below)
        old, old_step = anchor, step
        available = [alpha for alpha in (anchor, above, below) if trials.get(alpha) is not None]
        if available:
            anchor = max(available, key=trials.__getitem__)
            step *= 1.5 if anchor != old else .5
        elif step < UNMEASURED_REACH:
            step *= 1.5
        else:
            break
        # A side the budget left unmeasured is still listed here. ``trials`` holds what was measured.
        steps.append({'anchor': old, 'step': old_step, 'above': above, 'below': below,
                      'next_anchor': anchor, 'next_step': step})
    return trials, steps


def untried_side(trials, strongest=None):
    """Name the usable side of the best measured alpha that no probe covers, if any."""
    measured = {alpha: score for alpha, score in trials.items() if score is not None}
    if not measured:
        return None
    best = max(measured, key=measured.get)
    if best == max(measured):
        return None if best == strongest else 'stronger'
    return 'weaker' if best == min(measured) else None


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
