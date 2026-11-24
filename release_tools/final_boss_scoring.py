"""Score Final Boss edits from matched Jev character and image-damage ratings."""

from math import isfinite
from statistics import fmean


SCORE = "final_boss_score"
BOSS = "boss_distinction"
ORIGINAL = "original_boss_distinction"
DAMAGE = "image_breakage"
GAIN = "boss_effect"
VERSION = "final-boss-distinction-v1"
MINOR_DAMAGE = 1 / 3
FORMULA = ("max(0, Boss_ON - Boss_OFF) * Boss_ON**3 * (1 - image_damage)**2"
           " * min(1, (1 - image_damage) / (1 - 1/3))**3")


def score_case(ratings):
    """Return a 0–1 score, or None when a required judgment is missing."""
    boss, original, damage = (ratings.get(field) for field in (BOSS, ORIGINAL, DAMAGE))
    if any(value is None for value in (boss, original, damage)):
        return None
    if any(not isfinite(value) or not 0 <= value <= 1 for value in (boss, original, damage)):
        raise ValueError("Expected Jev ratings on a 0–1 scale")
    readability = (1 - damage)**2
    excess_damage = min(1, (1 - damage) / (1 - MINOR_DAMAGE))
    return max(0, boss - original) * boss**3 * readability * excess_damage**3


def score_panel(cases, status):
    """Average the cases equally after the existing structural qualification."""
    scores = [score_case(case) for case in cases.values()]
    if status != "pass":
        return None
    if not scores or any(score is None for score in scores):
        raise ValueError("A qualified panel requires complete Jev judgments")
    return fmean(scores)
