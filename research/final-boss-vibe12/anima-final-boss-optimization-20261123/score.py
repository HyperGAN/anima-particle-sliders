"""User-selected Final Boss leaderboard objective."""

from math import fsum
from statistics import fmean


METRIC = "final_score"
EFFECT = "boss_effect"
BREAKAGE = "image_breakage"
READABILITY_EXPONENT = 2
REPORT = "boss-effect-tuning.json"
PRIOR_REPORTS = ("distinct-boss-tuning.json", "boss-adjusted-tuning.json", "weighted-tuning.json", "tuning.json")
ADDITIVE_WEIGHTS = {"character": 3, "identity": 1, "scene": 1, "lighting": 1, "framing": 1, "coherence": 1}
QUALITY = ("identity", "scene", "lighting", "framing", "coherence")
DISTINCTION = "boss_distinction"
ORIGINAL_DISTINCTION = "original_boss_distinction"
PRESERVATION = "preservation_quality"
PRODUCT_OBJECTIVE = {"multiplier": DISTINCTION, "average": list(QUALITY), "aggregation": "mean of case scores"}
OBJECTIVE = {"primary": "mean per-case Final Boss effect adjusted for visible breakage",
             "effect": "nonnegative matched ON minus OFF Boss distinction",
             "readability_multiplier": "(1 - Jev image breakage severity)^2",
             "tiebreak_average": list(QUALITY)}
TOTAL_HELP = ("For each case, multiply positive Final Boss effect by the square of one minus image damage. "
              "Average the adjusted scores across matched cases. Equal totals use preservation as a tiebreaker.")


def case_effect(scores):
    on, off = scores[DISTINCTION], scores[ORIGINAL_DISTINCTION]
    return None if on is None or off is None else round(on - off, 12)


def adjusted_effect(effect, damage):
    if effect is None or damage is None:
        return None
    return round(max(0, effect) * (1 - damage) ** READABILITY_EXPONENT, 12)


def preservation(scores):
    return fmean(scores[name] for name in QUALITY)


def ranking(scores, metric=METRIC):
    if metric == METRIC:
        return scores[METRIC], scores[PRESERVATION]
    return (scores[metric],)


def weighted_total(scores):
    return fsum(weight * scores[name] for name, weight in ADDITIVE_WEIGHTS.items()) / sum(ADDITIVE_WEIGHTS.values())


def boss_adjusted_total(scores, character=DISTINCTION):
    return scores[character] * fmean(scores[name] for name in QUALITY)


def aggregate_total(cases, character=DISTINCTION):
    return fmean(boss_adjusted_total(scores, character) for scores in cases)
