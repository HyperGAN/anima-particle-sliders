"""Measure accidental rendering damage in the finished artwork with Jev."""

from math import isfinite

VERSION = "final-boss-visible-damage-v2"
STATE = ("Review this finished fantasy illustration on its own. The artist may have chosen a human, creature, "
         "statue, living machine or abstract antagonist. Assess only visible rendering damage.")
QUESTIONS = {"damage": {"type": "score",
    "instructions": ("How much would an illustrator need to repair an unmistakable rendering accident for this image "
                     "to read as an intentional finished drawing? Look for visibly broken contours, accidental "
                     "smearing or doubling, or chaotic patches that obscure forms. Strong shadows, brilliant glow, "
                     "an obscured face, elaborate armor, changes of subject and sparse detail can all be deliberate. "
                     "Give a clean drawing the lowest severity even when its design is strange."),
    "criteria": ["Clean intentional drawing, with no visible repair needed",
                 "Small localized rendering blemish, with the focal design clearly readable",
                 "Clear accidental defect obscures part of the focal design",
                 "Extensive rendering defects disrupt most of the focal design",
                 "Image collapse or corruption makes the artwork unreadable"]}}
QUESTIONS["collapse"] = {"type": "noul", "instructions": (
    "Is the entire image so corrupted, scrambled, or erased by noise, blank saturation, or random texture "
    "that there is no recognizable finished subject or coherent illustration to look at? An intentionally "
    "drawn creature, dark figure or bright magical scene is recognizable. Answer near zero for a readable illustration.")}


def damage(values):
    return max(values["damage"], values["collapse"])


def score_case(ratings):
    boss, original, damage = (ratings.get(key) for key in
                              ("boss_distinction", "original_boss_distinction", "image_breakage"))
    if None in (boss, original, damage):
        return None
    if not all(isfinite(v) and 0 <= v <= 1 for v in (boss, original, damage)):
        raise ValueError("Invalid Jev rating")
    return max(0, boss - original) * boss**3 * (1 - damage)
