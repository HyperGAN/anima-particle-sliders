"""Score distinct boss designs while retaining the matched effect and damage measurements."""
from itertools import combinations
from math import isfinite
from statistics import fmean


def summarize(case_scores, pairs):
    expected = {frozenset(p) for p in combinations(case_scores,2)}
    repeats = {frozenset(p["cases"]): p["repeat"] for p in pairs}
    values = (*case_scores.values(), *repeats.values())
    if (not expected or len(repeats) != len(pairs) or set(repeats) != expected or
            any(not isfinite(v) or not 0 <= v <= 1 for v in values)):
        raise ValueError("Expected all distinct case pairs and scores in the unit interval")
    scores = []
    for (a,b), repetition in repeats.items():
        left,right = case_scores[a],case_scores[b]
        scores.append((left+right-repetition*min(left,right))/2)
    return {"base":fmean(case_scores.values()), "repeat":fmean(repeats.values()), "score":fmean(scores)}
