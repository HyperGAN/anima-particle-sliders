"""Ranks among full-panel recipe winners with one matched case omitted at a time."""
from statistics import fmean


def ranks_by_omission(winners, cases):
    values = {r["method"]:{c:v["final_boss_score"] for c,v in r["cases"].items()} for r in winners}
    if len(values) != len(winners):
        raise ValueError("Expected one winning setting per recipe")
    return rank_scores_by_omission(values, cases)


def rank_scores_by_omission(values, cases):
    if len(cases) < 2 or len(set(cases)) != len(cases):
        raise ValueError("Expected at least two distinct matched cases")
    results = {m:{} for m in values}
    if any(set(scores) != set(cases) for scores in values.values()):
        raise ValueError("Expected one complete setting per recipe")
    for omitted in cases:
        scores = {m:fmean(scores[c] for c in cases if c != omitted) for m,scores in values.items()}
        for method, score in scores.items():
            results[method][omitted] = 1 + sum(other > score for other in scores.values())
    return results
