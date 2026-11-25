"""Adaptive scalar search and deterministic replay of its measured objective."""

import math


def search_constant(score, maximum=5.0, tolerance=1 / 32):
    """Bracket from one in both directions, expand rising edges, refine around the best observation."""
    trials = {}

    def probe(alpha):
        if alpha not in trials:
            value = score(alpha)
            if value is not None and not math.isfinite(value):
                raise ValueError("Alpha objective must be finite or unmeasured")
            trials[alpha] = value
        return trials[alpha]

    def best():
        measured = [a for a in trials if trials[a] is not None]
        return max(measured, key=lambda a: (trials[a], -a)) if measured else None

    # The archived two-sided walk visits measured local rivals even when the first step worsens.
    step, anchor = 0.25, 1.0
    probe(anchor)
    for _ in range(9):
        below = max(tolerance, anchor - step)
        above = min(maximum, anchor + step)
        for a in (below, above):
            probe(a)
        new = best()
        if new is None:
            step *= 1.5
        else:
            step *= 0.5 if new == anchor else 1.5
            anchor = new
        if step < tolerance or len(trials) >= 9:
            break

    # Include a weak control and an overdrive probe. A rising edge cannot win silently.
    for a in (0.5, 1.5, 2.0):
        probe(min(a, maximum))
    while (leader := best()) is not None:
        smaller = [a for a in trials if a < leader]
        larger = [a for a in trials if a > leader]
        fresh = []
        if not smaller and leader > tolerance:
            fresh.append(max(tolerance, leader / 2))
        elif smaller and leader - max(smaller) > tolerance:
            fresh.append((max(smaller) + leader) / 2)
        if not larger and leader < maximum:
            fresh.append(min(maximum, leader * 1.5))
        elif larger and min(larger) - leader > tolerance:
            fresh.append((min(larger) + leader) / 2)
        if not fresh:
            break
        for a in fresh:
            probe(a)
    return trials


def verify_search(record, selection):
    """Replay a completed search from measured scalar objectives and verify its reservation."""
    by_alpha = {alpha: value for alpha, value in record}
    if len(by_alpha) != len(record):
        raise ValueError("Duplicate scalar trial")
    try:
        replay = search_constant(by_alpha.__getitem__)
    except KeyError as error:
        raise ValueError("Missing adaptive scalar trial") from error
    if list(replay.items()) != [tuple(row) for row in record]:
        raise ValueError("The recorded alpha path differs from the objective")
    measured = [alpha for alpha, value in replay.items() if value is not None]
    expected = max(measured, key=lambda a: (replay[a], -a)) if measured else None
    if expected != selection:
        raise ValueError(
            "Reserved alpha differs from the strongest measured search setting"
        )
    return expected
