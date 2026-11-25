"""Bounded two-sided alpha refinement for finished-image scoring."""

from math import isfinite


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
