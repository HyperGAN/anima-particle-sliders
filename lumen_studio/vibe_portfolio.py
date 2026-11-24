"""The finish-aware portfolio selection used by Anima vibe curation."""
from functools import lru_cache
from itertools import combinations
from math import fsum
from statistics import fmean

DUPLICATE_THRESHOLD = .65
FAMILY_MINIMUM = .55
FAMILY_MARGIN = .12
FAMILY_LIMIT = 2


def finish_score(effect, overbake, layout):
    return effect * (1.0 - max(overbake, layout))


def observed_family(scores, *, minimum=FAMILY_MINIMUM):
    first, second, *_ = sorted(scores, key=lambda name: -scores[name])
    if scores[first] >= minimum and scores[first] - scores[second] >= FAMILY_MARGIN:
        return first
    return None


def pair_audit(rows, observe, pool):
    def measure_pair(pair):
        a, b = pair
        orders = [observe(left, right) for left, right in ((a, b), (b, a))]
        return dict(left=a["case"], right=b["case"], mean=fmean(row["near"] for row in orders), orders=orders)

    checked = list(pool.map(measure_pair, combinations(rows, 2)))
    return dict(pairs=checked, average=fmean(row["mean"] for row in checked) if checked else None,
                near_duplicates=sum(row["mean"] >= DUPLICATE_THRESHOLD for row in checked))


def select(candidates, limit, *, allow_partial=False):
    """Maximize independent suitability across cases under visible-family capacities."""
    rows = [{row["method"]: row for row in group if row["integrity"] == "pass" and row["observed_family"]}
            for group in candidates]
    families = sorted({row["observed_family"] for group in rows for row in group.values()})
    keys = {name: index for index, name in enumerate(families)}

    @lru_cache(maxsize=None)
    def pick(index, remaining):
        if index == len(rows):
            return 0, 0.0, ()
        best = None
        if allow_partial:
            count, score, choices = pick(index + 1, remaining)
            best = count, score, (None, *choices)
        for method, row in rows[index].items():
            slot = keys[row["observed_family"]]
            if remaining[slot] == 0:
                continue
            counts = list(remaining)
            counts[slot] -= 1
            suffix = pick(index+1, tuple(counts))
            if suffix is None:
                continue
            count, score, choices = suffix
            option = count + 1, row["usable"] + score, (method, *choices)
            if best is None or option[:2] > best[:2]:
                best = option
        return best

    solution = pick(0, (limit,) * len(keys))
    if solution is None:
        raise ValueError("No diverse assignment from confidently observed designs")
    return [group[method] for group, method in zip(rows, solution[2], strict=True) if method is not None]


def distinct_finish(rows, proof, *, panel_size=None):
    """Give finish credit to the best mutually distinct outputs in a fixed-size panel."""
    panel_size = len(rows) if panel_size is None else panel_size
    if not panel_size or panel_size < len(rows):
        raise ValueError("Distinct finish requires a positive size covering all selected cases")
    case_names = {row["case"] for row in rows}
    audited = {frozenset((pair["left"], pair["right"])): pair["mean"] for pair in proof["pairs"]}
    expected = {frozenset(pair) for pair in combinations(case_names, 2)}
    if len(case_names) != len(rows) or audited.keys() != expected or len(audited) != len(proof["pairs"]):
        raise ValueError("Distinct finish requires a complete pair audit for these cases")
    best_total, credited = 0.0, ()
    positive = [row for row in rows if row["finish_gain"] > 0]
    for size in range(1, len(positive) + 1):
        for group in combinations(positive, size):
            cases = tuple(row["case"] for row in group)
            if any(audited[frozenset(pair)] >= DUPLICATE_THRESHOLD for pair in combinations(cases, 2)):
                continue
            total = fsum(row["finish_gain"] for row in group)
            if total > best_total:
                best_total, credited = total, cases
    loss = fsum(min(row["finish_gain"], 0) for row in rows)
    return dict(gain=round((best_total + loss) / panel_size, 12), credited=list(credited), panel_size=panel_size)


def exclude_duplicates(pool, selected, audit, *, limit, allow_partial=False):
    """Repair a rendered collision using only independent selection evidence."""
    removed = []
    while True:
        proof = audit(selected)
        collisions = [pair for pair in proof["pairs"] if pair["mean"] >= DUPLICATE_THRESHOLD]
        if not collisions:
            return selected, proof, removed
        most_similar = max(collisions, key=lambda pair: pair["mean"])
        alternatives = []
        for name in (most_similar["left"], most_similar["right"]):
            current = next(row for row in selected if row["case"] == name)
            remove = (name, current["method"])
            proposal = [[row for row in group if (row["case"], row["method"]) != remove] for group in pool]
            try:
                candidates = select(proposal, limit, allow_partial=allow_partial)
            except ValueError:
                continue
            alternatives.append((remove, proposal, candidates))
        if not alternatives:
            return selected, proof, removed
        remove, pool, selected = max(alternatives, key=lambda option: (len(option[2]), sum(row["usable"] for row in option[2])))
        removed.append(dict(case=remove[0], method=remove[1], collision=most_similar))
