"""Search for a high-finish portfolio under family caps and measured visual conflicts."""
from collections import Counter
from functools import lru_cache
from heapq import heappop, heappush
from itertools import combinations, count, product
from math import fsum
from statistics import fmean

from .vibe_portfolio import DUPLICATE_THRESHOLD, FAMILY_LIMIT, distinct_finish


def _relaxed(groups, limit, banned):
    families = sorted({row["observed_family"] for group in groups for row in group})
    key = {family: i for i, family in enumerate(families)}

    @lru_cache(maxsize=None)
    def choose(index, slots):
        if index == len(groups):
            return 0.0, ()
        best = choose(index + 1, slots)
        for row in groups[index]:
            slot = key[row["observed_family"]]
            if not slots[slot] or (row["case"], row["method"]) in banned:
                continue
            available = list(slots)
            available[slot] -= 1
            other, rows = choose(index + 1, tuple(available))
            value = row["finish_gain"] + other
            if value > best[0]:
                best = value, (row, *rows)
        return best

    return choose(0, (limit,) * len(families))


def _restricted_audit(proof, cases):
    pairs = [pair for pair in proof["pairs"] if pair["left"] in cases and pair["right"] in cases]
    return dict(pairs=pairs, average=fmean(pair["mean"] for pair in pairs) if pairs else None,
                near_duplicates=sum(pair["mean"] >= DUPLICATE_THRESHOLD for pair in pairs))


def optimize(groups, audit, *, size, trials=12, limit=FAMILY_LIMIT):
    """Test optimistic assignments and branch on their worst verified conflict.

    The relaxed score ignores pair conflicts. A conflict requires dropping one
    of its two candidates. The best complete portfolio and its remaining upper
    bound are reported even if the observation budget ends the search early.
    """
    if trials < 1 or size < len(groups) or size < 1:
        raise ValueError("Invalid portfolio size or pair comparison budget")
    groups = [[row for row in group if row["finish_gain"] > 0 and row["integrity"] == "pass"
               and row["observed_family"]] for group in groups]
    queue = []
    examined = set()
    serial = count()

    def enqueue(banned):
        if banned in examined:
            return
        examined.add(banned)
        bound, rows = _relaxed(groups, limit, banned)
        heappush(queue, (-bound, next(serial), banned, rows))

    enqueue(frozenset())
    gain = 0.0
    selected = []
    evidence = {"pairs": [], "average": None, "near_duplicates": 0}
    spent = 0
    trace = []
    while queue and spent < trials and -queue[0][0] > gain + 1e-10:
        rank, _, banned, proposal = heappop(queue)
        proof = audit(proposal)
        spent += 1
        measure = distinct_finish(proposal, proof, panel_size=size)
        credited = set(measure["credited"])
        subset = [row for row in proposal if row["case"] in credited]
        total = fsum(row["finish_gain"] for row in subset)
        trace.append(dict(relaxed=-rank/size, feasible=total/size,
                          near_duplicates=proof["near_duplicates"],
                          candidates=[[row["case"], row["method"]] for row in proposal]))
        if total > gain:
            gain, selected = total, subset
            evidence = _restricted_audit(proof, credited)
        collisions = [pair for pair in proof["pairs"] if pair["mean"] >= DUPLICATE_THRESHOLD]
        if not collisions:
            break
        conflict = max(collisions, key=lambda pair: pair["mean"])
        by_case = {row["case"]: row["method"] for row in proposal}
        for case in (conflict["left"], conflict["right"]):
            enqueue(banned | {(case, by_case[case])})
    upper = max(gain, -queue[0][0]) if queue else gain
    result = distinct_finish(selected, evidence, panel_size=size)
    return dict(selected=selected, audit=evidence, distinct_finish=result, trials=spent, trace=trace,
                upper_bound=round(upper/size, 12), certified=upper-gain <= 1e-10)


def improve(groups, incumbent, audit, *, size, trials=24, lookahead=2, limit=FAMILY_LIMIT):
    """Improve an audited portfolio by testing one or two coordinated replacements."""
    if trials < 1 or not 1 <= lookahead <= 2 or size < len(groups) or size < 1:
        raise ValueError("Invalid portfolio size, move size, or audit budget")

    def key(row):
        return row["case"], row["method"]

    candidates = {
        key(row): row for group in groups for row in group
        if row["observed_family"] and row["finish_gain"] > 0 and row["integrity"] == "pass"
    }
    by_case = {}
    for row in candidates.values():
        by_case.setdefault(row["case"], []).append(row)
    current = {row["case"]: candidates[key(row)] for row in incumbent if key(row) in candidates}
    conflicts = set()
    tested = {}
    trace = []

    def check(rows):
        slate = tuple(sorted(map(key, rows)))
        if slate in tested:
            return tested[slate]
        if any(conflict.issubset(slate) for conflict in conflicts):
            return None
        if len(trace) >= trials:
            return None
        proof = audit(rows)
        labels = {row["case"]: key(row) for row in rows}
        for pair in proof["pairs"]:
            if pair["mean"] >= DUPLICATE_THRESHOLD:
                conflicts.add(frozenset((labels[pair["left"]], labels[pair["right"]])))
        trace.append(dict(gain=fsum(row["finish_gain"] for row in rows)/size,
                          candidates=slate, near_duplicates=proof["near_duplicates"]))
        tested[slate] = proof
        return proof

    chosen = list(current.values())
    proof = check(chosen)
    initial = distinct_finish(chosen, proof, panel_size=size)
    kept = set(initial["credited"])
    current = {case: row for case, row in current.items() if case in kept}
    proof = _restricted_audit(proof, kept)
    while len(trace) < trials:
        before = fsum(row["finish_gain"] for row in current.values())
        neighbors = []
        for n in range(1, lookahead + 1):
            for cases in combinations(by_case, n):
                fixed = {case: row for case, row in current.items() if case not in cases}
                counts = Counter(row["observed_family"] for row in fixed.values())
                choices = [[row for row in by_case[case] if row != current.get(case)] for case in cases]
                for replacements in product(*choices):
                    labels = counts + Counter(row["observed_family"] for row in replacements)
                    if max(labels.values()) > limit:
                        continue
                    portfolio = [*fixed.values(), *replacements]
                    gain = fsum(row["finish_gain"] for row in portfolio)
                    if gain > before + 1e-10:
                        neighbors.append((gain, portfolio))
        neighbors.sort(key=lambda item: -item[0])
        for _, slate in neighbors:
            observed = check(slate)
            if observed and not observed["near_duplicates"]:
                current, proof = {row["case"]: row for row in slate}, observed
                break
            if len(trace) >= trials:
                break
        else:
            break
        if fsum(row["finish_gain"] for row in current.values()) <= before + 1e-10:
            break

    selected = list(current.values())
    upper = _relaxed(list(by_case.values()), limit, frozenset())[0] / size
    result = distinct_finish(selected, proof, panel_size=size)
    return dict(selected=selected, audit=proof, distinct_finish=result, trials=len(trace), trace=trace,
                upper_bound=round(upper, 12), certified=upper - result["gain"] <= 1e-10)


def best_of_search(groups, incumbent, evidence, audit, *, size, global_trials=12, local_trials=20):
    """Try both a relaxed search and audited swaps when the incumbent leaves room."""
    baseline = dict(selected=incumbent, audit=evidence,
                    distinct_finish=distinct_finish(incumbent, evidence, panel_size=size), trials=0)
    relaxed = optimize(groups, audit, size=size, trials=global_trials)
    methods = {"incumbent": baseline, "constrained-search": relaxed}
    best = max(methods, key=lambda name: methods[name]["distinct_finish"]["gain"])
    if relaxed["upper_bound"] > methods[best]["distinct_finish"]["gain"] + 1e-10:
        methods["audited-swaps"] = improve(groups, incumbent, audit, size=size, trials=local_trials)
        best = max(methods, key=lambda name: methods[name]["distinct_finish"]["gain"])
    return methods[best] | {
        "won_by": best,
        "alternatives": {name: result["distinct_finish"]["gain"] for name, result in methods.items()},
        "total_trials": sum(result["trials"] for result in methods.values()),
    }


def checked_search(groups, incumbent, evidence, audit, compare, *, size, global_trials=12, local_trials=20):
    """Screen the strongest replacements against their scene's incumbent image."""
    old = {row["case"]: row for row in incumbent}
    nominal = []
    unopposed = []
    for group in groups:
        valid = [row for row in group if row["integrity"] == "pass" and row["finish_gain"] > 0
                 and row["observed_family"]]
        if not valid:
            nominal.append([])
            continue
        case = valid[0]["case"]
        reference = old.get(case)
        if reference is None:
            nominal.append(valid)
            continue
        known = [row for row in valid if row["method"] == reference["method"]]
        rivals = sorted((row for row in valid if row["method"] != reference["method"]),
                        key=lambda row: -row["finish_gain"])
        # Preserve candidate-family choices under a bounded independent check.
        per_family = {}
        for row in rivals:
            per_family.setdefault(row["observed_family"], row)
        chosen = list(per_family.values())[:3]
        if len(chosen) < 3:
            chosen.extend(row for row in rivals if row not in chosen)
            chosen = chosen[:3]
        nominal.append(known + chosen)
        unopposed.extend((row, reference) for row in chosen)
    reviews = compare(unopposed)
    if len(reviews) != len(unopposed):
        raise ValueError("The independent comparison did not measure every replacement")
    rejected = {(row["case"], row["method"]) for (row, _), result in zip(unopposed, reviews, strict=True)
                if result["preferred"] < .5}
    groups = [[row for row in group if (row["case"], row["method"]) not in rejected] for group in nominal]
    result = best_of_search(groups, incumbent, evidence, audit, size=size,
                            global_trials=global_trials, local_trials=local_trials)
    return result | {"reviews": [
        dict(case=row["case"], method=row["method"], incumbent=reference["method"], **record)
        for (row, reference), record in zip(unopposed, reviews, strict=True)
    ]}


def guarded_search(groups, incumbent, evidence, audit, remeasure, *, size, near_trials=12, swap_trials=20):
    """Rank by original finish gain among portfolios that preserve a second effect read."""
    def key(row):
        return row["case"], row["method"]

    old = {row["case"]: row for row in incumbent}
    shortlist = []
    all_rows = {key(row): row for row in incumbent}
    for group in groups:
        distinct = {}
        for row in sorted(group, key=lambda item: -item["finish_gain"]):
            if row["finish_gain"] > 0 and row["integrity"] == "pass" and row["observed_family"]:
                distinct.setdefault(row["observed_family"], row)
        options = list(distinct.values())[:4]
        for row in options:
            all_rows[key(row)] = row
        reference = old.get(group[0]["case"]) if group else None
        if reference is not None and all(key(row) != key(reference) for row in options):
            options.append(reference)
        shortlist.append(options)

    checked = remeasure(list(all_rows.values()))
    if checked.keys() != all_rows.keys():
        raise ValueError("The independent observer did not measure the frozen shortlist")

    def passes(row):
        reference = old.get(row["case"])
        return checked[key(row)] >= (checked[key(reference)] if reference else 0) - 1e-12

    allowed = [[row for row in group if passes(row)] for group in shortlist]

    def second_score(rows, proof):
        other = [row | {"finish_gain": checked[key(row)]} for row in rows]
        return distinct_finish(other, proof, panel_size=size)["gain"]

    baseline = dict(selected=incumbent, audit=evidence,
                    distinct_finish=distinct_finish(incumbent, evidence, panel_size=size), trials=0)
    reference_score = second_score(incumbent, evidence)
    relaxed = optimize(allowed, audit, size=size, trials=near_trials)
    swaps = improve(allowed, incumbent, audit, size=size, trials=swap_trials)
    comparisons = {"incumbent": baseline, "constrained-search": relaxed, "audited-swaps": swaps}
    independent = {name: second_score(row["selected"], row["audit"]) for name, row in comparisons.items()}
    winner = max(
        (name for name in comparisons if independent[name] >= reference_score - 1e-12),
        key=lambda name: comparisons[name]["distinct_finish"]["gain"],
    )
    return comparisons[winner] | {
        "won_by": winner,
        "independent": independent,
        "alternatives": {name: result["distinct_finish"]["gain"] for name, result in comparisons.items()},
        "total_trials": sum(result["trials"] for result in comparisons.values()),
        "second_read_by_case": [
            dict(case=case, method=method, finish_gain=gain) for (case, method), gain in checked.items()
        ],
    }
