"""Bounded coordinate search for absolute G, D and particle rates on a frozen training panel."""

from math import isfinite, ldexp, log2, sqrt

DEFAULT_ARM_BUDGET = 31
ROLES = ("g", "d", "particle")


def neighbors(reference, center, stride):
    # All candidates lie on one log2 lattice. Approaching the same point
    # from another axis must recover the exact float and checkpoint identity.
    for role in ROLES:
        index = round(2 * (log2(center[role]) - log2(reference[role])))
        for offset in (-stride, stride):
            power, half = divmod(index + offset, 2)
            try:
                value = ldexp(reference[role], power) * (sqrt(2) if half else 1)
            except OverflowError:
                continue
            if value > 0 and isfinite(value):
                yield center | {role: value}


def explore(reference, initial, evaluate, save, budget=DEFAULT_ARM_BUDGET):
    """Yield strongest eligible screens; allow a lower-G rescue if all offered fits fail."""
    if type(budget) is not int or budget < 7:
        raise ValueError(
            "The rate search needs at least the seven reference and axis probes"
        )
    rows = {}
    offered = set()

    def key(rates):
        return tuple(rates[k] for k in ROLES)

    def probe(rates):
        k = key(rates)
        if (
            all(isfinite(v) and v > 0 for v in k)
            and k not in rows
            and len(rows) < budget
        ):
            rows[k] = evaluate(rates)

    def eligible():
        # Stable sorting leaves exact ties in their original observation order.
        return sorted(
            (
                r
                for r in rows.values()
                if r["status"] == "pass"
                and r.get("objective_score") is not None
                and key(r["rates"]) not in offered
            ),
            key=lambda r: -r["objective_score"],
        )

    for seeds in (
        initial,
        [reference | {"g": reference["g"] / n} for n in (4, 8)],
        [reference | {"g": reference["g"] / n} for n in (16, 32)],
    ):
        for candidate in seeds:
            probe(candidate)
        ranked = eligible()
        if not ranked:
            continue
        center = ranked[0]
        for stride, factor in ((2, 2.0), (1, sqrt(2.0))):
            while True:
                candidates = list(neighbors(reference, center["rates"], stride))
                for candidate in candidates:
                    probe(candidate)
                missing = [r for r in candidates if key(r) not in rows]
                unavailable = [
                    r
                    for r in candidates
                    if key(r) in rows
                    and (
                        rows[key(r)]["status"] == "unmeasured"
                        or (
                            rows[key(r)]["status"] == "pass"
                            and rows[key(r)].get("objective_score") is None
                        )
                    )
                ]
                best = eligible()[0]
                if best["objective_score"] > center["objective_score"]:
                    center = best
                    continue
                status = (
                    "arm-budget"
                    if missing
                    else "unmeasured-neighbor"
                    if unavailable
                    else "neighborhood-tested"
                )
                save(status, center, factor, missing, unavailable, len(rows))
                break
            if missing:
                break
        for row in eligible():
            offered.add(key(row["rates"]))
            yield row["rates"]
        if len(rows) >= budget:
            return
    if not offered:
        save("no-qualified-rate", None, None, [], [], len(rows))
