"""Replay the published blind-finalist comparisons and human-panel agreement offline."""
import argparse
import gzip
import json
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
from math import isclose
from pathlib import Path
from statistics import fmean


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify(report, receipts):
    total = 0
    by_group = defaultdict(dict)
    for pair in report["comparisons"]:
        cases = pair["cases"]
        a, b = pair["a"], pair["b"]
        group = by_group[pair["group"]]
        if a == b or (a,b) in group or (b,a) in group:
            raise ValueError("Duplicate comparison")
        group[a,b] = pair
        for case in cases.values():
            source, left, right = case["images"]
            for direction, images in (("forward", [source,left,right]), ("reverse", [source,right,left])):
                proof = case[direction]
                receipt = receipts[proof["receipt"]]
                req = receipt["request"]
                if (digest(req) != proof["receipt"] or req["state"] != report["state"]
                        or req["questions"] != report["questions"] or req["model"] != report["model"]
                        or [img["sha256"] for img in req["images"]] != images or receipt["status"] != "ok"
                        or digest(receipt["response"]) != receipt["response_sha256"]):
                    raise ValueError("Observation does not match its prompt and image order")
                scale = len(req["questions"]["preference"]["criteria"]) - 1
                value = receipt["response"]["answers"]["preference"]["score"] / scale
                if not 0 <= value <= 1 or value != proof["value"]:
                    raise ValueError("Decision does not match its receipt")
                total += 1
            x,y = case["forward"]["value"],case["reverse"]["value"]
            if not isclose(case["margin"], y-x) or case["both_orders_agree"] != ((x-.5)*(y-.5)<0):
                raise ValueError("Incorrect order-balanced margin")
        baseline = [c["v2_margin"] for c in cases.values()]
        if all(x is not None for x in baseline):
            if not isclose(pair["v2_margin"], fmean(baseline)):
                raise ValueError("Incorrect baseline panel mean")
        elif pair["v2_margin"] is not None:
            raise ValueError("Missing baseline measurement")
        if not isclose(pair["margin"], fmean(c["margin"] for c in cases.values())):
            raise ValueError("Incorrect panel mean")
    ranked = by_group["current"]
    methods = report["settings"].keys()
    if {frozenset(p) for p in ranked} != {frozenset(p) for p in combinations(methods,2)}:
        raise ValueError("Round robin omitted one or more finalist pairings")
    official = {"ranger","smith","stationer","ferryman","watchmaker","guide","guard","violinist"}
    if any(set(p["cases"]) != official for p in ranked.values()):
        raise ValueError("Incomplete matched development panel")
    points = {m:fmean(p["margin"] if a==m else -p["margin"] for (a,b),p in ranked.items() if m in (a,b))
              for m in methods}
    if "margins" in report and (set(report["margins"]) != set(points) or
            any(not isclose(points[m], report["margins"][m]) for m in methods) or
            report["order"] != sorted(points, key=lambda m:(-points[m],m))):
        raise ValueError("Published standings differ from replay")
    summary = {"presentations":total, "preference_margins":points}
    if "historical-effect" in by_group:
        comparisons = [(by_group["historical-effect"], a,b) for a,b in combinations("4132",2)]
        comparisons.extend((by_group["historical-damage"], a,"1") for a in "23")
        for metric in ("margin","v2_margin"):
            summary[metric + "_agreement"] = sum((group[a,b][metric] if (a,b) in group else -group[b,a][metric]) > 0
                                                 for group,a,b in comparisons)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    with gzip.open(args.folder / "preference-receipts.json.gz","rt") as fp:
        receipts = json.load(fp)
    reports = json.loads((args.folder / "preferences.json").read_text())
    print(json.dumps({k:verify(r, receipts) for k,r in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
