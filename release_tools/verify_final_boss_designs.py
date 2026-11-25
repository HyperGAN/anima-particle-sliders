"""Reproduce the distinct-design rankings and check their Jev image bindings offline."""
import argparse
import gzip
import json
from math import isclose
from pathlib import Path
from statistics import fmean

from release_tools.final_boss_design_scores import summarize
from release_tools.final_boss_visible_damage import score_case
from release_tools.verify_final_boss_preferences import digest


def verify_orders(report, pair, image_hashes, receipts, question):
    for key, ordered in (("forward",image_hashes),("reverse",image_hashes[::-1])):
        observation = pair[key]
        proof = receipts[observation["receipt"]]
        binding = proof["request"]
        response = proof["response"]
        if (digest(binding) != observation["receipt"] or proof["status"] != "ok"
                or digest(response) != proof["response_sha256"] or binding["model"] != report["model"]
                or binding["state"] != report["state"] or binding["questions"] != report["questions"]
                or [im["sha256"] for im in binding["images"]] != ordered):
            raise ValueError("Jev receipt does not bind the two images in this order")
        scale = len(report["questions"][question]["criteria"])-1
        if observation["value"] != response["answers"][question]["score"]/scale:
            raise ValueError("Jev score differs from its response")


def verify_panel(report, receipts, snapshot=None):
    cases = report["cases"]
    if len(cases)!=len(set(cases)):
        raise ValueError("Repeated panel case")
    for check in report["controls"]:
        verify_orders(report,check,check["images"],receipts,"repeat")
        if not all(check[k]["value"]>.5 if check["expected_same"] else check[k]["value"]<.5 for k in ("forward","reverse")):
            raise ValueError("Same-design or different-design control failed")
    if snapshot:
        if snapshot["cases"]!=cases or snapshot["originals"]!=report["originals"]:
            raise ValueError("Development panel differs from alpha ranking")
        settings = {(s["method"],s["profile"],s["alpha"]):s for s in snapshot["settings"]}
    methods = [r["method"] for r in report["rows"]]
    if len(methods) != len(set(methods)) or (snapshot and set(methods) != set(snapshot["methods"])):
        raise ValueError("The panel must rank every trained recipe exactly once")
    for r in report["rows"]:
        if set(r["case_scores"])!=set(cases) or set(r["inputs"])!=set(cases):
            raise ValueError("Incomplete recipe")
        if snapshot:
            raw=settings[r["method"],r["profile"],r["alpha"]]
            if any(raw["cases"][c]["on_sha256"]!=r["inputs"][c] or score_case(raw["cases"][c])!=r["case_scores"][c] for c in cases):
                raise ValueError("Finish or base score differs from the saved measurement")
        for pair in r["pairs"]:
            hashes=[r["inputs"][c] for c in pair["cases"]]
            if hashes!=pair["images"]:
                raise ValueError("Wrong source for design pair")
            verify_orders(report,pair,hashes,receipts,"repeat")
            if pair["repeat"]!=fmean(pair[k]["value"] for k in ("forward","reverse")):
                raise ValueError("Incorrect order-balanced design rating")
        checks=summarize(r["case_scores"],r["pairs"])
        if any(not isclose(r[k],v,abs_tol=1e-12) for k,v in checks.items()):
            raise ValueError("Published score differs from replay")
    if report["rows"]!=sorted(report["rows"],key=lambda r:(-r["score"],r["method"])):
        raise ValueError("Published ranks differ from replay")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    folder=parser.parse_args().folder
    data=json.loads((folder/"design-range.json").read_text())
    with gzip.open(folder/"design-receipts.json.gz","rt") as f:
        receipts=json.load(f)
    board=json.loads((folder/"standings.json").read_text())
    verify_panel(data["development"],receipts,board)
    verify_panel(data["search_scenes"],receipts)
    lineup=data["lineups"]
    for row in lineup["groups"].values():
        verify_orders(lineup,row,row["sheet_images"],receipts,"preference")
        if not isclose(row["opening_margin"],row["reverse"]["value"]-row["forward"]["value"]):
            raise ValueError("Incorrect lineup margin")
    for rank,r in enumerate(data["development"]["rows"],1):
        print(f'{rank:2}. {r["method"]:<32} {100*r["score"]:6.2f}  repetition {100*r["repeat"]:5.1f}')


if __name__=="__main__":
    main()
