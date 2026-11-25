"""Measure recurrence of a boss design across the eight matched characters."""
import json
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from statistics import fmean
from time import sleep
from datetime import datetime, timezone
from design_scores import summarize

from review import HERE, ROOT, judge, harness

OUT = HERE / "design-range"
STATE = "Compare the central character designs in these two illustrations from the same visual style."
QUESTIONS = {"repeat": {"type":"score", "instructions": (
    "How much is the same underlying boss design repeated across the characters? Consider the silhouette, body "
    "construction, head or mask, costume shapes and signature visual idea together. Different people can be put "
    "in essentially the same boss design. Distinct bosses can both wear armor: look for individual design ideas. "
    "Differences in setting, lighting, hue, camera or pose alone say little about the design. Ignore beauty and "
    "rendering damage; those are judged separately."),
    "criteria":["Clearly different character designs with different signature forms",
                "Mostly different designs with a few shared motifs",
                "Related designs with substantial overlap and individual differences",
                "Mostly the same boss template with minor design differences",
                "Effectively the same character design or a repeat of the same illustration"]}}
PROTOCOL = ("Fix one RAW192 setting per recipe, selected by the existing finished-image score. Compare every "
            "unordered pair of its eight different source characters in both image orders. For each pair, sum "
            "the existing case scores and deduct the order-averaged repetition times the lower score. Average "
            "over all pairs and divide by two. Identical designs share the stronger score; distinct designs "
            "keep both. All 28 pairs receive equal weight. Validate on preselected older confirmation renders "
            "before ranking. For each same-design control both orders must exceed 0.5 and for each different-"
            "design control both must fall below 0.5. Report counts and raw values.")


def controls():
    archive = ROOT.parent / "anima-final-boss-optimization-20261123/confirmation/renders"
    constant = archive / "prompt-only-constant-0x1.54ccccccccccdp+0"
    early = archive / "prompt-only-early-0x1.3333333333333p+0"
    result = []
    for c in ("baker","mail-carrier","radio-engineer"):
        result.append((True,c,c,harness.pin(constant/c/"image.png"),harness.pin(early/c/"image.png")))
    for a,b in (("mail-carrier","muralist"),("bowler","printmaker"),("radio-engineer","pool-attendant")):
        result.append((False,a,b,harness.pin(constant/a/"image.png"),harness.pin(constant/b/"image.png")))
    return result


def pair_observer():
    judge()
    from particle_sliders.alpha_tune_jev import JevJudge, _Image, _digest, JEV_MODEL, JEV_URL
    from particle_sliders.candidate_failure import FileRecord
    client = JevJudge(OUT / "receipts", "Recurrence of boss character design")
    @lru_cache(maxsize=128)
    def image(path, sha):
        return _Image.read(FileRecord(path,Path(path).stat().st_size,sha))

    def decide(x,y):
        imgs = tuple(image(v["path"],v["sha256"]) for v in (x,y))
        request = dict(endpoint=JEV_URL,model=JEV_MODEL,state=STATE,questions=QUESTIONS,
                       images=[v.identity() for v in imgs])
        reply = client._decide(imgs,STATE,QUESTIONS)
        key = _digest(request)
        for attempt in (1,2):
            if reply is not None:
                break
            receipt_dir = OUT / ("receipts" if attempt == 1 else "receipts-retry-1")
            terminal = receipt_dir / (key+".json")
            if not terminal.is_file():
                break
            evidence = harness.read(terminal)
            if evidence["status"] != "http_error" or evidence["response"].get("http_status") not in (429,502,503,504):
                break
            retry = JevJudge(OUT / f"receipts-retry-{attempt}", "Recurrence of boss character design")
            reply = retry._decide(imgs,STATE,QUESTIONS)
        if reply is None:
            raise RuntimeError("Unmeasured design comparison " + _digest(request))
        return {"value":reply["repeat"],"receipt":_digest(request)}

    def pair(x,y):
        ab,ba = decide(x,y),decide(y,x)
        return {"images":[x["sha256"],y["sha256"]],"forward":ab,"reverse":ba,
                "repeat":fmean((ab["value"],ba["value"]))}

    return pair


def measure(board, signature):
    pair = pair_observer()
    from particle_sliders.alpha_tune_jev import JEV_MODEL
    with ThreadPoolExecutor(max_workers=12) as pool:
        fixtures = controls()
        trials = list(pool.map(lambda row:pair(row[3],row[4]),fixtures))
        anchored = [dict(r, expected_same=same, cases=[a,b]) for (same,a,b,_,_),r in zip(fixtures,trials)]
        harness.write(OUT / "controls.json",anchored)
        for r in anchored:
            print("control",r["cases"],r["expected_same"],r["forward"]["value"],r["reverse"]["value"],flush=True)
        if any(not all((r[k]["value"]>.5 if r["expected_same"] else r[k]["value"]<.5)
                       for k in ("forward","reverse")) for r in anchored):
            raise ValueError("The design question failed its transfer controls")
        jobs = []
        rows = [r for r in board["rows"] if r["best_for_recipe"]]
        names = board["cases"]
        originals = {c:rows[0]["cases"][c]["off"]["sha256"] for c in names}
        for row in rows:
            cases = row["cases"]
            if set(cases)!=set(names) or any(cases[c]["off"]["sha256"]!=originals[c] for c in names):
                raise ValueError("Mismatched original")
            pairs = [(a,b,pool.submit(pair,cases[a]["on"],cases[b]["on"])) for a,b in combinations(names,2)]
            jobs.append((row,pairs))
        results = []
        for row,pairs in jobs:
            scores = {c:row["cases"][c]["final_boss_score"] for c in names}
            values = [dict(f.result(),cases=[a,b]) for a,b,f in pairs]
            summary = {"method":row["method"], "profile":row["profile"], "alpha":row["alpha"],
                       "case_scores":scores, "inputs":{c:row["cases"][c]["on"]["sha256"] for c in names},
                       "pairs":values, **summarize(scores, values)}
            results.append(summary)
            print(row["method"],round(100*summary["score"],2),round(100*summary["repeat"],1),flush=True)
    return {"protocol":PROTOCOL,"model":JEV_MODEL,"state":STATE,"questions":QUESTIONS,"signature":signature,
            "originals":originals,"cases":names,"controls":anchored,"rows":sorted(results,key=lambda r:(-r["score"],r["method"]))}


def refresh():
    source = (ROOT / "storage/alpha-standings.json").read_bytes()
    board = json.loads(source)
    if board["scoring_version"]!="final-boss-visible-damage-v2":
        raise ValueError("Unexpected all-recipe base score")
    fields = [(r["method"],r["profile"],r["alpha"],[(c,r["cases"][c]["off"]["sha256"],r["cases"][c]["on"]["sha256"],
               r["cases"][c]["final_boss_score"]) for c in board["cases"]]) for r in board["rows"] if r["best_for_recipe"]]
    signature = sha256(json.dumps([fields,STATE,QUESTIONS,PROTOCOL],sort_keys=True).encode()).hexdigest()
    dest = OUT / signature / "results.json"
    if not dest.is_file():
        report = dict(measure(board,signature),board_sha256=sha256(source).hexdigest())
        harness.write(dest,report)
    latest = OUT / "latest.json"
    if not latest.is_file() or harness.read(latest)["signature"] != signature:
        report = harness.read(dest)
        for r in report["rows"]:
            checks = summarize(r["case_scores"], r["pairs"])
            if any(abs(r[k]-v) > 1e-12 for k,v in checks.items()):
                raise ValueError("Design score does not match its measured case pairs")
        harness.write(latest,dict(report,as_of=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")))
    return dest


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--watch",action="store_true")
    args = parser.parse_args()
    while True:
        print(refresh(),flush=True)
        if not args.watch:
            break
        sleep(60)
