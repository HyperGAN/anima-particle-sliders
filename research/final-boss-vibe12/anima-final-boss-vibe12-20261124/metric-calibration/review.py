"""Bind the Jev question to each finished image and keep durable, versioned receipts."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from math import isfinite
import json
from pathlib import Path
import sys
import fcntl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from control import ROOT
import optimization_worker as harness
from finished_image import STATE, QUESTIONS, VERSION, damage
from hashlib import sha256


@lru_cache(maxsize=1)
def judge():
    harness.ROOT = ROOT
    harness.attach_judge()
    from particle_sliders.alpha_tune_jev import JevJudge
    return JevJudge(HERE / 'jev-receipts', 'Visible rendering damage')


BINDING = sha256(json.dumps({"state": STATE, "questions": QUESTIONS, "version": VERSION,
                             "model": "typesafe/jev-1.14"}, sort_keys=True).encode()).hexdigest()
RATINGS = HERE / "ratings" / BINDING


def saved(sha):
    path = RATINGS / f"{sha}.json"
    if path.exists():
        evidence = harness.read(path)
        if evidence["image"] != sha or evidence["binding"] != BINDING or evidence["severity"] != damage(evidence["answers"]):
            raise ValueError("Finish-damage receipt mismatch")
        return evidence["severity"]
    return None


def assess(image):
    sha = image["sha256"]
    RATINGS.mkdir(parents=True, exist_ok=True)
    with (RATINGS / f"{sha}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = saved(sha)
        if previous is not None:
            return previous
        client = judge()
        from particle_sliders.alpha_tune_jev import _Image
        from particle_sliders.candidate_failure import FileRecord
        path = Path(image["path"])
        asset = _Image.read(FileRecord(str(path), path.stat().st_size, sha))
        reply = client._decide((asset,), STATE, QUESTIONS)
        if reply is None:
            return None
        result = damage(reply)
        if not isfinite(result) or not 0 <= result <= 1:
            raise ValueError("Invalid damage")
        harness.write(RATINGS / f"{sha}.json", {"image":sha, "binding":BINDING, "answers":reply, "severity":result})
        return result


def main():
    judge()
    fixtures = harness.read(harness.PRIOR / 'jev-rubric-audit-v2/fixtures.json')
    historical = harness.read(harness.ROOT.parent / 'anima-final-boss-optimization-20261123/image-breakage-audit/report.json')
    images = {k:v['on'] for k,v in fixtures.items()}
    for r,cases in historical['observations'].items():
        for c,v in cases.items():
            images[f'historical-{r}-{c}'] = v['inputs']['on']
    with ThreadPoolExecutor(6) as jobs:
        values=dict(zip(images,jobs.map(assess,images.values())))
    print(values,flush=True)
    harness.write(HERE / 'audit.json',{'version':VERSION,'binding':BINDING,'images':{k:v['sha256'] for k,v in images.items()},'damage':values})
    if None in values.values() or not (values['clean'] < .35 and values['broken'] > .65):
        raise ValueError('Clean and damaged anchors did not separate')


if __name__ == '__main__':
    main()
