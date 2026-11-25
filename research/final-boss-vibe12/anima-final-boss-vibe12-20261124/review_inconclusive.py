"""Apply one calibrated follow-up to inconclusive checks on the saved images."""

from concurrent.futures import ThreadPoolExecutor
import fcntl
from pathlib import Path
from time import sleep

from control import PREVIOUS, ROOT
from optimization_worker import PRIOR, pin, read, write


STATE = ("Review the candidate as a single finished illustration. Examine the visible figure, especially "
         "the arrangement of eyes, nose and mouth, the contour of the head, and where body parts connect. "
         "Flag a clearly visible structural drawing error. Anime proportions, stylized shadows, clothing "
         "or ordinary facial asymmetry can form a coherent illustration.")
STATUS = ("fixed_strength", "matched_constant")


def classify(values):
    if values is None:
        return "unmeasured"
    if max(values.values()) >= .8:
        return "fail"
    return "pass" if max(values.values()) < .5 else "unmeasured"


def reviewers():
    import optimization_worker as harness

    harness.ROOT = ROOT
    import boss_review
    import image_breakage
    import tune
    boss_review.FOLDER = ROOT / "boss-character-review"
    image_breakage.CACHE = ROOT / "image-breakage-review/cases"
    tune.BREAKAGE_RULE = image_breakage.RULE = PREVIOUS / "image-breakage-rule.json"
    _, decide = tune.reviewers()
    from final_boss_outcome import DEFECTS

    questions = {name: {"type": "noul", "instructions": question} for name, question in DEFECTS.items()}

    def check(image):
        values = decide((image,), STATE, questions)
        return {"status": classify(values), "checks": values}

    fixtures = read(PRIOR / "jev-rubric-audit-v2/fixtures.json")
    calibration = {}
    for name, expected in (("clean", "pass"), ("broken", "fail"), ("split", "fail")):
        fixture = fixtures[name]["on"]
        calibration[name] = {"image": fixture["sha256"], "expected": expected, **check(fixture)}
    with (ROOT / ".recovery-calibration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write(ROOT / "recovery-calibration.json", {"state": STATE, "questions": questions, "cases": calibration})
    if any(row["status"] != row["expected"] for row in calibration.values()):
        raise RuntimeError("Follow-up rubric did not pass all frozen calibration images")
    return check, decide, boss_review, image_breakage, tune


def recover(path, condition, native, judge):
    check, decide, boss_review, image_breakage, tune = judge
    fixed = native[condition]
    screen = native["screen"][fixed["profile"]][str(fixed["alpha"])]

    def review(group, pool):
        if group["status"] != "unmeasured":
            return group
        cases = {}
        for name, case in group["cases"].items():
            initial = case["integrity"]
            if initial["status"] == "unmeasured":
                case = case | {"integrity": check(case["on"]), "initial_integrity": initial}
            cases[name] = case
        if any(case["integrity"]["status"] != "pass" or case["scores"] is None for case in cases.values()):
            return group | {"cases": cases, "status": tune.summarize(cases)["status"]}
        concepts = {name: pool.submit(boss_review.augment, case, decide) for name, case in cases.items()}
        quality = {name: pool.submit(image_breakage.assess, case["off"], case["on"], decide)
                   for name, case in cases.items()}
        measured = {name: job.result() for name, job in concepts.items()}
        return group | image_breakage.with_readability(tune.summarize(measured),
                                                       {name: job.result() for name, job in quality.items()})

    with ThreadPoolExecutor(5) as pool:
        outcome = {"native_report": pin(path)["sha256"], "fixed": review(fixed, pool), "screen": review(screen, pool)}
    target = path.parent / "score-recovery" / (condition + ".json")
    write(target, outcome)
    print(path.parent.name, condition, outcome["fixed"]["status"], outcome["screen"]["status"], flush=True)


def main():
    judge = None
    while True:
        for path in sorted((ROOT / "tuning").glob("*/boss-effect-tuning.json")):
            native = read(path)
            for condition in STATUS:
                fixed = native[condition]
                screen = native["screen"][fixed["profile"]][str(fixed["alpha"])]
                destination = path.parent / "score-recovery" / (condition + ".json")
                if (not destination.is_file() and "unmeasured" in (fixed["status"], screen["status"])
                        and "fail" not in (fixed["status"], screen["status"])):
                    if judge is None:
                        judge = reviewers()
                    recover(path, condition, native, judge)
        sleep(30)


if __name__ == "__main__":
    main()
