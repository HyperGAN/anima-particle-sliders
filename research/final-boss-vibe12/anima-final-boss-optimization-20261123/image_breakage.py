"""Keep image breakage separate from the Final Boss effect score."""

from concurrent.futures import ThreadPoolExecutor
import fcntl
from functools import lru_cache
from statistics import fmean

from optimization_worker import ROOT, pin, read, write
from score import BREAKAGE, METRIC, adjusted_effect, case_effect


RULE = ROOT / "image-breakage-rule.json"
CACHE = ROOT / "image-breakage-review" / "cases"


@lru_cache
def rules():
    return read(RULE)


def receipt_path(off_sha, on_sha):
    return CACHE / f"{off_sha}-{on_sha}.json"


def saved(off_sha, on_sha):
    path = receipt_path(off_sha, on_sha)
    if not path.exists():
        return None
    receipt = read(path)
    inputs = {"off": off_sha, "on": on_sha, "rubric": rules()["rubric"]}
    value = receipt["severity"]
    if receipt["inputs"] != inputs or (value is not None and not 0 <= value <= 1):
        raise ValueError("Image breakage receipt does not match the comparison")
    return value


def assess(off, on, decide, *, offline=False):
    off_sha, on_sha = off["sha256"], on["sha256"]
    rubric = rules()["rubric"]
    metric = rules()["measurement"]

    def observe():
        result = decide((off, on), rubric["state"], rubric["questions"])
        return None if result is None else result[metric]

    path = receipt_path(off_sha, on_sha)
    if offline:
        value = observe()
        if not path.is_file() or saved(off_sha, on_sha) != value:
            raise ValueError("Image breakage differs from the original Jev receipt")
        return value
    CACHE.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            return saved(off_sha, on_sha)
        value = observe()
        write(path, {"inputs": {"off": off_sha, "on": on_sha, "rubric": rubric}, "severity": value})
        return value


def summarize(severities):
    config = rules()
    values = [value for value in severities.values() if value is not None]
    complete = len(values) == len(severities) and bool(values)
    mean = fmean(values) if complete else None
    repeated = sum(value >= config["minor_anchor"] for value in values)
    if any(value >= config["major_anchor"] for value in values):
        status, reason = "flagged", "major"
    elif complete and mean > config["minor_anchor"] and repeated * 2 >= len(values):
        status, reason = "flagged", "repeated"
    else:
        status, reason = ("pass" if complete else "unmeasured"), None
    return {"status": status, "reason": reason, "mean": mean, "severity": severities}


def with_readability(result, severities):
    review = summarize(severities)
    means = result["means"]
    complete = means is not None and review["mean"] is not None
    if means is not None:
        adjusted = (round(fmean(adjusted_effect(case_effect(row["scores"]), severities[case])
                                for case, row in result["cases"].items()), 12) if complete else None)
        means = means | {METRIC: adjusted, BREAKAGE: review["mean"]}
    status = result["status"] if complete or result["status"] == "fail" else "unmeasured"
    return result | {"readability": review, "status": status, "means": means}


def rescore():
    from publish_optimization import PUBLIC
    from score import EFFECT
    from tune import reviewers

    board = read(ROOT / "leaderboard.json")
    references = {case: pin(PUBLIC / f"images/{case}-signal-study-off.png") for case in board["cases"]}
    _, decide = reviewers()
    rows = sorted(board["rows"].values(), key=lambda row: (row["means"] or {}).get(EFFECT) or 0, reverse=True)
    jobs = {}
    for row in rows:
        for case, image in row["images"].items():
            off, on = references[case], pin(PUBLIC / image)
            jobs.setdefault((off["sha256"], on["sha256"]), (off, on))
    with ThreadPoolExecutor(max_workers=6) as pool:
        values = list(pool.map(lambda pair: assess(*pair, decide), jobs.values()))
    print(f"Image breakage measured for {sum(value is not None for value in values)}/{len(values)} matched renders",
          flush=True)


if __name__ == "__main__":
    rescore()
