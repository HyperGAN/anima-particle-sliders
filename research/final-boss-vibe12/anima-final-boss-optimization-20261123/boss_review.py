"""Evaluate visible endgame character design with the calibrated audit question."""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path

from optimization_worker import ROOT, pin, read, write
from score import DISTINCTION, ORIGINAL_DISTINCTION

STATE = "Image_1 is the original illustration. Image_2 is the edited illustration. Judge whether the edited person has become an unmistakable climactic video-game antagonist, from visible design of the character alone."
QUESTION = (
    "Score Final Boss distinction from 0 to 1. 0 means no discernible boss character. 0.25 means ordinary clothing, "
    "generic armor, or a stern face alone. 0.50 means a credible elite warrior or rival, with some boss cues. "
    "0.75 means a distinctive endgame antagonist with a recognizable, purposeful silhouette and signature character details. "
    "1.0 means a visually unmistakable final encounter character. Reward commanding presence built into the character design. "
    "Judge the outfit, silhouette, headpiece, insignia, and other visible evidence. Ignore scene, lighting, crop, and image polish."
)
QUESTIONS = {DISTINCTION: {"type": "noul", "instructions": QUESTION}}
RUBRIC = {"state": STATE, "questions": QUESTIONS}
FOLDER = ROOT / "boss-character-review" / "cases"


def receipt_path(off_sha, on_sha):
    return FOLDER / f"{off_sha}-{on_sha}.json"


def saved_distinction(off_sha, on_sha):
    path = receipt_path(off_sha, on_sha)
    if not path.is_file():
        return None
    row = read(path)
    if row["rubric"] != RUBRIC or row["off_sha"] != off_sha or row["on_sha"] != on_sha:
        raise ValueError("Boss distinction receipt belongs to a different comparison")
    score = row["value"]
    if score is not None and not 0 <= score <= 1:
        raise ValueError("Boss distinction score is out of range")
    return score


def assess(off, on, decide):
    hashes = (off["sha256"], on["sha256"])
    path = receipt_path(*hashes)
    FOLDER.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return saved_distinction(*hashes)
        reply = decide((off, on), STATE, QUESTIONS)
        value = None if reply is None else reply[DISTINCTION]
        write(path, {"rubric": RUBRIC, "off_sha": hashes[0], "on_sha": hashes[1], "value": value})
        return value


def augment(case, decide, *, use_index=True, include_original=True):
    if case["scores"] is None:
        return case

    def checked(off, on):
        if use_index:
            return assess(off, on, decide)
        reply = decide((off, on), STATE, QUESTIONS)
        value = None if reply is None else reply[DISTINCTION]
        if value != saved_distinction(off["sha256"], on["sha256"]):
            raise ValueError("Boss distinction index differs from the original Jev receipt")
        return value

    off = case["off"]
    scores = case["scores"] | {DISTINCTION: checked(off, case["on"])}
    if include_original:
        scores[ORIGINAL_DISTINCTION] = checked(off, off)
    return case | {"scores": scores}


def rescore(board_path):
    from publish_optimization import PUBLIC
    from tune import reviewers

    board = read(board_path)
    _, decide = reviewers()
    jobs = {}
    for name, row in board["rows"].items():
        for case, image in row["images"].items():
            off = pin(PUBLIC / f"images/{case}-signal-study-off.png")
            on = pin(PUBLIC / image)
            jobs.setdefault((off["sha256"], on["sha256"]), (off, on))
            jobs.setdefault((off["sha256"], off["sha256"]), (off, off))
    with ThreadPoolExecutor(max_workers=6) as pool:
        values = list(pool.map(lambda pair: assess(*pair, decide), jobs.values()))
    print(f"Boss distinction measured for {sum(value is not None for value in values)}/{len(values)} matched renders", flush=True)


if __name__ == "__main__":
    rescore(ROOT / "leaderboard.json")
