"""Check the nominated recipe and incumbent on four different saved alpha-search scenes."""
import json
from hashlib import sha256

from design_range import measure, OUT, ROOT, harness
from finished_image import score_case

METHODS = ("shared-best-opening","late-cues")
CASES = ["archivist","bookkeeper","botanist","seamstress"]


def main():
    saved = (ROOT / "storage/alpha-standings.json").read_bytes()
    board = json.loads(saved)
    rows = []
    sources = {}
    for method in METHODS:
        winner = next(r for r in board["rows"] if r["best_for_recipe"] and r["method"]==method)
        path = ROOT / "strength-search/leaderboard-retune-visible" / method / "renders" / winner["profile"] / winner["alpha"].hex() / "search.json"
        cases = harness.read(path)["rows"]
        if set(CASES)!=set(cases) or any(score_case(r)!=r["final_boss_score"] for r in cases.values()):
            raise ValueError("Search-scene ratings differ from the finished-image score")
        rows.append(dict(winner,cases=cases))
        sources[method] = harness.pin(path)["sha256"]
    binding = sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    report = measure({"cases":CASES,"rows":rows},binding)
    report["protocol"] = ("Use the same repetition rubric and pair scoring rule on the nominated opening-four "
                          "recipe and the leading late-cue recipe at their fixed eight-case alpha. Compare six pairs "
                          "on four disjoint search scenes. These were used to select alpha and are diagnostic.")
    harness.write(OUT/"search-scene-check.json",dict(report,sources=sources))


if __name__=="__main__":
    main()
