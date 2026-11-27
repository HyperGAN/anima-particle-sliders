from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path

path = Path(__file__).parents[1] / "research/vibe-regime/teacher_structure.py"
spec = importlib.util.spec_from_file_location("teacher_structure", path)
structure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(structure)


def test_resume_checks_skipped_candidates_and_retains_only_intact_choices():
    def option(name, **saved):
        return dict(positive=name, image={"sha256": name}, **saved)

    rows = [
        dict(scene="library", seed=11, neutral={"sha256":"off1"}, chosen="good",
             options=[option("good",baseline_structure="pass",candidate_structure="pass"), option("broken")]),
        dict(scene="station", seed=12, neutral={"sha256":"off2"}, chosen="bad",
             options=[option("bad"),option("unknown")]),
    ]
    calls = []
    def inspect(image):
        sha = image["sha256"]
        calls.append(sha)
        return {"off2":"pass","bad":"fail","broken":"fail","unknown":"unmeasured"}[sha]

    with ThreadPoolExecutor(2) as pool:
        structure.inspect_saved_rounds(rows, inspect, pool)
        structure.inspect_saved_rounds(rows, inspect, pool)
    assert sorted(calls) == ["bad","broken","off2","unknown"]
    saved = {"library":{"seed":11,"positive":"good"}, "station":{"seed":12,"positive":"bad"}}
    assert structure.restore_selections(rows, saved) == {"library":saved["library"]}
    assert rows[0]["options"][1]["candidate_structure"] == "fail"
    assert rows[1]["chosen"] is None and rows[1]["previous_choice"] == "bad"
