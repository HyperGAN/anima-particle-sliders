import gzip
import json
from pathlib import Path

import pytest

from release_tools.final_boss_design_alpha import verify


def test_saved_search_drives_refinement_and_selection_with_distinct_design_scores():
    folder = (
        Path(__file__).resolve().parents[1]
        / "docs/final-boss-raw192/vibe12/alpha-tuning/shared-best-opening"
    )
    evidence = json.loads((folder / "search.json").read_text())
    with gzip.open(folder / "receipts.json.gz", "rt") as f:
        receipts = json.load(f)
    selected = verify(evidence, receipts)
    assert selected == json.loads((folder / "tuning.json").read_text())
    first = next(
        r
        for r in evidence["trials"]
        if r["phase"] == "search" and r["profile"] == "early" and r["alpha"] == 1.0
    )
    assert first["design"]["base"] != first["design"]["score"]
    evidence["paths"]["early"]["trials"][0]["score"] = first["design"]["base"]
    with pytest.raises(ValueError, match="Adaptive path"):
        verify(evidence, receipts)


def test_replay_rejects_a_tuning_with_different_native_gains():
    folder = (
        Path(__file__).resolve().parents[1]
        / "docs/final-boss-raw192/vibe12/alpha-tuning/shared-best-opening"
    )
    evidence = json.loads((folder / "search.json").read_text())
    with gzip.open(folder / "receipts.json.gz", "rt") as f:
        receipts = json.load(f)
    evidence["selected"]["effective_strengths"][0] = 0
    with pytest.raises(ValueError, match="winning native trial"):
        verify(evidence, receipts)
