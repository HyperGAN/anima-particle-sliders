import gzip
import json
from pathlib import Path

import pytest

from release_tools.final_boss_design_scores import summarize
from release_tools.verify_final_boss_designs import verify_panel


def test_repeated_design_shares_credit_and_weak_edits_cannot_buy_variety():
    pair = {"cases":["a","b"], "repeat": 0}
    scores = {"a":.8, "b":.6}
    assert summarize(scores,[pair])["score"] == pytest.approx(.7)
    assert summarize(scores,[dict(pair,repeat=1)])["score"] == pytest.approx(.4)
    for repetition in (0,.5,1):
        assert summarize(dict(scores,b=0),[dict(pair,repeat=repetition)])["score"] == pytest.approx(.4)
    with pytest.raises(ValueError):
        summarize(scores,[pair,pair])


def test_published_design_range_binds_its_pairs():
    folder=Path(__file__).resolve().parents[1]/"docs/final-boss-raw192/vibe12"
    report=json.loads((folder/"design-range.json").read_text())["development"]
    with gzip.open(folder/"design-receipts.json.gz","rt") as f:
        receipts=json.load(f)
    verify_panel(report,receipts)
    report["rows"][0]["pairs"][0]["images"].reverse()
    with pytest.raises(ValueError,match="Wrong source"):
        verify_panel(report,receipts)
