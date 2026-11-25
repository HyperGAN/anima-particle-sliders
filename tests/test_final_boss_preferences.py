import gzip
import json
from pathlib import Path

import pytest

from release_tools.verify_final_boss_preferences import verify


ROOT = Path(__file__).resolve().parents[1] / "docs/final-boss-raw192/vibe12"


def observations(key):
    with gzip.open(ROOT / "preference-receipts.json.gz", "rt") as stream:
        receipts = json.load(stream)
    return json.loads((ROOT / "preferences.json").read_text())[key], receipts


def test_jev_decisions_replay_from_both_image_orders():
    report, receipts = observations("finished-edit")
    result = verify(report, receipts)
    assert result["margin_agreement"] == 8
    assert result["v2_margin_agreement"] == 5
    report["comparisons"][0]["cases"]["ranger"]["images"].reverse()
    with pytest.raises(ValueError, match="image order"):
        verify(report, receipts)


def test_round_robin_requires_every_matchup():
    report, receipts = observations("finalists")
    assert verify(report, receipts)["presentations"] == 240
    report["comparisons"].pop()
    with pytest.raises(ValueError, match="omitted"):
        verify(report, receipts)
