"""Bind the existing Anima comparison harness to the twelve-scene cohort."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
STORAGE = (ROOT / "storage").resolve(strict=True)
ORIGINAL = ROOT.parent / "anima-final-boss-shared-teacher-20261124"
PREVIOUS = ROOT.parent / "anima-final-boss-optimization-20261123"
SOURCE = ORIGINAL / "vibe-neutral-fix"
MANIFEST = (ORIGINAL / "storage/vibe-corrected-neutral-expanded/training" /
            "c2cb92f3c068a1bf20ae942f81ad73e4b8c18d60d769b05c049e78888e3b228d/manifest.json").resolve(strict=True)
CACHE = MANIFEST.parent / "targets"
sys.path.insert(0, str(PREVIOUS))


def run_guard():
    import shared_capacity_guard as captured

    stage, gpu, attempt, method = sys.argv[1:]
    captured.ROOT = ROOT
    captured.plan = lambda _: {}
    return captured.run(stage, int(gpu), attempt, (method,))
