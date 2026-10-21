"""Archived target/training campaign; refuse before any target or worker work."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lumen_studio.training_policy import legacy_training_unavailable


class TargetRuntime:
    """Resolve the current frozen runtime after each rendering handoff."""
    def __init__(self, worker):
        self.worker = worker

    def __getattr__(self, name):
        return getattr(self.worker.load_runtime(), name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'artifacts/anima')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--variation', action='append', choices=('candlelit', 'moonlit', 'theatrical'))
    parser.parse_args()
    legacy_training_unavailable("legacy target/training campaign")


if __name__ == '__main__':
    main()
