"""Archived release recipe CLI; native training requires the explicit PG09 API."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lumen_studio.training_policy import legacy_training_unavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variation', choices=['candlelit', 'moonlit', 'bad-intent', 'final-form', 'afterimage', 'dusk'])
    parser.add_argument('--root', type=Path, default=Path('artifacts/reproduction'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prepare-targets', action='store_true')
    parser.add_argument('--until', type=int, default=1600)
    parser.parse_args()
    legacy_training_unavailable("release training CLI")


if __name__ == '__main__':
    main()
