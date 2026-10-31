"""Build the separate opt-in cohort without rewriting the default manifest."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='anima-hndl-build-') as directory:
        staged = Path(directory) / 'source'
        shutil.copytree(root, staged, ignore=shutil.ignore_patterns(
            '.git', '.venv*', 'build', 'dist', '*.egg-info', '__pycache__', '.pytest_cache'))
        shutil.copyfile(Path(__file__).with_name('pyproject.toml'), staged / 'pyproject.toml')
        code = 'from setuptools.build_meta import build_wheel; build_wheel({!r})'.format(str(destination))
        subprocess.run([sys.executable, '-I', '-c', code], cwd=staged, check=True)


if __name__ == '__main__':
    main()
