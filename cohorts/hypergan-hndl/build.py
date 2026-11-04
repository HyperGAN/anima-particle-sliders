"""Build the separate opt-in cohort without rewriting the default manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--format', choices=('wheel', 'sdist', 'both'), default='wheel')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='anima-hndl-build-') as directory:
        staged = Path(directory) / 'source'
        shutil.copytree(root, staged, ignore=shutil.ignore_patterns(
            '.git', '.venv*', 'build', 'dist', '*.egg-info', '__pycache__', '.pytest_cache'))
        shutil.copyfile(Path(__file__).with_name('pyproject.toml'), staged / 'pyproject.toml')
        # A candidate sdist already contains this staged top-level namespace.
        # Rebuild only from its canonical cohort source, never merge stale files.
        if (staged / 'anima_hndl').exists():
            shutil.rmtree(staged / 'anima_hndl')
        shutil.copytree(Path(__file__).with_name('anima_hndl'), staged / 'anima_hndl')
        # Canonical LR development must not silently retarget the already locked
        # HNDL cohort. Reconstruct its exact original product closure; changed
        # canonical files have byte-identical snapshots, and newly added files
        # never leak into this independent historical distribution.
        cohort = json.loads((staged / 'anima_hndl/cohort.json').read_text())
        for package in ('lumen_studio', 'configs/anima'):
            shutil.rmtree(staged / package)
        for name, expected in cohort['product_files'].items():
            snapshot = Path(__file__).with_name('frozen_product') / name
            source = snapshot if snapshot.is_file() else root / name
            if name.startswith('anima_hndl/'):
                source = Path(__file__).with_name('anima_hndl') / name.removeprefix('anima_hndl/')
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError('Historical HNDL source differs from its locked bytes: ' + name)
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        functions = ('wheel', 'sdist') if args.format == 'both' else (args.format,)
        code = 'from setuptools import build_meta\n' + ''.join(
            'build_meta.build_{}({!r})\n'.format(kind, str(destination)) for kind in functions)
        subprocess.run([sys.executable, '-I', '-c', code], cwd=staged, check=True)


if __name__ == '__main__':
    main()
