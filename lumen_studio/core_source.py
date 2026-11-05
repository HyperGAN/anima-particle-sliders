"""Admit the new PG0.9.5 cohort before runtime construction or attachment."""
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path

LOCK = Path(__file__).with_name('core095-source.json')
COMMIT = '3dd672a7f9b8c169daef90fd63e4bb42b9ddc423'


def verify_core():
    import particle_sliders
    from particle_sliders.pg095 import PG_VERSION, PG_SOURCE_PROFILE, PG_SOURCE_COMMIT, PG_WHEEL_SHA256
    from particle_sliders.pg09 import _released_sources
    receipt = json.loads(LOCK.read_text())
    if (receipt['schema'], receipt['source_commit'], receipt['version'],
            receipt['pg_version'], receipt['pg_source_profile']) != (
            'anima-pg095-core-closure-v1', COMMIT, '0.13.0', '0.9.5', 'release-0.9.5'):
        raise ValueError('Anima PG095 source receipt contradicts this cohort')
    core = distribution('particle-sliders-core')
    origin = json.loads(core.read_text('direct_url.json') or '{}')
    if (core.version != receipt['version'] or version('particlegan') != receipt['pg_version']
            or (PG_VERSION, PG_SOURCE_PROFILE) != ('0.9.5', 'release-0.9.5')
            or receipt.get('namespaces') != ['concept_slider_core', 'particle_sliders']
            or (receipt.get('pg_source_commit'), receipt.get('pg_wheel_sha256')) != (PG_SOURCE_COMMIT, PG_WHEEL_SHA256)
            or origin.get('url') != 'https://github.com/HyperGAN/particle-sliders.git'
            or origin.get('vcs_info', {}).get('commit_id') != COMMIT
            or origin.get('subdirectory') != 'packages/particle-sliders-core'):
        raise ValueError('Anima requires the exact new core/PG095 installed source cohort')
    root = Path(core.locate_file('')).resolve()
    if Path(particle_sliders.__file__).resolve() != root / 'particle_sliders/__init__.py':
        raise ValueError('Anima shared-core import is shadowed')
    actual = {}
    for namespace in receipt['namespaces']:
        package = root / namespace
        if package.is_symlink() or not package.is_dir():
            raise ValueError('Anima core source closure requires real package directories')
        for path in package.rglob('*'):
            if '__pycache__' in path.parts:
                continue
            if path.is_symlink():
                raise ValueError('Anima core source closure rejects symlinks')
            if path.is_file():
                actual[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != receipt['source_sha256']:
        raise ValueError('Anima recursive core source/data closure differs')
    _released_sources(profile=PG_SOURCE_PROFILE)
    return hashlib.sha256(LOCK.read_bytes()).hexdigest()
