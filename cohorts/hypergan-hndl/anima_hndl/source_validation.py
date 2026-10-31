"""Exact installed artifact and complete source admission for this prerelease."""
import hashlib
from importlib.metadata import distribution, distributions
from importlib.util import find_spec
import json
from pathlib import Path
import sys

CORE_COMMIT = '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231'
PRODUCT_VERSION = '0.3.0a1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _files(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if path.is_symlink():
            raise ValueError('Anima HNDL source closure forbids symlinks')
        if path.is_file():
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _require_isolated_environment():
    prefix = Path(sys.prefix).resolve()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Anima HNDL prerelease requires a separate isolated virtual environment')
    config = prefix / 'pyvenv.cfg'
    if not config.is_file() or 'include-system-site-packages = false' not in config.read_text().lower():
        raise RuntimeError('Anima HNDL requires a venv without system site-packages')
    for entry in sys.path:
        if ('site-packages' in Path(entry).parts or 'dist-packages' in Path(entry).parts):
            if not Path(entry).resolve().is_relative_to(prefix):
                raise RuntimeError('Anima HNDL forbids shared dependency site-packages paths')
    products = [d for d in distributions()
                if d.metadata['Name'].lower().replace('_', '-') == 'anima-particle-sliders']
    if len(products) != 1 or products[0].version != PRODUCT_VERSION:
        raise RuntimeError('Anima HNDL requires only its explicit ' + PRODUCT_VERSION + ' prerelease')
    if not Path(products[0].locate_file('')).resolve().is_relative_to(prefix):
        raise RuntimeError('Anima HNDL product must be installed inside its isolated venv')
    for name in ('lumen_studio', 'anima_hndl', 'particle_sliders', 'torch', 'numpy',
                 'particlegan', 'hypergan', 'hndl'):
        spec = find_spec(name)
        if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(prefix):
            raise RuntimeError('Anima HNDL dependency must be installed in the isolated venv: ' + name)


def verify_cohort():
    """Refuse wrong Git/wheel provenance, changed files and extra source files."""
    _require_isolated_environment()
    manifest_path = Path(__file__).with_name('cohort.json')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['schema'], manifest['product_version'], manifest['core']['commit']) != (
            'anima-hypergan-hndl-cohort-v2', PRODUCT_VERSION, CORE_COMMIT):
        raise ValueError('Anima HNDL cohort manifest contradicts the binding')
    core = distribution('particle-sliders-core')
    if core.version != manifest['core']['version']:
        raise ValueError('Anima HNDL requires the exact core0.12 candidate')
    provenance = json.loads(core.read_text('direct_url.json') or '{}')
    vcs = provenance.get('vcs_info', {})
    archive = provenance.get('archive_info', {}).get('hashes', {})
    exact_git = (provenance.get('url', '').removesuffix('.git') ==
                 'https://github.com/HyperGAN/particle-sliders'
                 and provenance.get('subdirectory') == 'packages/particle-sliders-core'
                 and vcs.get('vcs') == 'git' and vcs.get('commit_id') == CORE_COMMIT)
    exact_wheel = archive.get('sha256') in manifest['core']['qualified_wheel_sha256']
    if not (exact_git or exact_wheel):
        raise ValueError('Anima HNDL core Git/wheel provenance differs from the exact reviewed artifact')
    core_root = Path(core.locate_file('particle_sliders'))
    if not core_root.resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError('Anima HNDL core must be installed in the isolated venv')
    if _files(core_root) != manifest['core_files']:
        raise ValueError('Anima HNDL complete core source closure differs')
    site = Path(__file__).resolve().parents[1]
    product_files = {}
    for package in ('lumen_studio', 'configs/anima', 'anima_hndl'):
        for name, digest in _files(site / package).items():
            if package == 'anima_hndl' and name == 'cohort.json':
                continue
            product_files[package + '/' + name] = digest
    for name in ('particle_io.py', 'single_particle.py', 'comfy_particle.py'):
        path = site / name
        if path.is_symlink():
            raise ValueError('Anima HNDL product sources must be regular installed files')
        product_files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if product_files != manifest['product_files']:
        raise ValueError('Anima HNDL complete product source closure differs')
    return dict(core_commit=CORE_COMMIT, core_sources_sha256=_digest(manifest['core_files']),
                product_sources_sha256=_digest(product_files),
                cohort_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
