"""Source installs retain cohort evidence and can rebuild the historical plugin."""
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def test_sdist_preserves_installers_locks_and_release_inputs(tmp_path):
    source = tmp_path / 'source'
    shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
        '.git', '.venv*', 'build', 'dist', '*.egg-info', '__pycache__', '.pytest_cache'))
    archives = tmp_path / 'archives'
    code = 'from setuptools.build_meta import build_sdist; build_sdist({!r})'.format(str(archives))
    result = subprocess.run([sys.executable, '-I', '-c', code], cwd=source,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    archive, = archives.glob('*.tar.gz')
    extracted = tmp_path / 'extracted'
    with tarfile.open(archive) as file:
        file.extractall(extracted, filter='data')
    unpacked, = extracted.iterdir()
    # A source archive must retain the actual installers and immutable evidence,
    # rather than only a README pointing to files absent from the archive.
    for name in ('requirements.txt', 'requirements-legacy.txt', 'requirements-pg09.txt',
                 'core.lock.json', 'core.pg09.lock.json', 'INSTALL-COHORTS.md',
                 'CORE081-CPU.md', 'PUBLICATION-STATUS.md', 'pytest.ini', '__init__.py',
                 'release_tools/build.py', 'data/release-samples.json',
                 'cohorts/hypergan-hndl/build.py', 'cohorts/hypergan-hndl/pyproject.toml',
                 'tests/lumen_studio/test_image_lifecycle081.py'):
        assert (unpacked / name).read_bytes() == (ROOT / name).read_bytes(), name
    requirements = (unpacked / 'requirements.txt').read_text().splitlines()
    assert [line for line in requirements if line and not line.startswith('#')] == ['.']
    # Build a wheel from only the extracted sdist and exercise its release helper.
    wheels = tmp_path / 'wheels'
    output = tmp_path / 'plugin'
    output.mkdir()
    code = (
        'import sys\n'
        'from setuptools.build_meta import build_wheel\n'
        f'build_wheel({str(wheels)!r})\n'
        'sys.path.insert(0, ".")\n'
        'from pathlib import Path\n'
        'from release_tools.build import package_plugin\n'
        f'package_plugin(Path({str(output)!r}))\n'
    )
    result = subprocess.run([sys.executable, '-I', '-c', code], cwd=unpacked,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = wheels.glob('*.whl')
    with zipfile.ZipFile(wheel) as file:
        assert 'lumen_studio/lifecycle.py' in file.namelist()
        assert 'configs/anima/released-identity.json' in file.namelist()
    with zipfile.ZipFile(output / 'comfyui/anima-particle-sliders.zip') as file:
        legacy = (ROOT / 'requirements-legacy.txt').read_bytes()
        for name in ('requirements.txt', 'requirements-legacy.txt'):
            assert file.read('anima-particle-sliders/' + name) == legacy
    # The independent build manifest must also work from only the sdist.
    candidate_wheels = tmp_path / 'candidate-wheels'
    result = subprocess.run([sys.executable, '-I', 'cohorts/hypergan-hndl/build.py',
                             '--out', str(candidate_wheels)], cwd=unpacked,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    candidate_wheel, = candidate_wheels.glob('*.whl')
    with zipfile.ZipFile(candidate_wheel) as file:
        metadata = file.read('anima_particle_sliders-0.3.0.dist-info/METADATA').decode()
        entrypoints = file.read('anima_particle_sliders-0.3.0.dist-info/entry_points.txt').decode()
        dependencies = [Requirement(line.removeprefix('Requires-Dist: '))
                        for line in metadata.splitlines() if line.startswith('Requires-Dist: ')]
        core, = [dep for dep in dependencies if dep.name == 'particle-sliders-core']
        assert core.extras == {'hypergan-hndl'}
        assert core.url == ('git+https://github.com/HyperGAN/particle-sliders.git@'
                            '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231#subdirectory=packages/particle-sliders-core')
        assert 'anima-hndl = lumen_studio.hndl_lifecycle:BACKEND' in entrypoints
        assert '\nanima = ' not in entrypoints
    # Building the new profile never retargets the default source manifest.
    assert (unpacked / 'pyproject.toml').read_bytes() == (ROOT / 'pyproject.toml').read_bytes()


def test_candidate_native_dependencies_preserve_the_current_model_cohort():
    default = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    candidate = tomllib.loads((ROOT / 'cohorts/hypergan-hndl/pyproject.toml').read_text())
    assert candidate['project']['optional-dependencies']['native'] == default['project']['optional-dependencies']['native']
    assert '32f6830f6402e85aa77b6631db64fcbbb2131f6b' in default['project']['dependencies'][-1]
    assert candidate['project']['version'] == '0.3.0'
