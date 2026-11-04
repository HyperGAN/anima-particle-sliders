"""The shipped Comfy archive imports without a checkout on Python's path."""
from pathlib import Path
import subprocess
import sys
import zipfile

import particle_sliders

from release_tools.build import package_plugin


def test_comfy_archive_contains_validation_and_imports_in_isolated_python(tmp_path):
    package_plugin(tmp_path)
    archive = tmp_path / 'comfyui/anima-particle-sliders.zip'
    with zipfile.ZipFile(archive) as file:
        assert 'anima-particle-sliders/lumen_studio/checkpoint_validation.py' in file.namelist()
        legacy = Path(__file__).resolve().parents[1] / 'requirements-legacy.txt'
        assert file.read('anima-particle-sliders/requirements.txt') == legacy.read_bytes()
        assert file.read('anima-particle-sliders/requirements-legacy.txt') == legacy.read_bytes()
        cohorts = legacy.with_name('INSTALL-COHORTS.md')
        assert file.read('anima-particle-sliders/INSTALL-COHORTS.md') == cohorts.read_bytes()
        assert b'a119ca1ecd3d5d6c437065839d22739b04f2f4d8' in legacy.read_bytes()
        file.extractall(tmp_path / 'extract')
    plugin = tmp_path / 'extract/anima-particle-sliders'
    shared_src = Path(particle_sliders.__file__).resolve().parents[1]
    code = (
        'import sys\nfrom pathlib import Path\n'
        f'sys.path[:0] = [{str(plugin)!r}, {str(shared_src)!r}]\n'
        'import comfy_particle, single_particle\n'
        'import lumen_studio.checkpoint_validation as validation\n'
        f'root = Path({str(plugin)!r}).resolve()\n'
        'for module in (comfy_particle, single_particle, validation):\n'
        '    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__\n'
        'import torch\nassert not torch.cuda.is_initialized()\n'
    )
    result = subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp_path,
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
