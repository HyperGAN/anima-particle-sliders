"""Current installer identity; historical locks have separate environments."""
from importlib.metadata import distribution, version
import json
from pathlib import Path
import tomllib

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def test_current_installer_resolves_exact_public_core_and_pg094():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    dependency, = [Requirement(value) for value in project['dependencies']
                   if Requirement(value).name == 'particle-sliders-core']
    commit = dependency.url.split('@')[-1].split('#')[0]
    core = distribution('particle-sliders-core')
    origin = json.loads(core.read_text('direct_url.json'))
    assert origin['vcs_info']['commit_id'] == commit
    assert origin['subdirectory'] == 'packages/particle-sliders-core'
    assert core.version == '0.12.0'
    assert version('anima-particle-sliders') == project['version'] == '0.3.0'
    assert version('particlegan') == '0.9.4'
    from particle_sliders.pg09 import ConditionalPG09, PG_VERSION, PG_SOURCE_PROFILE
    from particle_sliders.pg094 import ConditionalPG094
    from lumen_studio.pg09_training import ConditionalPG09 as product_engine
    assert product_engine is ConditionalPG09 is ConditionalPG094
    assert (PG_VERSION, PG_SOURCE_PROFILE) == ('0.9.4', 'release-0.9.4')
