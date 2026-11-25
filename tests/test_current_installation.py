"""Current installer identity; historical locks have separate environments."""
from importlib.metadata import distribution, version
import json
from pathlib import Path
import tomllib

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def test_current_installer_resolves_exact_public_core_and_particlegan():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    dependency, = [Requirement(value) for value in project['dependencies']
                   if Requirement(value).name == 'particle-sliders-core']
    commit = dependency.url.split('@')[-1].split('#')[0]
    core = distribution('particle-sliders-core')
    origin = json.loads(core.read_text('direct_url.json'))
    assert origin['vcs_info']['commit_id'] == commit
    assert origin['subdirectory'] == 'packages/particle-sliders-core'
    assert core.version == '0.15.3'
    assert version('anima-particle-sliders') == project['version'] == '0.4.1'
    assert version('particlegan') == '0.12.0'
    from particle_sliders.engine import ConditionalParticleGAN, PG_VERSION, PG_SOURCE_PROFILE
    from lumen_studio.engine_training import ConditionalParticleGAN as product_engine
    assert product_engine is ConditionalParticleGAN
    assert (PG_VERSION, PG_SOURCE_PROFILE) == ('0.12.0', 'release-0.12.0')


def test_current_recursive_source_guard_and_old_profile_refusal():
    import pytest
    from lumen_studio.core_source import verify_core
    assert len(verify_core()) == 64
    from particle_sliders.engine import validate_engine_domain
    with pytest.raises(ValueError, match='source profile'):
        validate_engine_domain((1, 4, 1, 2, 2), response_domain='continuous_field',
                               pg_source_profile='release-99.0.0')



def test_current_receipt_tampering_refuses_before_runtime(tmp_path, monkeypatch):
    import pytest
    from lumen_studio import core_source, lifecycle
    receipt = json.loads(core_source.LOCK.read_text())
    receipt['pg_version'] = '99.0.0'
    path = tmp_path / 'wrong-source.json'
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(core_source, 'LOCK', path)
    with pytest.raises(ValueError, match='contradicts'):
        lifecycle.validate_runtime({})
