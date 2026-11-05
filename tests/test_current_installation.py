"""Current installer identity; historical locks have separate environments."""
from importlib.metadata import distribution, version
import json
from pathlib import Path
import tomllib

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def test_current_installer_resolves_exact_public_core_and_pg095():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    dependency, = [Requirement(value) for value in project['dependencies']
                   if Requirement(value).name == 'particle-sliders-core']
    commit = dependency.url.split('@')[-1].split('#')[0]
    core = distribution('particle-sliders-core')
    origin = json.loads(core.read_text('direct_url.json'))
    assert origin['vcs_info']['commit_id'] == commit
    assert origin['subdirectory'] == 'packages/particle-sliders-core'
    assert core.version == '0.13.0'
    assert version('anima-particle-sliders') == project['version'] == '0.4.0'
    assert version('particlegan') == '0.9.5'
    from particle_sliders.pg09 import ConditionalPG09, PG_VERSION, PG_SOURCE_PROFILE
    from particle_sliders.pg095 import ConditionalPG095
    from lumen_studio.pg09_training import ConditionalPG09 as product_engine
    assert product_engine is ConditionalPG09 is ConditionalPG095
    assert (PG_VERSION, PG_SOURCE_PROFILE) == ('0.9.5', 'release-0.9.5')


def test_current_recursive_source_guard_and_old_profile_refusal():
    import pytest
    from lumen_studio.core_source import verify_core
    assert len(verify_core()) == 64
    from particle_sliders.pg094 import ConditionalPG094
    with pytest.raises(RuntimeError, match='pinned|0.9.4|rebind'):
        ConditionalPG094(None, None, None)


def test_current_receipt_tampering_refuses_before_runtime(tmp_path, monkeypatch):
    import pytest
    from lumen_studio import core_source, lifecycle
    receipt = json.loads(core_source.LOCK.read_text())
    receipt['pg_version'] = '0.9.4'
    path = tmp_path / 'wrong-source.json'
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(core_source, 'LOCK', path)
    with pytest.raises(ValueError, match='contradicts'):
        lifecycle.validate_runtime({})


def test_historical_pilots_refuse_new_engine_before_argument_or_model_work(monkeypatch):
    import importlib.util
    import pytest
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    for path in (ROOT / 'scripts').glob('anima_pg094*.py'):
        spec = importlib.util.spec_from_file_location('historical_' + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with pytest.raises(RuntimeError, match='Historical PG094'):
            module.main()
