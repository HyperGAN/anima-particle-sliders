"""The proposed PG09 dependency must execute from its installed wheel alone."""
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_immutable_candidate_lock_matches_optional_requirements():
    lock = json.loads((ROOT / 'core.pg09.lock.json').read_text())
    requirements = (ROOT / 'requirements-pg09.txt').read_text()
    assert lock['status'] == 'PROPOSED_NOT_ADA_APPROVED_OR_MERGED'
    assert not lock['native_adoption']
    assert len(lock['commit']) == 40
    assert f"particle-sliders-core=={lock['version']} --hash=sha256:{lock['wheel_sha256']}" in requirements
    pg = lock['particlegan']
    assert f"particlegan=={pg['version']} --hash=sha256:{pg['wheel_sha256']}" in requirements
    assert version('particle-sliders-core') == lock['version']
    assert version('particlegan') == pg['version']


def test_package_origin_and_sources_without_shared_checkout(tmp_path):
    code = r'''
import hashlib
import importlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import sys
import sysconfig

product = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(product))
lock = json.loads((product / 'core.pg09.lock.json').read_text())
site = Path(sysconfig.get_paths()['purelib']).resolve()
for name, expected in lock['source_sha256'].items():
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(site), (name, str(path), str(site))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
import particlegan
assert Path(particlegan.__file__).resolve().is_relative_to(site)
assert version('particle-sliders-core') == lock['version']
assert version('particlegan') == lock['particlegan']['version']
for package, expected in [('particle-sliders-core', lock['wheel_sha256']),
                          ('particlegan', lock['particlegan']['wheel_sha256'])]:
    direct = distribution(package).read_text('direct_url.json')
    if direct:
        archive = json.loads(direct)['archive_info']
    else:
        # Version+hash installs from a wheelhouse need not emit PEP610 data.
        report = json.loads((Path(sys.prefix) / 'pg09-install-report.json').read_text())
        records = [r for r in report['install']
                   if r['metadata']['name'].lower().replace('_', '-') == package]
        assert len(records) == 1, package
        archive = records[0]['download_info']['archive_info']
    assert archive['hashes']['sha256'] == expected, package
from particle_sliders.pg09 import ConditionalPG09, validate_pg09_domain
from particle_sliders.pg09_backends import AnimaRuntimeResponseAdapter
import lumen_studio.pg09_training as binding
assert binding.ConditionalPG09 is ConditionalPG09
assert binding.AnimaRuntimeResponseAdapter is AnimaRuntimeResponseAdapter
domain = validate_pg09_domain((1,4,1,2,2), response_domain='continuous_field')
assert domain['pg_source_profile'] == 'release'
assert domain['penalty_input_domain'] == 'auto'
assert domain['resolved_penalty_domain'] == 'image'
assert 'conceptmod' not in sys.modules
assert not any('particle-sliders-audit' in p or 'particle-sliders-core-pin' in p for p in sys.path)
import torch
assert not torch.cuda.is_initialized()
print(json.dumps(dict(package=version('particle-sliders-core'), pg=version('particlegan'),
    core=ConditionalPG09.__module__, adapter=AnimaRuntimeResponseAdapter.__module__,
    site=str(site), domain=domain, cuda_initialized=False), sort_keys=True))
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(ROOT)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt['cuda_initialized'] is False
    assert receipt['core'] == 'particle_sliders.pg09'
