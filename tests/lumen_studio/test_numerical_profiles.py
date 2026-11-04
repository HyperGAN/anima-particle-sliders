"""Numerical profile admission is explicit, process-bound and cache-bound."""
import json
from pathlib import Path
import subprocess
import sys
import pytest

from lumen_studio.pg09_training import CachedResponses
from lumen_studio.numerical_profiles import NAME,activate_numerical_profile,require_numerical_profile
from test_pg09_training import cache_fixture,runtime_fixture


def test_profile_activation_in_isolated_cpu_process_sets_strict_flags():
    root=str(Path(__file__).resolve().parents[2])
    code=f'''import sys,json
sys.path.insert(0,{root!r})
from lumen_studio.numerical_profiles import activate_numerical_profile,require_numerical_profile,NAME
p=activate_numerical_profile(NAME)
assert p==require_numerical_profile(NAME)
import torch
assert not torch.cuda.is_initialized()
torch.backends.cuda.enable_flash_sdp(True)
try:require_numerical_profile(NAME)
except ValueError:pass
else:raise AssertionError('changed numerical flags were accepted')
print(json.dumps(p))
'''
    result=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    manifest=json.loads(result.stdout)
    assert manifest['flags']['deterministic'] and not manifest['flags']['warn_only']
    assert len(manifest['implementation_sha256'])==64


def test_activation_rejected_after_cuda_initialization(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    with pytest.raises(RuntimeError,match='before CUDA'):
        activate_numerical_profile(NAME)


def test_unknown_profile_rejected():
    with pytest.raises(ValueError,match='unknown'):
        require_numerical_profile('unreviewed')


def test_profile_cache_requires_explicit_matching_contract(tmp_path):
    runtime=runtime_fixture()
    cache=cache_fixture(tmp_path/'cache',runtime)
    cache.index['identity']['numerical_profile']={'name':NAME}
    with pytest.raises(ValueError,match='explicit numerical_profile'):
        CachedResponses(cache,runtime,'finalform')
    with pytest.raises(ValueError,match='admitted CUDA'):
        CachedResponses(cache,runtime,'finalform',numerical_profile=NAME)
    runtime.mixer.close()
