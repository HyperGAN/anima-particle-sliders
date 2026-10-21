import hashlib, importlib.metadata, json, pathlib, sys
root = pathlib.Path('/ml2/codex-storage/future-workspaces/2026-10-21/task-4')
work = root / 'evidence/core081-qualification'
sys.path.insert(0, str(work / 'site'))
import torch
import pytest
from particle_sliders import lifecycle, inference, pg09
from lumen_studio import lifecycle as product
canonical = json.loads((root / 'anima/validation/core081-canonical-receipt.json').read_text())
for relative, digest in canonical['source_sha256'].items():
    assert hashlib.sha256((work / 'site' / relative).read_bytes()).hexdigest() == digest, relative
assert str(work / 'site') in lifecycle.__file__ and str(work / 'site') in product.__file__
assert importlib.metadata.version('particle-sliders-core') == '0.8.1'
assert importlib.metadata.version('particlegan') == '0.9.1'
assert lifecycle.backend('anima') is product.BACKEND
assert inference.inference_backend('anima') is product.NATIVE_BACKEND
assert pg09._released_sources() == '04624298310cd4b28d5bf687a1eac8951efd9439fb0310709ba5b9703208ceab'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert not torch.cuda.is_initialized()
class SingleCPU:
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()
    @pytest.hookimpl(trylast=True)
    def pytest_runtest_teardown(self, item, nextitem):
        assert torch.get_num_threads() == 1 and not torch.cuda.is_initialized()
sys.path.insert(0, str(root / 'anima/tests/lumen_studio'))
# Only preserved source command wrappers use this path; the already imported
# lumen_studio and particle_sliders package paths remain installed-wheel paths.
sys.path.insert(0, str(root / 'anima'))
paths = sys.argv[1:] or [str(root / 'anima/tests/lumen_studio/test_image_lifecycle.py')]
code = pytest.main([*paths, '-q', '-o', 'pythonpath=', '-p', 'no:cacheprovider', '--basetemp=' + str(work / 'pytest-temp'), '--junitxml=' + str(work / 'legacy-dispatch.junit.xml')], plugins=[SingleCPU()])
assert not torch.cuda.is_initialized()
assert all(str(work / 'site') in module.__file__ for name, module in tuple(sys.modules.items()) if name.startswith('lumen_studio') and getattr(module, '__file__', None))
print(json.dumps(dict(product_origin=product.__file__, core_origin=lifecycle.__file__, core_version='0.8.1', pg_version='0.9.1', source_hashes_matched=30, threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(), cuda_initialized=False), indent=2))
raise SystemExit(code)
