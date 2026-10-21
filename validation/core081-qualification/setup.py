import hashlib, json, os, pathlib, shutil, subprocess, sys, zipfile
from setuptools import build_meta
root = pathlib.Path('/ml2/codex-storage/future-workspaces/2026-10-21/task-4')
work = root / 'evidence/core081-qualification'
product = root / 'anima'
canonical = json.loads((product / 'validation/core081-canonical-receipt.json').read_text())
corewheel = root.parent / 'task-2/evidence/core081-build/particle_sliders_core-0.8.1-py3-none-any.whl'
assert hashlib.sha256(corewheel.read_bytes()).hexdigest() == canonical['artifacts'][corewheel.name]
assert not (work / 'site').exists(), 'fresh install target required'
(work / 'dist').mkdir(exist_ok=True)
os.chdir(product)
wheel = work / 'dist' / build_meta.build_wheel(str(work / 'dist'))
with zipfile.ZipFile(wheel) as archive:
    metadata = archive.read('anima_particle_sliders-0.2.0.dist-info/METADATA').decode()
    assert canonical['clean_head'] in metadata
    assert archive.read('lumen_studio/lifecycle.py') == (product / 'lumen_studio/lifecycle.py').read_bytes()
    entrypoints = archive.read('anima_particle_sliders-0.2.0.dist-info/entry_points.txt').decode()
    assert '[particle_sliders.inference_backends]' in entrypoints
subprocess.run([sys.executable, '-I', '-m', 'pip', 'install', '--no-index', '--no-deps', '--target', str(work / 'site'), str(corewheel), str(wheel)], check=True)
for path in (product / 'build', product / 'anima_particle_sliders.egg-info'):
    if path.exists():
        shutil.rmtree(path)
receipt = dict(product_wheel=str(wheel), product_wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(), core_wheel=str(corewheel), core_wheel_sha256=hashlib.sha256(corewheel.read_bytes()).hexdigest(), installation_target=str(work / 'site'), dependency_resolution=False, source_head=subprocess.check_output(['git','-C',str(product),'rev-parse','HEAD'],text=True).strip())
(work / 'artifacts.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
