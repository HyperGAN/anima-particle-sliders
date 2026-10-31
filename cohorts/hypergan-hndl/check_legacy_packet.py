"""Create a genuine installed core0.8.1 packet and reject it in the candidate.

Requires two explicitly selected environments. No checked-in model/checkpoint
blob, synthetic manifest substitution, migration or pretrained model is used.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


CODE = r'''
import json,sys,torch
from pathlib import Path
from importlib.metadata import version
fixtures,root,mode = Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
sys.path.insert(0,str(fixtures))
from test_pg09_training import runtime_fixture,equal
from test_image_lifecycle import configuration
from particle_sliders.lifecycle import SessionConfig,open_session,CheckpointError
runtime=runtime_fixture()
directory=root/mode
directory.mkdir()
config=configuration(directory,runtime)
if mode=='candidate':
    assert version('anima-particle-sliders')=='0.3.0a1'
    assert version('particle-sliders-core')=='0.12.0'
    config=SessionConfig('anima-hndl','anima-hypergan-hndl-v2-cpu',**config.options)
else:
    assert version('anima-particle-sliders')=='0.2.0'
    assert version('particle-sliders-core')=='0.8.1'
    assert version('particlegan')=='0.9.1'
with open_session(config,runtime=runtime) as session:
    session.train(stop_after=2)
    if mode=='legacy':
        packet=session.checkpoint(root/'actual-core081.pt')
        result={'created_actual_packet':True,'packet_sha256':packet['sha256']}
    else:
        before=session.checkpoint()
        try:session.restore(root/'actual-core081.pt')
        except CheckpointError as exc:
            reason=str(exc)
            assert reason=='unsupported session packet version; no automatic migration',reason
        else:raise AssertionError('historical packet was admitted')
        assert equal(before,session.checkpoint()) and not session.closed
        result={'actual_legacy_packet_rejected':True,'complete_state_unchanged':True,'reason':reason}
runtime.mixer.close()
assert not torch.cuda.is_initialized()
print(json.dumps(dict(result,product=version('anima-particle-sliders'),core=version('particle-sliders-core'),pg=version('particlegan'))))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-python', type=Path, required=True)
    parser.add_argument('--candidate-python', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    fixtures = root / 'tests/lumen_studio'
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2')
    records = []
    for mode, python in [('legacy', args.legacy_python), ('candidate', args.candidate_python)]:
        result = subprocess.run([str(python.absolute()), '-I', '-c', CODE,
                                 str(fixtures), str(workdir), mode],
                                cwd=workdir, env=env, capture_output=True, text=True, timeout=60)
        (workdir / (mode + '.log')).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(mode + ' environment failed; see ' + str(workdir / (mode + '.log')))
        records.append(json.loads(result.stdout.strip().splitlines()[-1]))
    (workdir / 'receipt.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records))


if __name__ == '__main__':
    main()
