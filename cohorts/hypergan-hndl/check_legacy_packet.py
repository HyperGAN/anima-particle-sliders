"""Reject genuine installed stable and optional older-preview packets.

Requires two or three explicitly selected environments. No checked-in model/checkpoint
blob, synthetic manifest substitution, migration or pretrained model is used.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


CODE = r'''
import hashlib,json,sys,torch
from pathlib import Path
from importlib.metadata import version
fixtures,root,mode = Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
sys.path.insert(0,str(fixtures))
from test_pg09_training import runtime_fixture,equal
from test_image_lifecycle import configuration
from particle_sliders.lifecycle import SessionConfig,open_session,CheckpointError,read_packet
from particle_sliders.inference import open_inference
runtime=runtime_fixture()
if mode!='legacy':
    from anima_hndl.lifecycle import LocalTurboRuntime
    from anima_hndl.source_validation import verify_cohort
    original=runtime
    runtime=LocalTurboRuntime.__new__(LocalTurboRuntime)
    runtime.__dict__.update(original.__dict__)
directory=root/mode
directory.mkdir()
config=configuration(directory,runtime)
if mode=='candidate':
    assert version('anima-particle-sliders')=='0.3.0a2'
    assert version('particle-sliders-core')=='0.12.0'
    config=SessionConfig('anima-hndl','anima-hypergan-hndl-v3-cpu',**config.options)
elif mode=='preview':
    assert version('anima-particle-sliders')=='0.3.0a1'
    assert version('particle-sliders-core')=='0.12.0'
    assert verify_cohort()['core_commit']=='4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231'
    config=SessionConfig('anima-hndl','anima-hypergan-hndl-v2-cpu',**config.options)
else:
    assert version('anima-particle-sliders')=='0.2.0'
    assert version('particle-sliders-core')=='0.8.1'
    assert version('particlegan')=='0.9.1'
with open_session(config,runtime=runtime) as session:
    session.train(stop_after=2)
    if mode in ('legacy','preview'):
        packet=session.checkpoint(root/('actual-core081.pt' if mode=='legacy' else 'actual-preview-a1.pt'))
        result={'created_actual_packet':True,'packet_sha256':packet['sha256']}
        if mode=='preview':
            for selection in ('raw','ema'):
                session.export_inference(root/('actual-preview-a1-'+selection+'.pt'),selection=selection)
    else:
        before=session.checkpoint()
        legacy_packet=read_packet(root/'actual-core081.pt')
        assert isinstance(legacy_packet,dict) and legacy_packet['schema'].endswith('-training')
        try:session.restore(legacy_packet)
        except CheckpointError as exc:
            reason=str(exc)
            assert reason=='packet kind, source, capabilities or cohort differs; no implicit rebinding',reason
        else:raise AssertionError('historical packet was admitted')
        assert equal(before,session.checkpoint()) and not session.closed
        result={'actual_legacy_packet_rejected':True,'complete_state_unchanged':True,'reason':reason,
                'restored_input_type':type(legacy_packet).__name__}
        preview=root/'actual-preview-a1.pt'
        if preview.is_file():
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('actual-preview-a1*.pt')}
            preview_packet=read_packet(preview)
            assert isinstance(preview_packet,dict)
            try:session.restore(preview_packet)
            except CheckpointError as exc:
                assert str(exc)=='packet kind, source, capabilities or cohort differs; no implicit rebinding',str(exc)
                result['preview_refusal_reason']=str(exc)
            else:raise AssertionError('old preview training packet admitted')
            assert equal(before,session.checkpoint()) and not session.closed
            for selection in ('raw','ema'):
                try:open_inference(root/('actual-preview-a1-'+selection+'.pt'),operation='response')
                except CheckpointError as exc:
                    assert str(exc)=='native factory source, protocol or capabilities differ',str(exc)
                    result['preview_'+selection+'_refusal_reason']=str(exc)
                else:raise AssertionError('old preview native packet admitted')
            assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('actual-preview-a1*.pt')}
            result['preview_packet_bytes_unchanged']=hashes
runtime.mixer.close()
assert not torch.cuda.is_initialized()
if mode!='legacy':result['cohort']=verify_cohort()
print(json.dumps(dict(result,mode=mode,product=version('anima-particle-sliders'),core=version('particle-sliders-core'),pg=version('particlegan'))))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-python', type=Path, required=True)
    parser.add_argument('--candidate-python', type=Path, required=True)
    parser.add_argument('--preview-python', type=Path, help='Optional original 0.3.0a1/core 4abc environment')
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    fixtures = root / 'tests/lumen_studio'
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2')
    records = []
    environments = [('legacy', args.legacy_python)]
    if args.preview_python:
        environments.append(('preview', args.preview_python))
    environments.append(('candidate', args.candidate_python))
    for mode, python in environments:
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
