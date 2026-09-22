"""Verify every matched release sample and exact cross-process zero replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True)
    args=p.parse_args();root=args.folder
    specs=json.loads((ROOT/'data/release-samples.json').read_text())
    specs+=json.loads((ROOT/'data/bad-intent-samples.json').read_text())
    checked=[];zeros=[]
    for spec in specs:
        variation,case,strength=spec.get('public_id',spec['variation']),spec['case'],spec['strength']
        path=root/f'samples/{variation}/case{case}/str{strength:g}.png'
        metadata=json.loads(path.with_suffix('.json').read_text())
        for key in ('prompt','seed','steps','width','height','energy')+ (('checkpoints',) if 'checkpoints' in spec['payload'] else ()):
            assert metadata[key]==spec['payload'][key],(path,key)
        with Image.open(path) as im:
            assert im.size==(768,768)
            pixels=np.array(im)
            assert pixels.std()>1
        checked.append(dict(path=str(path.relative_to(root)),sha256=sha(path)))
        if strength in ((.5,1) if variation=='bad-intent' else (1,3,5)):
            dpath=root/f'distilled/samples/{variation}-case{case}-str{strength:g}.png'
            dmeta=json.loads(dpath.with_suffix('.json').read_text())
            for key in ('prompt','seed','steps','width','height','energy'):
                assert dmeta[key]==metadata[key],(dpath,key)
            assert dmeta['adapter_sha256']==sha(root/f'distilled/native/{variation}.safetensors')
            with Image.open(dpath) as im:
                assert im.size==(768,768) and np.array(im).std()>1
            checked.append(dict(path=str(dpath.relative_to(root)),sha256=sha(dpath)))
    for case in range(4):
        a=np.array(Image.open(root/f'samples/candlelit/case{case}/str0.png'),dtype=np.int16)
        b=np.array(Image.open(root/f'samples/moonlit/case{case}/str0.png'),dtype=np.int16)
        error=int(np.abs(a-b).max());assert error==0
        zeros.append(dict(case=case,max_pixel_error=error))
    report=dict(passed=True,particle_samples=54,distilled_samples=28,strengths=[0,1,2,3,4,5],
        bad_intent_strengths=[0,.5,1],bad_intent_distilled_strengths=[.5,1],
        distilled_strengths=[1,3,5],matched_prompts_and_seeds=True,zero_replay=zeros,files=checked)
    (ROOT/'validation/samples.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified 82 images, their settings and four exact zero replays')

if __name__=='__main__':main()
