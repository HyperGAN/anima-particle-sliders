"""Stage release artifacts with original checkpoint and sample provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image
import requests
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--api',default='http://127.0.0.1:8876/api')
    args=p.parse_args();out=args.output
    catalog=[]
    from safetensors import safe_open
    from particle_io import ParticleNetwork
    for variation in ('candlelit','moonlit'):
        run=args.root/'runs'/variation
        weight=run/'ema-001600.safetensors'
        network,meta=ParticleNetwork.load(weight)
        assert meta['step']==1600 and meta['weights']=='ema'
        dest=f'weights/{variation}.safetensors'
        copy(weight,out/dest)
        write(out/f'weights/{variation}.json',meta)
        evidence=out/'evidence'/variation
        for name in ('run.json','updates.jsonl','normalization.pt','pilot-qualification.json','probe-000000.json','probe-001600.json','resume-check.json'):
            copy(run/name,evidence/name)
        lines=[json.loads(l) for l in (run/'updates.jsonl').read_text().splitlines()]
        assert [r['step'] for r in lines]==list(range(1,1601))
        for name in ('report.json','probe-parity.json'):
            copy(args.root/'diagnostics/convergence'/variation/name,evidence/'convergence'/name)
        for split in ('train','dev'):
            copy(args.root/'targets'/variation/split/'index.json',evidence/f'{split}-targets-index.json')
        catalog.append(dict(id=variation,label=variation.title(),step=1600,particle=dest,
            sha256=digest(weight),native_lora=f'distilled/native/{variation}.safetensors',
            comfyui_lora=f'distilled/comfyui/{variation}.safetensors',samples=[]))
    jobs=json.loads(args.jobs.read_text());pending=[]
    for row in jobs:
        variation,case,strength=row['variation'],row['case'],row['strength']
        ident=row['jobs']['ids'][0]
        source=args.root/'studio/images'/f'{ident}.png'
        name=f'samples/{variation}/case{case}/str{strength}'
        # A standalone rendering pass can supply any samples that were pending
        # when another Studio task took GPU ownership.
        if not source.exists() and (out/(name+'.png')).exists():continue
        if not source.exists():pending.append(ident);continue
        with Image.open(source) as im:
            saved=json.loads(im.info['anima'])
            expected=row['payload']
            for key in ('prompt','seed','width','height','steps','checkpoints','energy'):
                assert saved[key]==expected[key],(ident,key)
            # Copy pixels losslessly and put public provenance in a JSON sidecar.
            target=out/(name+'.png');target.parent.mkdir(parents=True,exist_ok=True)
            im.save(target)
        sidecar=dict(**row['payload'],case_id=row['source_row']['id'],split='dev',
            format='particles',strength=strength,model_identity=saved['model_identity'],
            source_png_sha256=digest(source),source_job_id=ident)
        write(out/(name+'.json'),sidecar)
    for entry in catalog:
        for row in jobs:
            if row['variation']!=entry['id']:continue
            base=f"samples/{entry['id']}/case{row['case']}/str{row['strength']}"
            entry['samples'].append(dict(case=row['case'],strength=row['strength'],image=base+'.png',metadata=base+'.json'))
    write(out/'catalog.json',dict(model='circlestone-labs/Anima:turbo-v1.1',sliders=catalog))
    print('Staged weights and samples; pending:',len(pending))
    write(out/'pending-samples.json',pending)

if __name__=='__main__':main()
