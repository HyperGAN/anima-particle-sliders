"""Publish an audited release folder and verify every remote file hash."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import zipfile
from huggingface_hub import HfApi,ModelCard,hf_hub_download
ROOT=Path(__file__).resolve().parents[1]
REPO='ntc-ai/anima-particle-sliders'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def release_files(folder):
    # snapshot_download bookkeeping is local. The Hub manages .gitattributes
    # and adds LFS entries during upload, so it is not an immutable artifact.
    return [p for p in sorted(folder.rglob('*')) if p.is_file()
            and p.relative_to(folder).parts[0] not in ('.cache','.git','.gitattributes')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--publish',action='store_true',help='Upload after all checks pass')
    p.add_argument('--expected-parent',help='Exact Hub commit required for an additive release update')
    args=p.parse_args();folder=args.folder.resolve()
    for name in ('source','comfyui','shared-core'):
        assert json.loads((ROOT/f'validation/{name}.json').read_text())['passed']
    catalog=json.loads((folder/'catalog.json').read_text())
    ids=[e['id'] for e in catalog['sliders']]
    assert ids[:3]==['bad-intent','candlelit','moonlit'] and len(ids)==len(set(ids))
    for e in catalog['sliders']:
        assert digest(folder/e['particle'])==e['sha256']
        for key in ('native_lora','comfyui_lora'):assert (folder/e[key]).is_file()
        assert len(e['samples'])==2*len(e['comparisons'])
        assert len(e['distilled_samples'])==len(e['comparisons'])
        assert e['recommended_strength']==1
        for sample in e['samples']+e['distilled_samples']:
            assert 0<=sample['strength']<=5
            for key in ('image','metadata'):assert (folder/sample[key]).is_file()
    assert len(list((folder/'distilled/samples').glob('*.png')))==28
    bad=catalog['sliders'][0]
    assert bad['training_id']=='uncanny' and bad['recommended_strength']==1
    assert {s['strength'] for s in bad['samples']}=={0,1}
    calibration=json.loads((ROOT/'validation/unit-alpha.json').read_text())
    assert calibration['passed']
    assert calibration['particle_alphas']=={e['id']:e['particle_alpha'] for e in catalog['sliders']}
    assert calibration['lora_alphas']=={e['id']:e['lora_alpha'] for e in catalog['sliders']}
    comfy=json.loads((ROOT/'validation/comfyui.json').read_text())
    assert set(ids)<={r['variation'] for r in comfy['sliders']}
    additions=set(ids)-{'bad-intent','candlelit','moonlit'}
    if additions:
        expansion=json.loads((ROOT/'validation/expansion.json').read_text())
        assert expansion['passed'] and additions<={r['id'] for r in expansion['sliders'] if r['passed']}
    card=(folder/'README.md').read_text()
    assert card.index('## Samples')<card.index('## Get the adapters')<card.index('## How the sliders learn')
    assert card.index('### Bad Intent')<card.index('### Moonlit')<card.index('### Candlelit')
    assert 'https://github.com/HyperGAN/anima-particle-sliders#comfyui' in card
    for name in re.findall(r'https://huggingface.co/ntc-ai/anima-particle-sliders/resolve/main/([^)?\s]+)',card):
        assert (folder/name).is_file(),('Broken release link',name)
    # Verify finite tensors in every original/exported checkpoint.
    import torch
    torch.set_num_threads(4)
    from safetensors.torch import load_file
    for weight in folder.rglob('*.safetensors'):
        assert all(torch.isfinite(t).all() for t in load_file(str(weight)).values()),weight
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit source before publishing'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    files=[f for f in files if f]
    core=json.loads((ROOT/'core.lock.json').read_text())
    assert core['package']=='particle-sliders-core' and core['import']=='particle_sliders'
    assert core['commit']=='a119ca1ecd3d5d6c437065839d22739b04f2f4d8'
    assert 'files' not in core,'Retired concept-slider-core file hashes are not the source of truth'
    import particle_sliders
    from importlib.metadata import distribution
    installed=json.loads(distribution('particle-sliders-core').read_text('direct_url.json') or '{}')
    assert installed.get('vcs_info',{}).get('commit_id')==core['commit'],installed
    assert particle_sliders.winning_formulation().architecture_id==core['architecture_id']
    provenance=dict(repository='https://github.com/HyperGAN/anima-particle-sliders',commit=commit,shared_core=core,
        files={f:dict(sha256=digest(ROOT/f),bytes=(ROOT/f).stat().st_size) for f in files})
    (folder/'source-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    with zipfile.ZipFile(folder/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(ROOT/name,'anima-particle-sliders/'+name)
    manifest=dict(source_commit=commit,files={str(f.relative_to(folder)):dict(bytes=f.stat().st_size,sha256=digest(f))
        for f in release_files(folder) if f.name!='release-manifest.json'})
    (folder/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    ModelCard.load(folder/'README.md').validate()
    print('Validated',len(manifest['files']),'release files; source',commit,flush=True)
    if not args.publish:return
    api=HfApi()
    api.create_repo(REPO,repo_type='model',private=False,exist_ok=True)
    before=api.model_info(REPO,files_metadata=True)
    existing={f.rfilename:f for f in before.siblings}
    if set(existing)-{'.gitattributes'}:
        assert args.expected_parent==before.sha,'Supply the inspected release parent before updating'
        local={str(p.relative_to(folder)) for p in release_files(folder)}
        assert set(existing)-{'.gitattributes'}<=local,'Reconcile unrecognized remote files'
        # Source/docs may evolve; preserve every released checkpoint and sample.
        for name,r in existing.items():
            if not name.startswith(('weights/','samples/','distilled/','assets/','evidence/')):continue
            path=folder/name
            assert r.size==path.stat().st_size,('Protected release artifact changed',name)
            if r.lfs:assert r.lfs.sha256==digest(path),name
            else:
                data=path.read_bytes()
                assert r.blob_id==hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest(),name
    elif args.expected_parent:
        assert args.expected_parent==before.sha,'Release parent changed'
    result=api.upload_folder(repo_id=REPO,repo_type='model',folder_path=folder,
        parent_commit=before.sha,
        ignore_patterns=['.cache/**','.git/**','.gitattributes'],
        commit_message='Publish calibrated particle sliders, ordinary LoRAs and matched examples')
    revision=result.oid
    remote={f.rfilename:f for f in api.model_info(REPO,revision=revision,files_metadata=True).siblings}
    checked=0
    for path in release_files(folder):
        name=str(path.relative_to(folder));r=remote[name]
        assert r.size==path.stat().st_size,name
        if r.lfs:assert r.lfs.sha256==digest(path),name
        else:
            data=path.read_bytes()
            assert r.blob_id==hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest(),name
        checked+=1
    readbacks=['README.md']+[e[k] for e in catalog['sliders'] for k in ('particle','native_lora','comfyui_lora')]
    for name in readbacks:
        downloaded=Path(hf_hub_download(REPO,name,revision=revision))
        assert digest(downloaded)==digest(folder/name),name
    report=dict(repo=REPO,commit=revision,source_commit=commit,core_commit=core['commit'],verified_files=checked,readbacks=len(readbacks),
        previous_commit=before.sha,existing_checkpoints_and_samples_preserved=True,
        hub_managed_files=['.gitattributes'])
    (folder.parent/'publication.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
