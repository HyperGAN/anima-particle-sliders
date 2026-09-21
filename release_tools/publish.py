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
REPO='ntc-ai/anima-concept-sliders'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--publish',action='store_true',help='Upload after all checks pass')
    p.add_argument('--expected-parent',help='Exact Hub commit required for an additive release update')
    args=p.parse_args();folder=args.folder.resolve()
    for name in ('source','comfyui','shared-core'):
        assert json.loads((ROOT/f'validation/{name}.json').read_text())['passed']
    catalog=json.loads((folder/'catalog.json').read_text())
    assert [e['id'] for e in catalog['sliders']]==['candlelit','moonlit']
    for e in catalog['sliders']:
        assert digest(folder/e['particle'])==e['sha256']
        for key in ('native_lora','comfyui_lora'):assert (folder/e[key]).is_file()
        assert len(e['samples'])==24
        for sample in e['samples']:
            assert 0<=sample['strength']<=5
            for key in ('image','metadata'):assert (folder/sample[key]).is_file()
    assert len(list((folder/'distilled/samples').glob('*.png')))==24
    card=(folder/'README.md').read_text()
    assert card.index('## Samples')<card.index('## Get the adapters')<card.index('## How the sliders learn')
    assert 'https://github.com/mikkel/anima-concept-sliders#comfyui' in card
    for name in re.findall(r'https://huggingface.co/ntc-ai/anima-concept-sliders/resolve/main/([^)?\s]+)',card):
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
    import concept_slider_core
    core_root=Path(concept_slider_core.__file__).parent
    for name,expected in core['files'].items():
        assert digest(core_root/name)==expected,('Shared core source mismatch',name)
    provenance=dict(repository='https://github.com/mikkel/anima-concept-sliders',commit=commit,shared_core=core,
        files={f:dict(sha256=digest(ROOT/f),bytes=(ROOT/f).stat().st_size) for f in files})
    (folder/'source-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    with zipfile.ZipFile(folder/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(ROOT/name,'anima-concept-sliders/'+name)
    manifest=dict(source_commit=commit,files={str(f.relative_to(folder)):dict(bytes=f.stat().st_size,sha256=digest(f))
        for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='release-manifest.json'})
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
        local={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
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
        commit_message='Publish reproducible Anima release with pinned sliders-conceptmod shared core')
    revision=result.oid
    remote={f.rfilename:f for f in api.model_info(REPO,revision=revision,files_metadata=True).siblings}
    checked=0
    for path in sorted(folder.rglob('*')):
        if not path.is_file():continue
        name=str(path.relative_to(folder));r=remote[name]
        assert r.size==path.stat().st_size,name
        if r.lfs:assert r.lfs.sha256==digest(path),name
        else:
            data=path.read_bytes()
            assert r.blob_id==hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest(),name
        checked+=1
    for name in ('README.md','weights/candlelit.safetensors','distilled/comfyui/moonlit.safetensors'):
        downloaded=Path(hf_hub_download(REPO,name,revision=revision))
        assert digest(downloaded)==digest(folder/name),name
    report=dict(repo=REPO,commit=revision,source_commit=commit,core_commit=core['commit'],verified_files=checked,readbacks=3,
        previous_commit=before.sha,existing_checkpoints_and_samples_preserved=True)
    (folder.parent/'publication.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
