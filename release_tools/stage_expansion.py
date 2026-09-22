"""Add calibrated sliders without replacing previously published artifacts."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from release_tools.build import copy,sha
from release_tools.calibrated_release import write_json
from lumen_studio.alpha_adapter import read_metadata


def stage(studio,results,folder):
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    plan=json.loads((ROOT/'data/expansion-plan.json').read_text())
    choices=json.loads((ROOT/'data/expansion-selection.json').read_text())
    catalog=json.loads((folder/'catalog.json').read_text())
    reports=[]
    for spec in plan['sliders']:
        name=spec['id'];selected=choices[name];result=results/name
        source=studio/spec['run_directory']/f"ema-{selected['step']:06}.safetensors"
        assert read_metadata(source)['manifest_sha256']==spec['manifests']['train']
        original=f'weights/{name}.safetensors'
        particle=f'weights/{name}-unit-alpha.safetensors'
        copy(source,folder/original);copy(result/'particle.safetensors',folder/particle)
        for path in (original,particle):write_json((folder/path).with_suffix('.json'),read_metadata(folder/path))
        before,after=[load_file(str(folder/p)) for p in (original,particle)]
        assert all(torch.equal(v,after[k]) for k,v in before.items() if not k.endswith('.alpha'))
        entry=dict(id=name,training_id=spec['training_id'],label=spec['label'],step=selected['step'],
            training_until=max(spec['checkpoint_candidates']),particle=particle,sha256=sha(folder/particle),
            rank=8,particle_alpha=selected['particle_alpha'],lora_alpha=selected['lora_alpha'],recommended_strength=1,
            experimental=selected['step']==200,render_requests=spec['render_requests'],samples=[],distilled_samples=[],
            comparisons=[],off_references={},archived_release=dict(particle=original),selection=selected)
        for fmt,key in [('native','native_lora'),('comfyui','comfyui_lora')]:
            path=f'distilled/{fmt}/{name}-unit-alpha.safetensors'
            old=f'distilled/{fmt}/{name}.safetensors'
            copy(result/f'final/{fmt}/{name}-unit-alpha.safetensors',folder/path)
            copy(result/f'distilled/{fmt}/{name}.safetensors',folder/old)
            entry[key]=path;entry[key+'_sha256']=sha(folder/path);entry['archived_release'][key]=old
        evidence=folder/'evidence'/name
        for source_dir,part in [('particle-sweep','particle-alpha-sweep'),('lora-sweep','lora-alpha-sweep')]:
            for path in (result/source_dir).iterdir():
                if path.is_file():copy(path,evidence/part/path.name)
        copy(result/'distilled/evaluation.json',evidence/'distillation.json')
        write_json(evidence/'selection.json',selected)
        run=studio/spec['run_directory']
        for filename in ('run.json','updates.jsonl','resume-check.json','expressive-pilot-evidence.json',
                         'pilot-qualification.json','normalization.pt'):
            if (run/filename).exists():copy(run/filename,evidence/'training'/filename)
        for path in run.glob('probe-*.json'):copy(path,evidence/'training'/path.name)
        for split in ('train','dev'):
            copy(ROOT/f'data/{name}-{split}.json',folder/f'data/{name}-{split}.json')
            copy(studio/'targets'/spec['training_id']/split/'index.json',evidence/f'{split}-targets-index.json')
        studio_source=studio.parents[2]
        compiler='material_concepts.py' if name=='opal-fever' else 'experiments.py' if name=='dusk' else 'expressive_concepts.py'
        manifest=json.loads((ROOT/f'data/{name}-train.json').read_text())
        assert sha(studio_source/'lumen_studio'/compiler)==manifest['compiler_sha256']
        copy(studio_source/'lumen_studio'/compiler,evidence/'source'/compiler)
        definition=studio_source/f"configs/anima/candidates/{spec['training_id']}-v1.json"
        assert sha(definition)==manifest['source_spec_sha256']
        copy(definition,evidence/'source'/definition.name)
        replay=[]
        gain=selected['lora_alpha']/selected['particle_alpha']
        for request in spec['render_requests']:
            case=request['case'];samples=[]
            for fmt,strength in [('particles',1),('lora',1),('off',0)]:
                image=f"samples/{plan['version']}/{name}/{case}/{fmt}.png"
                local=result/'final/samples'/case/f'{fmt}.png'
                copy(local,folder/image)
                metadata=json.loads(local.with_suffix('.json').read_text())
                adapter_path=entry['particle'] if fmt=='particles' else entry['native_lora'] if fmt=='lora' else None
                metadata.update(variation=name,adapter=adapter_path,
                    sampling='fixed prompt and seed; rendered from published-format export')
                assert metadata['adapter_sha256']==(sha(folder/adapter_path) if adapter_path else None)
                write_json((folder/image).with_suffix('.json'),metadata)
                sample=dict(case=case,format=fmt,strength=strength,image=image,metadata=str(Path(image).with_suffix('.json')))
                samples.append(sample)
                entry['distilled_samples' if fmt=='lora' else 'samples'].append(sample)
                candidate=(result/'particle-sweep'/f"step{selected['step']}-alpha{selected['particle_alpha']:g}-{case}.png" if fmt=='particles' else
                           result/'lora-sweep'/f'gain{gain:g}-{case}.png' if fmt=='lora' else
                           result/'particle-sweep'/f'{case}-off.png')
                assert np.array_equal(np.asarray(Image.open(local)),np.asarray(Image.open(candidate)))
                replay.append(dict(case=case,format=fmt,pixel_exact=True,image=image))
            entry['comparisons'].append(dict(case=case,asset=f"assets/{plan['version']}-{name}-{case}.jpg",samples=samples))
            entry['off_references'][case]=f'evidence/{name}/particle-alpha-sweep/{case}-off.png'
        entry['featured_comparison']=entry['comparisons'][0]
        entry['heldout_projection_relative_mse']=json.loads((result/'distilled/evaluation.json').read_text())['heldout_projection_relative_mse']
        from release_tools.refresh_examples import apply_refreshes
        apply_refreshes(dict(sliders=[entry]))
        old=next((i for i,e in enumerate(catalog['sliders']) if e['id']==name),None)
        if old is None:catalog['sliders'].append(entry)
        else:
            assert catalog['sliders'][old]==entry,'Previously staged expansion changed'
        reports.append(dict(id=name,passed=True,selected=selected,learned_particle_tensors_unchanged=True,
            source_sha256=sha(source),particle_sha256=entry['sha256'],pixel_replays=replay))
    write_json(folder/'catalog.json',catalog)
    write_json(ROOT/'validation/expansion.json',dict(passed=True,final_test_used=False,sliders=reports))
    for name in ('expansion-plan','expansion-selection'):copy(ROOT/f'data/{name}.json',folder/f'data/{name}.json')
    print('Staged',len(reports),'sliders and verified',sum(len(r['pixel_replays']) for r in reports),'pixel replays')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('studio','results','folder'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();stage(args.studio,args.results,args.folder)
