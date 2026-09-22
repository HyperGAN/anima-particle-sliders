"""Add Bad Intent to an inspected release without replacing existing samples or weights."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from release_tools.build import comparison,copy,sha,RAW,WEB
from lumen_studio.alpha_adapter import read_metadata


def json_file(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True,help='Original Studio artifact root for evidence')
    parser.add_argument('--teacher',type=Path,required=True)
    args=parser.parse_args();out=args.folder;results=args.results
    metadata=read_metadata(args.teacher)
    assert metadata['step']==1600 and metadata['network_alpha']==8
    teacher_sha=sha(args.teacher)
    report=json.loads((results/'distilled/bad-intent-evaluation.json').read_text())
    assert report['teacher_sha256']==teacher_sha
    copy(args.teacher,out/'weights/bad-intent.safetensors')
    json_file(out/'weights/bad-intent.json',metadata)
    for path in (results/'distilled').rglob('*'):
        if path.is_file():copy(path,out/'distilled'/path.relative_to(results/'distilled'))
    for path in (results/'samples/bad-intent').rglob('*'):
        if path.is_file():copy(path,out/'samples/bad-intent'/path.relative_to(results/'samples/bad-intent'))
    requests=json.loads((ROOT/'data/bad-intent-samples.json').read_text())
    samples=[]
    for request in requests:
        stem=f"samples/bad-intent/case{request['case']}/str{request['strength']:g}"
        record=json.loads((out/(stem+'.json')).read_text())
        for key in ('prompt','seed','width','height','steps','energy'):
            assert record[key]==request['payload'][key],(stem,key)
        assert record['model_identity']==metadata['model_identity']
        assert record['adapter_sha256']==teacher_sha
        samples.append(dict(case=request['case'],strength=request['strength'],image=stem+'.png',metadata=stem+'.json'))
    for case in (0,1):
        paths=[f'samples/bad-intent/case{case}/str1.png',f'distilled/samples/bad-intent-case{case}-str1.png',f'samples/bad-intent/case{case}/str0.png']
        records=[json.loads((out/p).with_suffix('.json').read_text()) for p in paths]
        for key in ('prompt','seed','width','height','steps','cfg','model_identity'):
            assert all(r[key]==records[0][key] for r in records),(case,key)
        assert [r['energy'] for r in records]==[1,1,0]
        comparison([out/p for p in paths],out/f'assets/bad-intent-case{case}-particle-distill-off.jpg')
    catalog=json.loads((out/'catalog.json').read_text())
    entry=dict(id='bad-intent',label='Bad Intent',training_id='uncanny',step=1600,
        particle='weights/bad-intent.safetensors',sha256=teacher_sha,
        native_lora='distilled/native/bad-intent.safetensors',comfyui_lora='distilled/comfyui/bad-intent.safetensors',
        samples=samples,identity_preservation=False,recommended_strength=1,
        featured_comparison=dict(asset='assets/bad-intent-case0-particle-distill-off.jpg',samples=[
            dict(format=fmt,strength=strength,image=path,metadata=str(Path(path).with_suffix('.json')))
            for fmt,strength,path in [('particles',1,'samples/bad-intent/case0/str1.png'),
                ('lora',1,'distilled/samples/bad-intent-case0-str1.png'),('off',0,'samples/bad-intent/case0/str0.png')]]))
    catalog['sliders']=[entry]+[e for e in catalog['sliders'] if e['id']!='bad-intent']
    json_file(out/'catalog.json',catalog)
    evidence=out/'evidence/bad-intent'
    for pattern in ('run.json','updates.jsonl','probe-*.json','resume-check.json','continuation-pilot-evidence.json'):
        for path in (args.root/'runs/uncanny').glob(pattern):copy(path,evidence/path.name)
    for split in ('train','dev'):
        copy(args.root/f'targets/uncanny/{split}/index.json',evidence/f'{split}-targets-index.json')
    copy(args.root/'runs/uncanny/normalization.pt',evidence/'normalization.pt')
    json_file(evidence/'particle-metadata.json',metadata)
    json_file(evidence/'release-notes.json',dict(public_name='Bad Intent',training_id='uncanny',
        teacher_sha256=teacher_sha,original_checkpoint_sha256=metadata['source_checkpoint_sha256'],
        original_training_unchanged=True,identity_preservation=False,final_test_evaluated=False,
        distillation_method=report['method'],heldout_projection_relative_mse=report['heldout_projection_relative_mse'],
        visual_review='At strength 1 the distill is much milder: it does not reproduce the particle teacher\'s looming crop or pronounced expression on these two development cases. The bare male particle sample also changes rendering medium. No gore in these comparisons.'))
    card=(out/'README.md').read_text()
    if '### Bad Intent' not in card:
        card=card.replace('**Moonlit and Candlelit for Anima Turbo v1.1.** Same prompt and seed; turn the atmosphere from strength 0 to 5.',
            '**Bad Intent — psychological horror for Anima Turbo v1.1.** An eager smile, a looming gaze, something wrong. Plus Moonlit and Candlelit lighting sliders. Each comes as the original Particle adapter and a standard LoRA Distill.')
        prompt=requests[0]['payload']['prompt']
        bare=next(r['payload']['prompt'] for r in requests if r['case']==1)
        section=f'''### Bad Intent

Intense psychological horror, without gore. Start at **strength 1**. This deliberately changes expression, pose, framing and sometimes appearance or medium; identity preservation is outside its brief. Formerly named Uncanny.

**Particle · strength 1** → **Distill · strength 1** → **Off · strength 0**

![Bad Intent: Particle first, Distill second, Off third]({RAW}assets/bad-intent-case0-particle-distill-off.jpg)

Full-resolution samples: [Particle]({RAW}samples/bad-intent/case0/str1.png) · [Distill]({RAW}distilled/samples/bad-intent-case0-str1.png) · [Off]({RAW}samples/bad-intent/case0/str0.png)

Same prompt and seed **29001**. 768 × 768, 10 Euler steps, CFG 1. **The Distill is much milder at the same strength:** it does not reproduce the original's looming pose or pronounced expression in these examples. Choose Particle for the featured effect.

<details><summary>Prompt</summary>

{prompt}

</details>

<details><summary>Bad Intent: bare prompt comparison and strength 0.5 samples</summary>

![Bad Intent bare prompt: particle versus distilled LoRA and Off at strength 1]({RAW}assets/bad-intent-case1-particle-distill-off.jpg)

Particle → Distill → Off, at strengths 1 → 1 → 0. Same seed and settings as above. The particle version also shifts this case toward a rendered/realistic medium.

{bare}

[All Bad Intent particle samples, strengths 0 / 0.5 / 1]({WEB}/tree/main/samples/bad-intent) · [Distill samples]({WEB}/tree/main/distilled/samples)

</details>

'''
        card=card.replace('### Moonlit\n',section+'### Moonlit\n',1)
        row=f"| Bad Intent | [Download]({RAW}{entry['particle']}?download=true) | [Download]({RAW}{entry['comfyui_lora']}?download=true) | [Download]({RAW}{entry['native_lora']}?download=true) |\n"
        card=card.replace('|---|---|---|---|\n','|---|---|---|---|\n'+row,1)
        card=card.replace('|---|---:|\n',f"|---|---:|\n| Bad Intent | {report['heldout_projection_relative_mse']:.6f} |\n",1)
        card=card.replace('Candlelit and Moonlit each have their own adapter, critic, and learned particle cloud.',
            'Bad Intent, Candlelit and Moonlit each have their own adapter, critic, and learned particle cloud. Bad Intent uses the same game with a broader psychological-horror target: expression, pose and framing may change as well as atmosphere. Its internal training identifier remains `uncanny`.')
        card=card.replace('Six lighting definitions train each slider; two extra paraphrases are held out.',
            'Six definitions train each slider; two extra paraphrases are held out. Candlelit and Moonlit edit lighting; Bad Intent uses expression, pose, framing and atmosphere definitions.')
        card=card.replace('More samples: strengths 0–5, four matched prompts per slider','More lighting samples: strengths 0–5, four matched prompts per slider')
        card=card.replace('Compare particles and distills at strengths 1, 3 and 5','Compare lighting particles and distills at strengths 1, 3 and 5')
    (out/'README.md').write_text(card)
    github=card.split('---\n',2)[2] if card.startswith('---\n') else card
    for name in ('catalog.json','release-manifest.json'):
        github=github.replace(']('+name+')',']('+WEB+'/blob/main/'+name+')')
    if '<a id="comfyui"></a>' not in github:
        github=github.replace('## Get the adapters','<a id="comfyui"></a>\n\n## Get the adapters',1)
    (ROOT/'README.md').write_text(github)
    for name in ('FORMULATION.md','DISTILLATION.md','REPRODUCE.md','COMFYUI.md'):
        copy(ROOT/name,out/name)
    for path in (ROOT/'validation').glob('*.json'):copy(path,out/'validation'/path.name)
    for path in (ROOT/'data').glob('bad-intent-*.json'):copy(path,out/'data'/path.name)
    plugin=['__init__.py','comfy_particle.py','particle_io.py','requirements.txt','core.lock.json','COMFYUI.md','LICENSE','ANIMA-LICENSE.md','NOTICE',
        'lumen_studio/__init__.py','lumen_studio/particles.py','lumen_studio/contracts.py','lumen_studio/alpha_adapter.py',
        'lumen_studio/vendor/__init__.py','lumen_studio/vendor/reference.py','lumen_studio/vendor/LICENSE','lumen_studio/vendor/provenance.json']
    with zipfile.ZipFile(out/'comfyui/anima-concept-sliders.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in plugin:z.write(ROOT/name,'anima-concept-sliders/'+name)
    print('Staged Bad Intent: 6 particle samples, 4 distill samples, 2 comparisons, three adapter formats')


if __name__=='__main__':main()
