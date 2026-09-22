"""Build the samples-first Hub card, source card, and complete file manifest."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
REPO='ntc-ai/anima-concept-sliders'
WEB='https://huggingface.co/'+REPO
RAW=WEB+'/resolve/main/'
GITHUB='https://github.com/HyperGAN/anima-particle-sliders'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def grid(images,path,labels):
    cell=384;bar=40
    canvas=Image.new('RGB',(cell*3,(cell+bar)*2),'#111621')
    draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',22)
    except OSError:font=ImageFont.load_default()
    for i,(p,label) in enumerate(zip(images,labels)):
        with Image.open(p) as im:
            assert im.size==(768,768),(p,im.size)
            im=im.convert('RGB').resize((cell,cell),Image.Resampling.LANCZOS)
        x=i%3*cell;y=i//3*(cell+bar)
        canvas.paste(im,(x,y+bar));draw.text((x+12,y+8),label,fill='white',font=font)
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path,quality=94,subsampling=0)


def copy(source,dest):
    if source.resolve()==dest.resolve():return
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)


def package_plugin(out):
    plugin=['__init__.py','comfy_particle.py','particle_io.py','requirements.txt','core.lock.json','COMFYUI.md','LICENSE','ANIMA-LICENSE.md','NOTICE',
        'lumen_studio/__init__.py','lumen_studio/particles.py','lumen_studio/contracts.py','lumen_studio/alpha_adapter.py','single_particle.py',
        'lumen_studio/vendor/__init__.py','lumen_studio/vendor/reference.py','lumen_studio/vendor/LICENSE','lumen_studio/vendor/provenance.json']
    zpath=out/'comfyui/anima-concept-sliders.zip';zpath.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for name in plugin:z.write(ROOT/name,'anima-concept-sliders/'+name)


def comparison(images,path):
    """Three matched samples, with the original particle result leading."""
    cell=384;bar=72
    canvas=Image.new('RGB',(cell*3,cell+bar),'#111621')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('DejaVuSans-Bold.ttf',40)
    for i,(image,label) in enumerate(zip(images,('PARTICLE','DISTILL','OFF'))):
        x=i*cell
        with Image.open(image) as im:
            assert im.size==(768,768),(image,im.size)
            canvas.paste(im.convert('RGB').resize((cell,cell),Image.Resampling.LANCZOS),(x,bar))
        draw.rectangle((x,0,x+cell-1,bar-1),fill='#087f8c' if i==0 else '#111621')
        draw.text((x+18,12),label,font=font,fill='white')
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path,quality=94,subsampling=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output
    catalog=json.loads((out/'catalog.json').read_text())
    if any(e['id'] == 'bad-intent' for e in catalog['sliders']):
        raise ValueError('Use release_tools/add_bad_intent.py to preserve the additive release and its featured portraits')
    entries=sorted(catalog['sliders'],key=lambda e:0 if e['id']=='moonlit' else 1)
    intro='# Anima Concept Sliders\n\n**Moonlit and Candlelit for Anima Turbo v1.1.** Same prompt and seed; turn the atmosphere from strength 0 to 5.\n\n'
    card=intro+'## Samples\n\n**Start with Particle — the original slider.** Each comparison reads **Particle → Distill → Off**, left to right. Distill is the ordinary-LoRA approximation; Off is the base model.\n\n'
    samples=json.loads((ROOT/'data/release-samples.json').read_text())
    for entry in entries:
        name=entry['id'];label=entry['label']
        paths=[f'samples/{name}/case0/str3.png',f'distilled/samples/{name}-case0-str3.png',f'samples/{name}/case0/str0.png']
        records=[json.loads((out/path).with_suffix('.json').read_text()) for path in paths]
        for key in ('prompt','seed','width','height','steps'):
            assert len({record[key] for record in records})==1,(name,key)
        assert [r['energy'] for r in records]==[3,3,0]
        dest=out/f'assets/{name}-particle-distill-off.jpg'
        comparison([out/path for path in paths],dest)
        card+=f'### {label}\n\n**Particle · strength 3** → **Distill · strength 3** → **Off · strength 0**\n\n'
        card+=f'![{label}: Particle first, Distill second, Off third]({RAW}assets/{name}-particle-distill-off.jpg)\n\n'
        card+='Full-resolution samples: '+ ' · '.join(f'[{kind}]({RAW}{path})' for kind,path in zip(('Particle','Distill','Off'),paths))+'\n\n'
        request=records[0]
        card+=f"Same prompt and seed **{request['seed']}**. 768 × 768, 10 Euler steps, CFG 1. Strength 3 extrapolates beyond the trained endpoint at 1.\n\n"
        card+='<details><summary>Prompt</summary>\n\n'+request['prompt']+'\n\n</details>\n\n'
    card+='<details>\n<summary>More samples: strengths 0–5, four matched prompts per slider</summary>\n\nEach grid reads **0, 1, 2** across the first row, then **3, 4, 5**. Original step-1600 EMA particles.\n\n'
    for entry in entries:
        name=entry['id'];label=entry['label']
        card+=f'### {label}\n\n'
        for case in (0,3,1,2):
            images=[out/f'samples/{name}/case{case}/str{s}.png' for s in range(6)]
            dest=out/f'assets/{name}-case{case}.jpg'
            grid(images,dest,[f'Strength {s}' for s in range(6)])
            if case==1:card+='<details>\n<summary>Two more matched prompts</summary>\n\n'
            description='Bare prompt' if case==3 else f'Comparison {case+1}'
            card+=f'**{description}** · [PNG images and prompt]({WEB}/tree/main/samples/{name}/case{case})\n\n'
            card+=f'![{label}: matched samples at strengths 0 through 5]({RAW}assets/{name}-case{case}.jpg)\n\n'
            request=next(s for s in samples if s['variation']==name and s['case']==case)['payload']
            card+='<details><summary>Prompt and seed</summary>\n\n'+request['prompt']+f"\n\nSeed: `{request['seed']}`.\n\n</details>\n\n"
        card+='</details>\n\n'
    card+='</details>\n\n'
    card+='Strength **1** is the trained endpoint. Higher strengths extrapolate and can change clothing, framing and scene details; occasional camera-frame marks are visible in the unfiltered examples. These development samples illustrate behavior, not a final-test quality benchmark.\n\n'
    card+='## Get the adapters\n\n| Slider | Original particles | Distilled LoRA for ComfyUI | Distilled LoRA for native runtime |\n|---|---|---|---|\n'
    for e in entries:
        card+=f"| {e['label']} | [Download]({RAW}{e['particle']}?download=true) | [Download]({RAW}{e['comfyui_lora']}?download=true) | [Download]({RAW}{e['native_lora']}?download=true) |\n"
    card+=f'''
**Particles:** use the [ComfyUI particle plugin]({GITHUB}#comfyui), or the [native loader]({GITHUB}/blob/main/REPRODUCE.md). [Plugin ZIP]({RAW}comfyui/anima-concept-sliders.zip?download=true) · [ComfyUI setup]({GITHUB}/blob/main/COMFYUI.md).

**Distilled LoRAs:** use standard **Load LoRA**, MODEL strength 1 and CLIP strength 0. These rank-8 linear approximations need no particle plugin. Compare their images below before choosing a format.

[Public source and training reproduction]({GITHUB}) · [Every sample and checkpoint](catalog.json) · [File hashes](release-manifest.json) · [Related YuE2 release](https://huggingface.co/ntc-ai/yue2-concept-sliders).

The reusable algorithms come from the [shared core in sliders-conceptmod](https://github.com/mikkel/sliders-conceptmod/tree/main/packages/concept-slider-core). This Anima repository pins an exact core revision and owns the model integration, recipe and release evidence.

## Distilled comparisons

<details>
<summary>Compare particles and distills at strengths 1, 3 and 5</summary>

Top row: original particles at strengths **1, 3, 5**. Bottom row: the distilled ordinary LoRA at the same strengths. Prompt, seed, dimensions and sampler are identical. [All four prompt comparisons]({WEB}/tree/main/distilled/samples).

'''
    for e in entries:
        name=e['id'];case=0
        images=[out/f'samples/{name}/case{case}/str{s}.png' for s in (1,3,5)]
        images += [out/f'distilled/samples/{name}-case{case}-str{s}.png' for s in (1,3,5)]
        dest=out/f'assets/{name}-distilled.jpg'
        grid(images,dest,[f'{fmt} · {s}' for fmt in ('Particles','Distilled') for s in (1,3,5)])
        card+=f"### {e['label']} — particles and distilled\n\n![{e['label']} particle versus distilled LoRA at strengths 1, 3 and 5]({RAW}assets/{name}-distilled.jpg)\n\n"
    card+='</details>\n\n'
    card+='The featured comparisons show a milder lighting change from the distills at the same strength. They are experimental approximations, not interchangeable replacements.\n\nThe distills were fitted on training activations. Development projection-relative MSE (lower is closer to the particle delta; **not** an image-quality score):\n\n| Slider | Relative MSE |\n|---|---:|\n'
    distill_summary=[]
    for e in entries:
        result=json.loads((out/f"distilled/{e['id']}-evaluation.json").read_text())
        metric=result['heldout_projection_relative_mse']
        card+=f"| {e['label']} | {metric:.6f} |\n"
        distill_summary.append(dict(variation=e['id'],relative_mse=metric))
    card+='\n[Distillation formulation and reproduction](DISTILLATION.md).\n\n'
    card+=(ROOT/'FORMULATION.md').read_text().replace('\n## ','\n### ').replace('# How the Anima sliders learn','## How the sliders learn',1)
    card+='\n## License\n\nThe adapters are derivatives of [CircleStone Anima](https://huggingface.co/circlestone-labs/Anima) and are distributed under the [CircleStone Labs Non-Commercial License](ANIMA-LICENSE.md). Rights are granted directly by CircleStone Labs LLC. See [NOTICE](NOTICE). Independently authored source code is MIT; vendored reference code retains its included license. This is an independent ntc-ai release.\n'
    yaml='''---
license: other
license_name: circlestone-labs-non-commercial-license
license_link: https://huggingface.co/ntc-ai/anima-concept-sliders/blob/main/ANIMA-LICENSE.md
base_model: circlestone-labs/Anima
base_model_relation: adapter
pipeline_tag: text-to-image
language:
- en
tags:
- anima
- concept-sliders
- particle-adapter
- lora
- comfyui
---
'''
    (out/'README.md').write_text(yaml+card)
    # The same samples-first card is useful from GitHub; Hub-local paths become absolute.
    github=card
    for local in ('catalog.json','release-manifest.json'):
        github=github.replace(']('+local+')',']('+WEB+'/blob/main/'+local+')')
    github=github.replace('## Get the adapters','<a id="comfyui"></a>\n\n## Get the adapters')
    (ROOT/'README.md').write_text(github)
    for name in ('FORMULATION.md','DISTILLATION.md','REPRODUCE.md','COMFYUI.md','ANIMA-LICENSE.md','NOTICE','LICENSE','core.lock.json','requirements.txt'):
        copy(ROOT/name,out/name)
    for path in (ROOT/'validation').glob('*.json'):copy(path,out/'validation'/path.name)
    copy(ROOT/'data/release-samples.json',out/'data/release-samples.json')
    for name in ('train','dev'):copy(ROOT/f'data/{name}.json',out/f'data/{name}.json')
    (out/'pending-samples.json').unlink(missing_ok=True)
    # Package only the files required to import the ComfyUI plugin.
    package_plugin(out)
    print('Built card, grids, documentation and plugin ZIP')

if __name__=='__main__':main()
