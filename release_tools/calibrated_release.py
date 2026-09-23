"""Stage immutable calibrated exports and regenerate matched strength-one examples."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DISPLAY_ORDER = ['bad-intent','final-form','afterimage','moonlit','candlelit','dusk','opal-fever']
sys.path.insert(0, str(ROOT))
from release_tools.build import comparison, copy, sha, RAW, WEB
from release_tools.rescale_lora_alpha import rescale_alpha
from lumen_studio.alpha_adapter import read_metadata, export_with_alpha


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def stage(folder, studio):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    catalog = json.loads((folder/'catalog.json').read_text())
    inventory = json.loads((studio/'exports/alpha-unit-v1/exports.json').read_text())
    balance_path = ROOT/'data/bad-intent-balance.json'
    balance = json.loads(balance_path.read_text()) if balance_path.exists() else None
    for entry in catalog['sliders']:
        name = entry['id']
        if name not in ('bad-intent','candlelit','moonlit'):
            continue  # Additional releases are staged from their own audited selections.
        if 'archived_release' not in entry:
            entry['archived_release'] = {k: entry[k] for k in
                ('particle', 'sha256', 'native_lora', 'comfyui_lora', 'samples', 'featured_comparison') if k in entry}
        old = entry['archived_release']
        selected = dict(inventory['uncanny' if name == 'bad-intent' else name])
        balanced = name == 'bad-intent' and balance is not None
        if balanced:
            selected['alpha'] = balance['particle_alpha']
            particle_dest = folder/'weights/bad-intent-balanced-alpha.safetensors'
            source = studio/'runs/uncanny/ema-001600.safetensors'
            assert sha(source) == selected['source_sha256']
            if not particle_dest.exists():
                export_with_alpha(source,particle_dest,alpha=selected['alpha'],
                    provenance=dict(method='matched-intensity-development-sweep',selection=balance))
            selected.update(path=str(particle_dest),sha256=sha(particle_dest))
            if 'unit_alpha_release' not in entry:
                entry['unit_alpha_release'] = {k:entry[k] for k in
                    ('particle','sha256','native_lora','comfyui_lora','particle_alpha','lora_alpha','samples','distilled_samples','comparisons','featured_comparison')}
        particle_path = Path(selected['path'])
        assert sha(particle_path) == selected['sha256']
        metadata = read_metadata(particle_path)
        old_state, new_state = load_file(str(folder/old['particle'])), load_file(str(particle_path))
        assert {k for k in old_state if not k.endswith('.alpha')} == {k for k in new_state if not k.endswith('.alpha')}
        assert all(torch.equal(v, new_state[k]) for k, v in old_state.items() if not k.endswith('.alpha'))
        entry['particle'] = ('weights/bad-intent-balanced-alpha.safetensors' if balanced else
                             old['particle'] if name == 'bad-intent' else f'weights/{name}-unit-alpha.safetensors')
        copy(particle_path, folder/entry['particle'])
        write_json((folder/entry['particle']).with_suffix('.json'), metadata)
        entry.update(sha256=selected['sha256'], rank=8, particle_alpha=selected['alpha'], recommended_strength=1)
        for fmt, key in [('native', 'native_lora'), ('comfyui', 'comfyui_lora')]:
            lora_alpha = balance['lora_alpha'] if balanced else 24. if name == 'bad-intent' else selected['alpha']
            suffix = 'balanced-alpha' if balanced and lora_alpha != 24 else 'unit-alpha'
            entry[key] = f'distilled/{fmt}/{name}-{suffix}.safetensors'
            source, dest = folder/old[key], folder/entry[key]
            gain = lora_alpha/8.
            if not dest.exists():
                rescale_alpha(source, dest, gain)
            before, after = load_file(str(source)), load_file(str(dest))
            assert {k for k in before if not k.endswith('.alpha')} == {k for k in after if not k.endswith('.alpha')}
            assert all(torch.equal(v, after[k]) for k, v in before.items() if not k.endswith('.alpha'))
            assert {float(v) for k, v in after.items() if k.endswith('.alpha')} == {8.*gain}
            entry[key+'_sha256'] = sha(dest)
        entry['lora_alpha'] = lora_alpha
        if balanced:entry['balance_selection'] = balance
        requests = []
        if name == 'bad-intent':
            requests = [dict(case=f"case{s['case']}", **{k:s['payload'][k] for k in
                ('prompt', 'seed', 'width', 'height', 'steps')}) for s in
                json.loads((ROOT/'data/bad-intent-samples.json').read_text()) if s['strength'] == 1]
        else:
            featured = json.loads((folder/f'samples/featured-portrait/{name}-particle.json').read_text())
            requests.append(dict(case='portrait', **{k:featured[k] for k in ('prompt','seed','width','height','steps')}))
            requests += [dict(case=f"case{s['case']}", **{k:s['payload'][k] for k in
                ('prompt','seed','width','height','steps')}) for s in
                json.loads((ROOT/'data/release-samples.json').read_text()) if s['variation'] == name and s['strength'] == 1]
        entry['render_requests'] = requests
        entry['samples'], entry['distilled_samples'], entry['comparisons'] = [], [], []
        for request in requests:
            case = request['case']
            samples = []
            for fmt, strength in [('particles',1),('lora',1),('off',0)]:
                version = 'balanced-alpha' if balanced else 'unit-alpha'
                path = f'samples/{version}/{name}/{case}/{fmt}.png'
                sample = dict(case=case, format=fmt, strength=strength, image=path, metadata=str(Path(path).with_suffix('.json')))
                samples.append(sample)
                entry['distilled_samples' if fmt == 'lora' else 'samples'].append(sample)
            entry['comparisons'].append(dict(case=case, asset=f'assets/{version}-{name}-{case}.jpg', samples=samples))
        refresh_path = ROOT/'data/featured-refresh.json'
        refresh = json.loads(refresh_path.read_text()) if refresh_path.exists() else None
        if refresh and name == refresh['slider']:
            if 'balanced_alpha_release' not in entry:
                entry['balanced_alpha_release'] = json.loads(json.dumps({k:entry[k] for k in
                    ('render_requests','samples','distilled_samples','comparisons')}))
            request = refresh['request']
            source_rows = json.loads((ROOT/refresh['source_manifest']).read_text())['rows']
            source = next(r for r in source_rows if r['id']==refresh['source_row_id'])
            assert source['split']=='dev' and source['character']==refresh['character']
            assert request['prompt']==source['neutral'] and request['seed'] in source['seeds']
            index = next(i for i,r in enumerate(requests) if r['case']==refresh['replaces'])
            requests[index] = request
            samples = []
            for fmt,strength in [('particles',1),('lora',1),('off',0)]:
                path = f"samples/{refresh['version']}/{name}/{request['case']}/{fmt}.png"
                samples.append(dict(case=request['case'],format=fmt,strength=strength,
                    image=path,metadata=str(Path(path).with_suffix('.json'))))
            entry['comparisons'][index] = dict(case=request['case'],
                asset=f"assets/{refresh['version']}-{name}-{request['case']}.jpg",samples=samples)
            entry['samples'] = [s for c in entry['comparisons'] for s in c['samples'] if s['format']!='lora']
            entry['distilled_samples'] = [s for c in entry['comparisons'] for s in c['samples'] if s['format']=='lora']
            entry['featured_refresh'] = refresh
        entry['featured_comparison'] = entry['comparisons'][0]
    catalog['calibration'] = 'unit-alpha-v2' if balance else 'unit-alpha-v1'
    from release_tools.refresh_examples import apply_refreshes
    apply_refreshes(catalog)
    write_json(folder/'catalog.json', catalog)
    return catalog


def render(folder, studio):
    import torch
    from lumen_studio.backends.anima import TurboRuntime
    from lumen_studio.alpha_adapter import AlphaParticleAdapter
    from single_particle import adapter_class
    from release_tools.distill import attach_lora
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    catalog = json.loads((folder/'catalog.json').read_text())
    runtime = TurboRuntime(studio/'model', 'cuda:0')
    try:
        for entry in catalog['sliders']:
            adapter = adapter_class(folder/entry['particle'])(runtime.transformer).to(runtime.device).eval().requires_grad_(False)
            adapter.load_export(folder/entry['particle'], model_identity=runtime.identity)
            runtime.mixer.add('particle', adapter)
            for fmt, strength in [('particles',1),('lora',1),('off',0)]:
                handles = attach_lora(runtime.transformer, folder/entry['native_lora'], 1.) if fmt == 'lora' else []
                try:
                    with runtime.mixer.scales({'particle': 1.} if fmt == 'particles' else {}):
                        for request in entry['render_requests']:
                            comp = next(c for c in entry['comparisons'] if c['case']==request['case'])
                            dest = folder/next(s['image'] for s in comp['samples'] if s['format']==fmt)
                            adapter_path = entry['particle'] if fmt == 'particles' else entry['native_lora'] if fmt == 'lora' else None
                            alpha = entry['particle_alpha'] if fmt == 'particles' else entry['lora_alpha'] if fmt == 'lora' else None
                            metadata = dict(request, variation=entry['id'], format=fmt, energy=strength, strength=strength,
                                cfg=1, split='dev', model_identity=runtime.identity, adapter=adapter_path,
                                adapter_sha256=sha(folder/adapter_path) if adapter_path else None,
                                alpha=alpha, rank=8 if adapter_path else None, postprocessing='none',
                                sampling='fixed prompt and seed; rendered from published-format export')
                            if dest.exists():
                                old = json.loads(dest.with_suffix('.json').read_text())
                                assert all(old[k] == v for k,v in metadata.items())
                                assert old['image_sha256'] == sha(dest)
                                continue
                            image = runtime.render(request['prompt'], request['seed'], request['width'],request['height'],request['steps'])
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            image.save(dest)
                            metadata['image_sha256'] = sha(dest)
                            write_json(dest.with_suffix('.json'), metadata)
                            print('rendered',dest.relative_to(folder),flush=True)
                finally:
                    for h in handles: h.remove()
            runtime.mixer.close()
            del adapter
    finally:
        runtime.close()


def build_card(folder):
    catalog = json.loads((folder/'catalog.json').read_text())
    entries = sorted(catalog['sliders'], key=lambda e: DISPLAY_ORDER.index(e['id']))
    old_card = (folder/'README.md').read_text()
    frontmatter = old_card[:old_card.index('# Anima Particle Sliders')]
    formulation = old_card[old_card.index('## How the sliders learn'):]
    formulation = formulation.replace('the gallery includes extrapolation through', 'the historical gallery included extrapolation through')
    note = 'Calibrated inference additionally multiplies the branch by its stored alpha/rank; the current gallery uses nominal strength 1. No training weights or losses change. '
    if note not in formulation:
        formulation = formulation.replace('For multiple sliders the deltas sum on the same input.', note+'For multiple sliders the deltas sum on the same input.')
    lines = ['# Anima Particle Sliders', '',
        '**'+', '.join(e['label'] for e in entries)+' for Anima Turbo v1.1.** Original Particle adapters and ordinary LoRA Distills, with strength stored in each file’s alpha.', '',
        '## Samples', '',
        '**Start at strength 1.0.** Every current comparison is **Particle 1 → Distill 1 → Off 0**, with the same prompt, seed, 768 × 768 resolution, 10 Euler steps and CFG 1. These are newly rendered examples from the calibrated files linked below.', '',
        'Bad Intent adds menacing expressions and close framing. Final Form adds supernatural transformations; Afterimage adds translucent, offset echoes of faces and gestures. Moonlit, Candlelit and Dusk change atmosphere. Distills are linear approximations of the original particles; their images need not match exactly.', '']
    def show(entry, comp, lead=False):
        for sample in comp['samples']:
            assert (folder/sample['image']).is_file()
        comparison([folder/s['image'] for s in comp['samples']],folder/comp['asset'])
        alt = f"{entry['label']}: Particle first, Distill second, Off third" if lead else f"{entry['label']}: matched strength-one samples, {comp['case']}"
        result = [f"![{alt}]({RAW}{comp['asset']})", '',
            'Full resolution: '+' · '.join(f"[{label}]({RAW}{s['image']})" for label,s in zip(('Particle','Distill','Off'),comp['samples'])), '',
            '<details><summary>Prompt and seed</summary>', '',
            next(r['prompt'] for r in entry['render_requests'] if r['case']==comp['case']), '',
            f"Seed: `{next(r['seed'] for r in entry['render_requests'] if r['case']==comp['case'])}`.", '', '</details>', '']
        return result
    for entry in entries:
        lines += [f"### {entry['label']}", '', '**Particle · strength 1** → **Distill · strength 1** → **Off · strength 0**', '']
        lines += show(entry,entry['featured_comparison'],True)
    lines += ['<details><summary>More freshly rendered strength-one comparisons</summary>', '']
    for entry in entries:
        for comp in entry['comparisons'][1:]:
            lines += [f"### {entry['label']} · {comp['case']}", '']+show(entry,comp)
    lines += ['</details>', '',
        'These are development examples, not a final-test benchmark. Strength 1 is the calibrated default; higher settings extrapolate and can change pose, clothing, framing or appearance.', '',
        '## Get the adapters', '',
        '| Slider | Original particles | Distilled LoRA for ComfyUI | Distilled LoRA for native runtime |',
        '|---|---|---|---|']
    for entry in entries:
        lines.append(f"| {entry['label']} | [Download]({RAW}{entry['particle']}?download=true) | [Download]({RAW}{entry['comfyui_lora']}?download=true) | [Download]({RAW}{entry['native_lora']}?download=true) |")
    lines += ['', '**Particles:** use the [ComfyUI particle plugin](https://github.com/HyperGAN/anima-particle-sliders#comfyui) at strength **1.0**. Update the plugin for embedded-alpha files. [Plugin ZIP]('+RAW+'comfyui/anima-particle-sliders.zip?download=true) · [Setup](https://github.com/HyperGAN/anima-particle-sliders/blob/main/COMFYUI.md).', '',
        '**Distilled LoRAs:** use standard **Load LoRA**, **MODEL strength 1.0, CLIP strength 0**. No particle plugin is needed.', '',
        '| Slider | Particle alpha | Distill alpha | Rank |', '|---|---:|---:|---:|']
    for entry in entries:
        lines.append(f"| {entry['label']} | {entry['particle_alpha']:.10g} | {entry['lora_alpha']:.10g} | 8 |")
    lines += ['', 'Calibration is embedded in the files; do not add an external gain. Bad Intent’s old alpha-8 LoRA needs strength 3 for the same operation as the new alpha-24 file at strength 1. Lighting LoRAs carry the same alpha gain as their calibrated particle teacher; this does not claim a new fit or exact image equivalence.', '',
        '[Source and reproduction](https://github.com/HyperGAN/anima-particle-sliders) · [Catalog](catalog.json) · [File hashes](release-manifest.json) · [Distillation and alpha audit](DISTILLATION.md).', '',
        'The reusable algorithms come from the [pinned particle-sliders core](https://github.com/HyperGAN/particle-sliders/tree/a119ca1ecd3d5d6c437065839d22739b04f2f4d8/packages/particle-sliders-core). Training follows `winning_formulation()` (gmix, provisional particle-gmix-1600-v2). The published weights used noise hold 1.0; the stamp hold ratio is 1.3.', '',
        '<details><summary>Historical exports and examples</summary>', '',
        'The original alpha-8 exports and their samples remain available for reproduction. Their strength numbers use the original scale and do not describe the new calibrated files. The catalog’s `archived_release` entries retain the original paths.', '',
        '[Original lighting/Bad Intent particle samples]('+WEB+'/tree/main/samples) · [Original distill samples]('+WEB+'/tree/main/distilled/samples).', '',
        'Original, uncalibrated projection-relative fitting MSE: Bad Intent 0.005434, Moonlit 0.007559, Candlelit 0.020457. These local diagnostics are not image-quality scores and do not measure the calibrated complete-denoiser edit.', '', '</details>', '', formulation]
    card='\n'.join(lines)
    (folder/'README.md').write_text(frontmatter+card)
    github=card.replace('](catalog.json)',']('+WEB+'/blob/main/catalog.json)').replace('](release-manifest.json)',']('+WEB+'/blob/main/release-manifest.json)')
    (ROOT/'README.md').write_text(github.replace('## Get the adapters','<a id="comfyui"></a>\n\n## Get the adapters'))


def verify(folder):
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from release_tools.distill import comfy_name
    torch.set_num_threads(4)
    catalog = json.loads((folder/'catalog.json').read_text())
    report = dict(passed=True, nominal_strength=1, final_test_used=False, samples=[], exports=[], zero_replays=[], new_baselines=[])
    for entry in catalog['sliders']:
        old = entry['archived_release']
        for key in ('particle','native_lora','comfyui_lora'):
            before, after = load_file(str(folder/old[key])), load_file(str(folder/entry[key]))
            assert {k for k in before if not k.endswith('.alpha')} == {k for k in after if not k.endswith('.alpha')}
            assert all(torch.equal(v,after[k]) for k,v in before.items() if not k.endswith('.alpha'))
            expected = entry['particle_alpha' if key=='particle' else 'lora_alpha']
            assert sum(k.endswith('.alpha') for k in after)==224
            assert all(float(v)==expected for k,v in after.items() if k.endswith('.alpha'))
            report['exports'].append(dict(slider=entry['id'],format=key,path=entry[key],sha256=sha(folder/entry[key]),
                source=old[key],source_sha256=sha(folder/old[key]),alpha=expected,rank=8,learned_tensors_unchanged=True))
        native, comfy = [load_file(str(folder/entry[k])) for k in ('native_lora','comfyui_lora')]
        for k,v in native.items():
            if not k.endswith('.lora_A.weight'):continue
            name=k.removesuffix('.lora_A.weight');prefix='diffusion_model.'+comfy_name(name)
            for a,b in (('.lora_A.weight','.lora_down.weight'),('.lora_B.weight','.lora_up.weight'),('.alpha','.alpha')):
                assert torch.equal(native[name+a],comfy[prefix+b])
        for comp,request in zip(entry['comparisons'],entry['render_requests']):
            records=[]
            for sample in comp['samples']:
                path=folder/sample['image'];meta=json.loads((folder/sample['metadata']).read_text())
                assert meta['image_sha256']==sha(path)
                assert meta['strength']==meta['energy']==sample['strength']
                assert meta['cfg']==1 and meta['split']=='dev'
                assert all(meta[k]==request[k] for k in ('prompt','seed','width','height','steps'))
                expected_path=entry['particle'] if sample['format']=='particles' else entry['native_lora'] if sample['format']=='lora' else None
                assert meta['adapter']==expected_path
                assert meta['adapter_sha256']==(sha(folder/expected_path) if expected_path else None)
                if expected_path:assert meta['alpha']==entry['particle_alpha' if sample['format']=='particles' else 'lora_alpha']
                with Image.open(path) as image:assert image.size==(768,768) and np.array(image).std()>1
                records.append(meta)
                report['samples'].append(dict(path=sample['image'],metadata=sample['metadata'],sha256=sha(path),
                    adapter_sha256=meta['adapter_sha256'],strength=meta['strength'],alpha=meta['alpha']))
            assert records[0]['model_identity']==records[1]['model_identity']==records[2]['model_identity']
            previous = (folder/entry['off_references'][comp['case']] if comp['case'] in entry.get('off_references',{}) else
                        folder/'samples/featured-portrait/off.png' if comp['case']=='portrait' else
                        folder/f"samples/{entry['id']}/{comp['case']}/str0.png")
            if previous.exists():
                assert np.array_equal(np.array(Image.open(previous)),np.array(Image.open(folder/comp['samples'][2]['image'])))
                report['zero_replays'].append(dict(slider=entry['id'],case=comp['case'],pixel_exact=True))
            else:
                refresh = entry.get('example_refreshes',{}).get(comp['case'], entry.get('featured_refresh'))
                assert request==refresh['request'] and records[2]['adapter'] is None
                report['new_baselines'].append(dict(slider=entry['id'],case=comp['case'],
                    source_manifest=refresh['source_manifest'],source_row_id=refresh['source_row_id'],
                    image=comp['samples'][2]['image'],sha256=records[2]['image_sha256']))
    report.update(images=len(report['samples']),comparisons=sum(len(e['comparisons']) for e in catalog['sliders']),
                  particle_alphas={e['id']:e['particle_alpha'] for e in catalog['sliders']},
                  lora_alphas={e['id']:e['lora_alpha'] for e in catalog['sliders']},
                  limitation='Development examples rendered in the pinned native runtime; full ComfyUI GPU generation is not covered.')
    write_json(ROOT/'validation/unit-alpha.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','exports','zero_replays')},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['stage','render','build','verify'])
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--studio',type=Path,required=True)
    args=p.parse_args()
    if args.action=='stage':stage(args.folder,args.studio)
    elif args.action=='build':build_card(args.folder)
    elif args.action=='verify':verify(args.folder)
    else:
        from lumen_studio.coordinator import GpuLease
        with GpuLease(Path('/tmp'),os.environ['ANIMA_GPU_LEASE']):render(args.folder,args.studio)
