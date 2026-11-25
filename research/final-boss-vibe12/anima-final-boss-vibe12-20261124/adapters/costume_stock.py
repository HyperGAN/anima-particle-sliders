"""Re-run the qualified-costume and soft-foreground mixture on the shared scenes."""
from importlib.util import module_from_spec, spec_from_file_location
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from lumen_studio.cache import TargetCache, save_tensor_file
from lumen_studio.contracts import atomic_json, digest, file_hash

from adapters.seed_cross_stock import FINAL_BOSS, judge, qualify, vibe_observer, vibe_portfolio, vibe_teacher_checks, vibe_teacher_selection
from control import CACHE, MANIFEST, ROOT, STORAGE
from optimization_worker import guarded, pin, read


def target_source(path):
    spec = spec_from_file_location('lumen_studio.archived_final_boss_target', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(runtime, module, row, recipe, folder):
    key = digest(dict(seed=row['seeds'][0], recipe=recipe, model=runtime.identity))
    destination = folder/'images'/f'{key}.png'
    receipt = destination.with_suffix('.json')
    if receipt.is_file():
        saved = read(receipt)
        if saved['sha256'] != file_hash(destination):
            raise ValueError('Saved costume teacher render changed')
        return saved
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as work:
        path = Path(work)/'image.png'
        module.TargetField(runtime,recipe).render(row['seeds'][0],512,512).save(path)
        path.replace(destination)
    saved = dict(path=str(destination), sha256=file_hash(destination), seed=row['seeds'][0],
                 prompt=recipe['targets'][0], request_id=key)
    atomic_json(receipt,saved)
    return saved


def capture(runtime, module, row, recipe):
    field = module.TargetField(runtime, recipe)
    initial = runtime.noise(row['seeds'][0],512,512)
    records = []
    for track in ('neutral','positive'):
        scheduler = runtime.scheduler(10)
        latent = initial.clone()
        for position, step in enumerate(scheduler.timesteps):
            guarded()
            neutral = runtime.predict(latent,step,field.neutral)
            target = field.velocity(latent,step,position,neutral=neutral)
            records.append(dict(position=position,timestep=float(step),trajectory=track,latent=latent.cpu().clone(),
                                neutral=neutral.cpu(),positive=target.cpu()))
            latent = runtime.step(scheduler,neutral if track=='neutral' else target,step,latent)
    return records


@torch.no_grad()
def prepare_cache(runtime, method, entry):
    base = TargetCache(CACHE, verify=True)
    floor = TargetCache(STORAGE/'targets/soft-floor45', verify=True)
    if floor.index['identity']['parent'] != base.index['fingerprint'] or len(floor)!=240 or floor.index['identity']['model']!=runtime.identity:
        raise ValueError('Expected soft foreground targets from the new teacher cohort')
    expected = pin(ROOT/'ready'/f'{method}.json')
    output = STORAGE/'targets'/method
    if (output/'index.json').exists():
        cache = TargetCache(output,verify=True)
        if cache.index['identity']['recipe'] != expected:
            raise ValueError('Archived qualified-costume recipe changed')
        return cache
    manifest = read(MANIFEST)
    staging = STORAGE/'costume-teachers'
    fixture = read(ROOT/'prompt-variants.json')['rows']
    module = target_source(Path(entry['target_source']))
    readers=partial(judge,staging,manifest['collection_identity'])
    helpers=dict(observer=vibe_observer, portfolio=vibe_portfolio, checks=vibe_teacher_checks, digest=digest,
                 selection=SimpleNamespace(qualifies=partial(vibe_teacher_selection.qualifies,max_neutral_score=.35)))
    selected = {}
    with runtime.mixer.scales({}):
        for row in manifest['rows']:
            key = row['id']
            cues = fixture[key]
            neutral = row['selection']['neutral']['output']
            if file_hash(neutral['path']) != neutral['sha256']:
                raise ValueError('The original neutral image changed')
            variants = [('replace', cues['in_scene'], module.DIRECT), ('early', cues['in_scene'], module.EARLY),
                        ('explicit-early', cues['explicit'], module.EARLY), ('matched-replace', row['positive'],module.DIRECT)]
            qualified = []
            for label,prompt,gains in variants:
                guarded()
                recipe = dict(method=label, neutral=row['neutral'], targets=[prompt], gains=list(gains), edge_floor=None)
                positive = (row['selection']['positive']['output'] if label=='matched-replace' else render(runtime,module,row,recipe,staging))
                if file_hash(positive['path']) != positive['sha256']:
                    raise ValueError('The costume target image changed')
                score = qualify.measure_candidate(readers,helpers,FINAL_BOSS,key,row['seeds'][0],dict(neutral=neutral,positive=positive))
                atomic_json(staging/'qualifications'/f'{key}-{label}.json',score)
                if score['qualified']:
                    qualified.append((score['finish_gain'],score['second_read']['finish_gain'],recipe,score))
            if qualified:
                selected[key] = max(qualified,key=lambda item:item[:2])
        winners = sorted(selected,key=lambda key:(-selected[key][0],-selected[key][1],key))[:entry['costume_scenes']]
        if len(winners) != entry['costume_scenes']:
            raise RuntimeError('Jev did not qualify eight costume teacher scenes')
        evidence = {key:dict(recipe=selected[key][2],qualification=selected[key][3]) for key in winners}
        atomic_json(staging/'selection.json',evidence)
        identity = base.index['identity'] | dict(parent=base.index['fingerprint'], layout='qualified-costume-with-soft-floor-vibe12/v1',
                    recipe=expected, selection=pin(staging/'selection.json'), soft_floor=floor.index['fingerprint'])
        fingerprint = digest(identity)
        output.mkdir(exist_ok=True)
        shards=[]
        for item,origin in zip(floor.index['shards'],manifest['rows'],strict=True):
            key=item['row']
            data=torch.load(floor.directory/item['path'],map_location='cpu',weights_only=True)
            if data['row'] != origin:
                raise ValueError('Frozen soft-foreground context differs')
            if key in evidence:
                recipe=evidence[key]['recipe']
                data['records']=capture(runtime,module,origin,recipe)
                data['positive_embedding']=runtime.encode(recipe['targets'][0]).cpu()
            path = output/item['path']
            save_tensor_file(path,data | dict(fingerprint=fingerprint))
            shards.append(item | {'sha256':file_hash(path)})
    atomic_json(output/'index.json',dict(identity=identity,input_fingerprint=fingerprint,shards=shards,
                 fingerprint=digest(dict(input=fingerprint,shards=shards))))
    return TargetCache(output,verify=True)


@torch.no_grad()
def parity(runtime, method, cache, request, entry):
    from optimization_targets import apply_target
    from adapters.spatial_stock import subject_matte
    if request != dict(model_identity=runtime.identity,cache_fingerprint=cache.index['fingerprint'],
                       response_shape=list(cache[0]['positive'].shape)):
        return False
    shape = cache[0]['latent'].shape[-2:]
    mattes=read(ROOT/'mattes/index.json')['mattes']
    selection_file=cache.index['identity']['selection']
    if pin(Path(selection_file['path']))!=selection_file:
        raise ValueError('Costume teacher evidence changed')
    evidence=read(Path(selection_file['path']))
    module=target_source(Path(entry['target_source']))
    checks=[]
    with runtime.mixer.scales({}):
        for j,item in enumerate(cache.index['shards']):
            guarded()
            row=cache[j*20+(j*7+2)%20]
            label=item['row']; latent=row['latent'].to(runtime.device); step=row['timestep']
            embed=runtime.encode(row['row']['neutral'])
            neutral=runtime.predict(latent,step,embed)
            if label in evidence:
                field=module.TargetField(runtime,evidence[label]['recipe'])
                target=field.velocity(latent,step,row['position'],neutral=neutral)
            else:
                positive=runtime.predict(latent,step,runtime.encode(row['row']['positive']))
                weight=.45+.55*subject_matte(mattes[label]['path'],shape,runtime.device)
                target=apply_target(dict(weight=weight,blend=0.),neutral,positive,None)[0]
            checks.append(dict(scene=label, embedding=torch.equal(embed.cpu(),row['embedding']),
                neutral=torch.equal(neutral.cpu(),row['neutral']),target=torch.equal(target.cpu(),row['positive'])))
    passed=all(c['embedding'] and c['neutral'] and c['target'] for c in checks)
    atomic_json(ROOT/'cohorts'/method/'native-parity.json',dict(passed=passed,request=request,fields=checks))
    return passed
