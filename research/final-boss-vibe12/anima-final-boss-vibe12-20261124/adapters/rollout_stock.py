"""Replace the positive path with a fresh soft-foreground RAW192 adapter rollout."""
from contextlib import contextmanager
from pathlib import Path

import torch
from lumen_studio.cache import TargetCache, save_tensor_file
from lumen_studio.contracts import atomic_json, digest, file_hash

from adapters.spatial_stock import subject_matte, validated_parent
from control import CACHE, ROOT, STORAGE
from optimization_worker import gains, guarded, pin, read
from replay import loaded
from shared_data import equal


@contextmanager
def frozen_adapter(runtime, raw):
    mixer = runtime.mixer
    handles = tuple(mixer.handles)
    strengths = mixer.strengths.copy()
    with torch.random.fork_rng(devices=[]):
        name = loaded(runtime, raw, 192)
    mixer.adapters[name].eval().requires_grad_(False)
    try:
        yield name
    finally:
        mixer.adapters.pop(name)
        mixer.strengths = strengths
        for handle in tuple(mixer.handles):
            if handle not in handles:
                handle.remove()
                mixer.handles.remove(handle)


def source_caches(runtime, entry):
    base = TargetCache(CACHE, verify=True)
    soft = TargetCache(STORAGE/'targets/soft-floor45', verify=True)
    donor = read(Path(entry['donor']))
    if (len(base)!=240 or len(soft)!=240 or soft.index['identity']['parent']!=base.index['fingerprint']
            or donor['cache']!=soft.index['fingerprint'] or donor['step']!=192 or donor['ema_decay'] is not None
            or runtime.identity!=base.index['identity']['model']):
        raise ValueError('Rollout requires the newly retrained soft foreground native RAW192 donor')
    return base, soft, donor


def norm_matched(neutral, positive, weight, reference):
    edit=(positive-neutral)*weight
    norm=(reference['positive']-reference['neutral']).to(neutral.device).norm()
    measured=edit.norm()
    if measured.item()<=1e-10 or norm.item()<=1e-10:
        raise ValueError('Degenerate rollout contrast')
    return neutral+edit*(norm/measured)


def capture(runtime, parent, inherited, key, schedule, masks):
    row=parent['row']; scene=row['id']; seed=parent['seed']
    scheduler=runtime.scheduler(10)
    latent=runtime.noise(seed,512,512)
    neutral_prompt, positive_prompt = (runtime.encode(row[side]) for side in ('neutral','positive'))
    shape=latent.shape[-2:]
    weight=.45+.55*subject_matte(masks[scene]['path'],shape,runtime.device)
    if (not torch.equal(latent.cpu(),parent['records'][0]['latent']) or not equal(parent['row'], inherited['row'])
            or not torch.equal(neutral_prompt.cpu(),parent['embedding']) or not torch.equal(positive_prompt.cpu(),parent['positive_embedding'])):
        raise ValueError('Rollout initial state or embedding differs from the frozen pair')
    records=list(inherited['records'][:10]); changed=0
    for position,(step,gain) in enumerate(zip(scheduler.timesteps,schedule,strict=True)):
        guarded()
        base=parent['records'][10+position]
        with runtime.mixer.scales({}):
            neutral=runtime.predict(latent,step,neutral_prompt)
            positive=runtime.predict(latent,step,positive_prompt)
        target=norm_matched(neutral,positive,weight,base)
        on=base | dict(latent=latent.cpu().clone(),neutral=neutral.cpu(),positive=target.cpu())
        if not torch.isfinite(target).all() or abs((target-neutral).norm().item()/(base['positive']-base['neutral']).norm().item()-1)>2e-5:
            raise ValueError('Rollout edit changed its same-position dose')
        if position==0 and any(not torch.allclose(on[name],inherited['records'][10][name],rtol=1e-6,atol=1e-7)
                              for name in ('latent','neutral','positive')):
            raise ValueError('The first rollout response differs at the identical initial state')
        changed+=int(not torch.equal(on['latent'],base['latent']))
        records.append(on)
        with runtime.mixer.scales({key:gain}):
            velocity=runtime.predict(latent,step,neutral_prompt)
        latent=runtime.step(scheduler,velocity,step,latent)
        if not torch.isfinite(latent).all():
            raise ValueError('Nonfinite native rollout state')
    if not changed:
        raise ValueError('Adapter rollout never diverged from the stored positive trajectory')
    return records


@torch.no_grad()
def prepare_cache(runtime, method, entry):
    base, soft, donor = source_caches(runtime,entry)
    masks=read(ROOT/'mattes/index.json')['mattes']
    schedule=gains(runtime,2/3,'early')
    identity=base.index['identity'] | dict(parent=base.index['fingerprint'],layout='native-adapter-rollout-vibe12/v1',
        soft_floor=soft.index['fingerprint'],recipe=pin(ROOT/'ready'/f'{method}.json'), donor=donor['raw'],
        gains=schedule,sigmas=runtime.scheduler(10).sigmas.tolist())
    directory=STORAGE/'targets'/method
    if (directory/'index.json').exists():
        result=TargetCache(directory,verify=True)
        if result.index['identity']!=identity:
            raise ValueError('Pinned rollout donor or schedule changed')
        return result
    directory.mkdir(exist_ok=True)
    fingerprint=digest(identity)
    shards=[]
    with frozen_adapter(runtime, donor['raw']) as key:
        for item, prior in zip(base.index['shards'],soft.index['shards'],strict=True):
            parent=validated_parent(base,item)
            inherited=validated_parent(soft,prior)
            path=directory/item['path']
            if path.exists():
                data=torch.load(path,map_location='cpu',weights_only=True)
                if data['fingerprint']!=fingerprint or data['row']!=parent['row'] or len(data['records'])!=20:
                    raise ValueError('Partial rollout shard changed')
            else:
                fields=capture(runtime,parent,inherited,key,schedule,masks)
                save_tensor_file(path,parent | dict(fingerprint=fingerprint,records=fields))
            shards.append(item | {'sha256':file_hash(path)})
    atomic_json(directory/'index.json',dict(identity=identity,input_fingerprint=fingerprint,shards=shards,
                 fingerprint=digest(dict(input=fingerprint,shards=shards))))
    return TargetCache(directory,verify=True)


@torch.no_grad()
def parity(runtime, method, cache, request, entry):
    base, soft, donor = source_caches(runtime,entry)
    if request!=dict(model_identity=runtime.identity,cache_fingerprint=cache.index['fingerprint'],
                     response_shape=list(cache[0]['positive'].shape)):
        return False
    masks=read(ROOT/'mattes/index.json')['mattes']
    checks=[]
    with frozen_adapter(runtime,donor['raw']) as key:
        for j in (0,5,11):
            item=base.index['shards'][j]
            expected=capture(runtime,validated_parent(base,item),validated_parent(soft,soft.index['shards'][j]),
                             key,cache.index['identity']['gains'],masks)
            actual=validated_parent(cache,cache.index['shards'][j])
            checks.append(dict(scene=item['row'],passed=equal(expected,actual['records'])))
    passed=all(row['passed'] for row in checks)
    atomic_json(ROOT/'cohorts'/method/'native-parity.json',dict(passed=passed,request=request,fields=checks))
    return passed
