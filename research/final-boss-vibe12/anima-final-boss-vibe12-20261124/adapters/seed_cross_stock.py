"""Cross candidate B seeds into the original twenty slots without changing the teacher law."""
import sys
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from lumen_studio.cache import TargetCache, save_tensor_file
from lumen_studio.contracts import atomic_json, digest, file_hash

from control import CACHE, MANIFEST, ROOT, SOURCE, STORAGE
from optimization_worker import guarded, pin, read
from shared_data import COORDINATES, equal, parity as shared_parity


OLD = ROOT.parent / 'anima-final-boss-shared-seed-split-20261124'
DEPS = Path('/mnt/nvme/sliders/experiments/anima-final-boss-shared-seed-split-20261124/vibe-deps')
search_path = sys.path.copy()
try:
    sys.path[:0] = [str(OLD), str(DEPS)]
    import qualify
    import seed_split as previous
    from particle_sliders_vibe import _call
finally:
    sys.path[:] = search_path
sys.path.append(str(DEPS))
import lumen_studio
lumen_studio.__path__.insert(0, str(SOURCE/'lumen_studio'))
from lumen_studio import vibe_observer, vibe_portfolio, vibe_teacher_checks, vibe_teacher_selection
from lumen_studio.vibe_profiles import FINAL_BOSS
lumen_studio.__path__.pop(0)


def image_record(runtime, control, row, side, seed):
    text = row[side]
    destination = control/'images'/row['id']/str(seed)/f'{side}.png'
    signature = destination.with_suffix('.json')
    if signature.is_file():
        saved = read(signature)
        if saved['sha256'] != file_hash(destination) or saved['prompt'] != text or saved['seed'] != seed:
            raise ValueError('Saved second-seed render changed')
        return saved
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary)/'image.png'
        runtime.render(text, seed, width=512, height=512, steps=10).save(staged)
        staged.replace(destination)
    output = dict(path=str(destination), sha256=file_hash(destination), prompt=text, seed=seed,
                  request_id=digest(dict(prompt=text, seed=seed)))
    atomic_json(signature, output)
    return output


def judge(root, identity, images, state, questions):
    inputs = dict(provider=identity['provider'], model=identity['judge'], state=state,
                  questions=questions, images=[image['sha256'] for image in images])
    key = digest(dict(kind='judge', inputs=inputs))
    receipt = _call(root, 'judge', inputs, lambda _: vibe_observer.observe(
        inputs['provider'], inputs['model'], [image['path'] for image in images], state, questions))
    verified = vibe_observer.decision_values(receipt['raw'], questions)
    if verified != receipt['values']:
        raise ValueError('The seed qualifier no longer matches the archived raw response')
    return verified, key


def capture(runtime, row, seed):
    neutral, positive = (runtime.encode(row[side]) for side in ('neutral','positive'))
    initial = runtime.noise(seed, 512, 512)
    records = []
    first = None
    for trajectory in ('neutral','positive'):
        latent = initial.clone()
        scheduler = runtime.scheduler(10)
        for position, step in enumerate(scheduler.timesteps):
            guarded()
            if trajectory == 'positive' and position == 0:
                vn, vp = first
            else:
                vn, vp = runtime.predict(latent, step, neutral), runtime.predict(latent, step, positive)
                if position == 0:
                    first = vn, vp
            records.append(dict(position=position, timestep=float(step), trajectory=trajectory,
                                latent=latent.cpu().clone(), neutral=vn.cpu(), positive=vp.cpu()))
            latent = runtime.step(scheduler, vn if trajectory=='neutral' else vp, step, latent)
    return dict(row=row, seed=seed, embedding=neutral.cpu(), positive_embedding=positive.cpu(), records=records)


def edit_norms(data):
    return {(record['trajectory'], record['position']):
            (record['positive'].double()-record['neutral'].double()).norm().item() for record in data['records']}


@torch.no_grad()
def prepare_cache(runtime, method, entry):
    base = TargetCache(CACHE, verify=True)
    manifest = read(MANIFEST)
    if len(base)!=240 or base.index['identity']['model'] != runtime.identity:
        raise ValueError('Wrong frozen base for seed split')
    directory = STORAGE/'targets'/method
    if (directory/'index.json').exists():
        saved=TargetCache(directory, verify=True)
        if saved.index['identity']['recipe'] != pin(ROOT/'ready'/f'{method}.json'):
            raise ValueError('Seed split settings changed')
        return saved
    directory.mkdir(exist_ok=True)
    staging = STORAGE/'seed-cross'
    staging.mkdir(exist_ok=True)
    order = [row['id'] for row in manifest['rows']]
    original, a_norms, b_norms = {}, {}, {}
    readers = partial(judge, staging, manifest['collection_identity'])
    modules = dict(observer=vibe_observer, portfolio=vibe_portfolio, checks=vibe_teacher_checks, digest=digest,
                   selection=SimpleNamespace(qualifies=partial(vibe_teacher_selection.qualifies, max_neutral_score=.35)))
    choices = {row_id: [] for row_id in order}
    previous.MINIMUM_SPLIT_SCENES = entry['minimum_scenes']
    for item in base.index['shards']:
        a_data = torch.load(CACHE/item['path'], map_location='cpu', weights_only=True)
        original[item['row']] = a_data
        a_norms[item['row']] = edit_norms(a_data)
    with runtime.mixer.scales({}):
        for offset in entry['seed_offsets']:
            for row_id in order:
                guarded()
                a_data = original[row_id]
                row, seed = a_data['row'], a_data['seed'] + offset
                outputs = {side:image_record(runtime,staging,row,side,seed) for side in ('neutral','positive')}
                scored = qualify.measure_candidate(readers,modules,FINAL_BOSS,row_id,seed,outputs)
                atomic_json(staging/'qualifications'/f'{row_id}-{seed}.json',scored)
                if not scored['qualified']:
                    continue
                path=staging/'fields'/f'{row_id}-{seed}.pt'
                path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    save_tensor_file(path,capture(runtime,row,seed))
                data=torch.load(path,map_location='cpu',weights_only=True)
                if data['row']!=row or data['seed']!=seed or not all(equal(data[k],a_data[k]) for k in ('embedding','positive_embedding')):
                    raise ValueError('Second-seed native conditioning changed')
                choices[row_id].append(seed)
                b_norms[row_id,seed]=edit_norms(data)
            verdict=qualify.g3_select(order,choices,a_norms,b_norms)
            atomic_json(staging/'seed-gate.json',verdict)
            if verdict['status']=='pass':
                break
    if verdict['status']!='pass':
        raise RuntimeError('No second-seed selection passed Jev and the inherited edit-norm gate')
    identity=base.index['identity'] | dict(parent=base.index['fingerprint'], layout='crossed-native-seeds-vibe12/v1',
                recipe=pin(ROOT/'ready'/f'{method}.json'), seed_gate=pin(staging/'seed-gate.json'))
    fingerprint=digest(identity)
    shards=[]
    for item in base.index['shards']:
        row_id=item['row']; seed=verdict['seed_b'][row_id]; original_row=original[row_id]
        records=original_row['records']
        if seed is not None:
            other=torch.load(staging/'fields'/f'{row_id}-{seed}.pt',map_location='cpu',weights_only=True)['records']
            records=[(a if previous.source_for(*coordinate)=='A' else b | {'native_slot_seed':seed})
                     for a,b,coordinate in zip(records,other,COORDINATES,strict=True)]
        path=directory/item['path']
        save_tensor_file(path,original_row | dict(fingerprint=fingerprint,records=records))
        shards.append(item | {'sha256':file_hash(path)})
    atomic_json(directory/'index.json',dict(identity=identity,input_fingerprint=fingerprint,shards=shards,
                 fingerprint=digest(dict(input=fingerprint,shards=shards))))
    return TargetCache(directory,verify=True)


def parity(runtime, method, cache, request, entry):
    return shared_parity(runtime,method,cache,request)
