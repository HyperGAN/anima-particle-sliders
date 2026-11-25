"""Apply the historical spatial teacher recipes at the twelve frozen native states."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch
from lumen_studio.cache import TargetCache, save_tensor_file
from lumen_studio.contracts import atomic_json, digest, file_hash

from control import CACHE, MANIFEST, ROOT, SOURCE, STORAGE
from optimization_worker import guarded, pin, read
from optimization_targets import apply_target

# Load the pinned target math without letting this checkout select the trainer.
sys.path.insert(0, str(SOURCE))
try:
    from release_tools.final_boss_targets import edited_target, matched_attire_target, percentile_delta, subject_matte
    from release_tools.final_boss_apg import project_subject_target
finally:
    sys.path.pop(0)


def context(runtime, row, parent, entry):
    mask = read(ROOT / 'mattes/index.json')['mattes'][row['id']]
    matte = subject_matte(mask['path'], parent['latent'].shape[-2:], runtime.device)
    parameters = entry['parameters']
    cues = read(Path(entry['prompt_fixture']))['rows'][row['id']] if 'prompt_fixture' in entry else {}
    return dict(matte=matte, weight=parameters['floor'] + (1 - parameters['floor']) * matte,
                prompts=cues, neutral=runtime.encode(row['neutral']), positive=runtime.encode(row['positive']))


def transformed(runtime, latent, step, neutral, positive, context, params, position, prediction):
    kind = params['kind']
    weight = context['weight']
    prompts = context['prompts']
    def predict(key):
        return prediction(prompts[key])
    if kind == 'soft':
        return apply_target(dict(weight=weight, blend=0.), neutral, positive, None)[0]
    if kind == 'percentile':
        return neutral + percentile_delta(neutral, positive, weight, active=position in params['positions'])
    if kind == 'apg':
        baseline = edited_target(neutral, positive, weight, 0.)
        sigma = float(runtime.scheduler(10).sigmas[position])
        return project_subject_target(latent, sigma, neutral, positive, baseline, context['matte'])
    if kind == 'paired':
        directions = [positive-neutral, predict('relight_positive')-predict('relight_neutral'),
                      predict('resite_positive')-predict('resite_neutral')]
        return apply_target(dict(weight=weight, blend=.5), neutral, positive, neutral + torch.stack(directions).mean(dim=0))[0]
    if kind == 'exterior':
        from background_orthogonal_teacher_worker import check_field, project_exterior
        baseline = apply_target(dict(weight=weight, blend=0.), neutral, positive, None)[0]
        result, diagnostics = project_exterior(neutral, baseline, [predict('projection_light'), predict('projection_site')],
                                               context['matte'])
        check_field(diagnostics)
        return result
    if kind == 'projection':
        first = (predict('projection_light') - neutral).flatten()
        second = (predict('projection_site') - neutral).flatten()
        first = first / first.norm().clamp_min(1e-5)
        second = second - torch.dot(second, first) * first
        second = second / second.norm().clamp_min(1e-5)
        raw = positive - neutral
        flat = raw.flatten()
        projected = raw - .7 * (torch.dot(flat, first) * first + torch.dot(flat, second) * second).reshape_as(raw)
        donor = neutral + projected * (raw.norm() / projected.norm().clamp_min(1e-10))
        return apply_target(dict(weight=weight, blend=.5), neutral, positive, donor)[0]
    if kind == 'attire':
        return matched_attire_target(neutral, positive, predict('explicit'), predict('ordinary'), weight, .35)
    if kind == 'cue':
        active = position in params['positions']
        cues = [predict(key) for key in params['prompt_keys']] if active else []
        return edited_target(neutral, positive, weight, params['mix'], *cues)
    raise ValueError(f'Unknown field recipe {kind}')


class Auxiliary:
    """Reuse a whole prompt's raw velocity stack across recipes at the same frozen states."""

    def __init__(self, runtime, source, row, parent, *, fresh=False):
        self.runtime = runtime
        self.row = row
        self.parent = parent
        self.source = source
        self.fresh = fresh
        self.items = {}
        self.embeddings = {}

    def predict(self, prompt, position):
        if self.fresh:
            if prompt not in self.embeddings:
                self.embeddings[prompt] = self.runtime.encode(prompt)
            record = self.parent['records'][position]
            return self.runtime.predict(record['latent'].to(self.runtime.device), record['timestep'], self.embeddings[prompt])
        if prompt not in self.items:
            identity = dict(parent=self.source.index['fingerprint'], row=self.row['id'], prompt=prompt,
                            model=self.runtime.identity)
            key = digest(identity)
            directory = STORAGE / 'targets' / 'auxiliary'
            directory.mkdir(exist_ok=True)
            path = directory / f'{key}.pt'
            if not path.exists():
                embedding = self.runtime.encode(prompt)
                fields = []
                for record in self.parent['records']:
                    guarded()
                    fields.append(self.runtime.predict(record['latent'].to(self.runtime.device), record['timestep'], embedding).cpu())
                # Each process stages its own file; a concurrent equivalent writer may publish the same complete tensor stack.
                with TemporaryDirectory(dir=directory, prefix='.field-') as work:
                    staged = Path(work) / path.name
                    save_tensor_file(staged, dict(identity=identity, fields=fields))
                    staged.replace(path)
            data = torch.load(path, map_location='cpu', weights_only=True)
            if data['identity'] != identity or len(data['fields']) != len(self.parent['records']):
                raise ValueError('Auxiliary native velocity changed')
            self.items[prompt] = data['fields']
        return self.items[prompt][position].to(self.runtime.device)


def validated_parent(source, item):
    parent = torch.load(source.directory / item['path'], map_location='cpu', weights_only=True)
    positions = [(record['trajectory'], record['position']) for record in parent['records']]
    if (parent['fingerprint'] != source.index['input_fingerprint'] or parent['row']['id'] != item['row']
            or parent['seed'] != item['seed'] or positions != [(t, p) for t in ('neutral','positive') for p in range(10)]):
        raise ValueError('Transformed teacher must retain its actual frozen path')
    return parent


@torch.no_grad()
def prepare_cache(runtime, method, entry):
    source = TargetCache(CACHE, verify=True)
    if source.index['identity']['model'] != runtime.identity or len(source) != 240:
        raise ValueError('Missing the single native twelve-scene source cache')
    directory = STORAGE / 'targets' / method
    identity = source.index['identity'] | dict(parent=source.index['fingerprint'], layout=f'frozen-vibe12/{method}/v1',
        recipe=pin(ROOT / 'ready' / f'{method}.json'))
    fingerprint = digest(identity)
    if (directory / 'index.json').is_file():
        result = TargetCache(directory, verify=True)
        if result.index['identity'] != identity or len(result) != 240:
            raise ValueError('The transformed teacher cache no longer matches this recipe')
        return result
    directory.mkdir(exist_ok=True)
    shards = []
    with runtime.mixer.scales({}):
        for item in source.index['shards']:
            parent = validated_parent(source, item)
            path = directory / item['path']
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if saved['fingerprint'] != fingerprint or saved['row'] != parent['row'] or len(saved['records']) != 20:
                    raise ValueError('Partial derived fields are bound to a different recipe')
            else:
                config = context(runtime, parent['row'], parent['records'][0], entry)
                predictor = Auxiliary(runtime, source, parent['row'], parent)
                fields = []
                for i, record in enumerate(parent['records']):
                    guarded()
                    neutral, positive, latent = (record[key].to(runtime.device) for key in ('neutral','positive','latent'))
                    target = transformed(runtime, latent, record['timestep'], neutral, positive, config,
                                         entry['parameters'], record['position'], lambda prompt: predictor.predict(prompt, i))
                    norm_ratio = (target-neutral).norm() / (positive-neutral).norm()
                    if not torch.isfinite(target).all() or abs(norm_ratio.item()-1.) > 2e-5:
                        raise ValueError('Derived target changed the intended per-field edit norm')
                    fields.append(record | {'positive': target.cpu()})
                save_tensor_file(path, parent | dict(records=fields, fingerprint=fingerprint))
            shards.append(item | {'sha256': file_hash(path)})
    index = dict(identity=identity, input_fingerprint=fingerprint, shards=shards,
                 fingerprint=digest(dict(input=fingerprint, shards=shards)))
    atomic_json(directory / 'index.json', index)
    return TargetCache(directory, verify=True)


@torch.no_grad()
def parity(runtime, method, cache, request, entry):
    source = TargetCache(CACHE, verify=True)
    if request != dict(model_identity=runtime.identity, cache_fingerprint=cache.index['fingerprint'],
                       response_shape=list(cache[0]['positive'].shape)):
        return False
    checked = []
    with runtime.mixer.scales({}):
        for j, item in enumerate(source.index['shards']):
            guarded()
            parent = validated_parent(source, item)
            i = (j * 7 + 1) % 20
            record = parent['records'][i]
            stored = cache[j * 20 + i]
            config = context(runtime, parent['row'], record, entry)
            predictor = Auxiliary(runtime, source, parent['row'], parent, fresh=True)
            step, latent = record['timestep'], record['latent'].to(runtime.device)
            native_neutral, native_positive = (runtime.predict(latent, step, config[key]) for key in ('neutral', 'positive'))
            target = transformed(runtime, latent, step, native_neutral, native_positive, config,
                                 entry['parameters'], record['position'], lambda prompt: predictor.predict(prompt, i))
            checked.append(dict(row=item['row'], position=record['position'], trajectory=record['trajectory'],
                embedding=torch.equal(config['neutral'].cpu(), stored['embedding']),
                native_pair=torch.equal(native_neutral.cpu(), record['neutral']) and torch.equal(native_positive.cpu(), record['positive']),
                transformed=torch.equal(stored['positive'], target.cpu()),
                path=torch.equal(stored['latent'], record['latent']) and torch.equal(stored['neutral'], record['neutral'])))
    passed = all(all(row[key] for key in ('embedding','native_pair','transformed','path')) for row in checked)
    atomic_json(ROOT / 'cohorts' / method / 'native-parity.json', dict(passed=passed, request=request, fields=checked))
    return passed
