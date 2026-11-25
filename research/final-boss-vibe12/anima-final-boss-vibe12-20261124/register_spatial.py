"""Pin each historical spatial recipe once its new scene inputs are ready."""
from hashlib import file_digest
from pathlib import Path
import json

from control import PREVIOUS, ROOT, SOURCE


RECIPES = {
    'soft-floor45': dict(kind='soft', floor=.45),
    'incumbent-refit': dict(kind='soft', floor=.65),
    'sparse-first': dict(kind='percentile', floor=.45, positions=list(range(3))),
    'sparse-full': dict(kind='percentile', floor=.45, positions=list(range(10))),
    'subject-apg': dict(kind='apg', floor=.45),
    'strong-boss-45-dose100': dict(kind='cue', floor=.45, mix=.85, positions=list(range(10)), prompt_keys=['explicit']),
    'strong-boss-45-mix35': dict(kind='cue', floor=.45, mix=.35, positions=list(range(10)), prompt_keys=['explicit']),
    'early-cues': dict(kind='cue', floor=.45, mix=.35, positions=list(range(3)), prompt_keys=['explicit']),
    'late-cues': dict(kind='cue', floor=.45, mix=.35, positions=list(range(7, 10)), prompt_keys=['explicit']),
    'two-cue-average': dict(kind='cue', floor=.45, mix=.35, positions=list(range(10)), prompt_keys=['explicit', 'alternate']),
    'in-scene-costume': dict(kind='cue', floor=.45, mix=.35, positions=list(range(10)), prompt_keys=['in_scene']),
    'matched-attire': dict(kind='attire', floor=.45),
    'soft45-background-orthogonal': dict(kind='exterior', floor=.45),
    'soft-contexts-45-blend50': dict(kind='paired', floor=.45),
    'soft-contexts-65-blend50': dict(kind='paired', floor=.65),
    'soft-projection-65-blend50': dict(kind='projection', floor=.65),
}


def pin(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        return dict(path=str(path), sha256=file_digest(stream, 'sha256').hexdigest())


def main():
    matte_index = ROOT/'mattes/index.json'
    mattes = json.loads(matte_index.read_text())
    common = [matte_index, *[Path(x['path']) for x in mattes['mattes'].values()],
              ROOT/'prepare_mattes.py', PREVIOUS/'optimization_targets.py',
              SOURCE/'release_tools/final_boss_targets.py', SOURCE/'release_tools/final_boss_apg.py']
    prompts = ROOT/'prompt-variants.json'
    for name, parameters in RECIPES.items():
        inputs = common.copy()
        extra = {}
        if parameters['kind'] not in ('soft', 'percentile', 'apg'):
            if not prompts.exists():
                continue
            inputs.append(prompts)
            extra['prompt_fixture'] = str(prompts)
        if parameters['kind']=='exterior':
            inputs.append(PREVIOUS/'background_orthogonal_teacher_worker.py')
        entry = dict(implementation=pin(ROOT/'adapters/spatial.py'), inputs=[pin(path) for path in inputs],
                     parameters=parameters) | extra
        path = ROOT/'ready'/f'{name}.json'
        content = json.dumps(entry, indent=2)+'\n'
        if path.exists():
            if path.read_text()!=content:
                raise ValueError(f'Published ready recipe changed: {name}')
        else:
            next_file = path.with_suffix('.new')
            next_file.write_text(content)
            next_file.replace(path)
            print(name, flush=True)


if __name__ == '__main__':
    main()
