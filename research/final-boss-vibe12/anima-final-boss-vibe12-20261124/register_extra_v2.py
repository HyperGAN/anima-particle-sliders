"""Pin the seed and costume arms and register the rollout when its new donor closes."""
import argparse
import json
from pathlib import Path
from time import sleep

from control import ROOT, SOURCE, STORAGE
from register_spatial import pin


def register(method, adapter, inputs, **settings):
    path=ROOT/'ready'/f'{method}.json'
    entry=dict(implementation=pin(ROOT/'adapters'/adapter),inputs=[pin(item) for item in inputs]) | settings
    exact=json.dumps(entry,indent=2)+'\n'
    if path.exists():
        if path.read_text()!=exact:
            raise ValueError(f'Published recipe changed: {method}')
        return
    next_file=path.with_suffix('.new')
    next_file.write_text(exact)
    next_file.replace(path)
    print(f'Ready {method}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    older=ROOT.parent/'anima-final-boss-shared-seed-split-20261124'
    previous=ROOT.parent/'anima-final-boss-optimization-20261123'
    seed_sources=[older/'seed_split.py',older/'qualify.py',SOURCE/'lumen_studio/vibe_observer.py',
                  SOURCE/'lumen_studio/vibe_teacher_selection.py',SOURCE/'lumen_studio/vibe_teacher_checks.py']
    register('shared-best-seed-split','seed_cross_stock.py',seed_sources,minimum_scenes=8,seed_offsets=[1,2,3])
    grounded=ROOT.parent/'anima-vibe-teacher-targets-20261124/lumen_studio/vibe_target_search.py'
    register('costume-teachers','costume_stock.py',[ROOT/'prompt-variants.json',ROOT/'adapters/seed_cross_stock.py',
              ROOT/'ready/soft-floor45.json',grounded,*seed_sources],target_source=str(grounded),costume_scenes=8)
    result=ROOT/'cohorts/soft-floor45/result.json'
    while not result.is_file():
        if not args.watch:
            return
        sleep(25)
    source=json.loads(result.read_text())
    if source['step']!=192 or source['ema_decay'] is not None:
        raise ValueError('Adapter rollout needs a fresh qualified RAW192 donor')
    raw=Path(source['raw']['path'])
    register('soft45-adapter-rollout','rollout_stock.py',[result,raw,raw.with_suffix('.scoped.json'),
              STORAGE/'targets/soft-floor45/index.json',ROOT/'mattes/index.json',ROOT/'ready/soft-floor45.json',
              ROOT/'adapters/spatial_stock.py',previous/'adapter_rollout_teacher_worker.py'],donor=str(result))


if __name__=='__main__':
    main()
