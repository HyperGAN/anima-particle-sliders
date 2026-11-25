"""Verify the finished leaderboard against every bound native RAW192 and teacher cache."""
from collections import Counter
from hashlib import file_digest
from pathlib import Path
import json
from time import sleep
from urllib.request import urlopen

from control import CACHE, MANIFEST, ROOT, STORAGE
from optimization_worker import read, write


COUNTS = {
    'shared-best-opening':336, 'shared-best-middle':336,
    'shared-best-neutral-path':120, 'shared-best-mixed-path':120,
    'shared-best-uniform-steps':144, 'shared-best-divergence':144,
}
URLs = ('http://127.0.0.1:8765/final-boss-vibe12-leaderboard.html',
        'http://127.0.0.1:8765/final-boss-vibe12-leaderboard-constant.html')
TERMINAL = {'complete','scientific-stop','blocked'}


def file_hash(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def verify_files(state, leaderboard):
    manifest=read(MANIFEST); stock=read(CACHE/'index.json')
    receipt=read(STORAGE/'targets/receipt.json')
    if leaderboard['teacher']!=receipt or file_hash(MANIFEST)!=receipt['manifest_file_sha256']:
        raise ValueError('Leaderboard is bound to a different teacher set')
    recorded = set(leaderboard['qualification']['fixed_strength'])
    if set(state['recipes']) != recorded or len(recorded) != 26:
        raise ValueError('The full leaderboard recipe inventory differs')
    results={}
    for method,status in state['recipes'].items():
        if status['state']=='scientific-stop':
            latest=sorted((ROOT/'cohorts'/method/'milestones').glob('RAW*.json'))[-1]
            screen=read(latest)
            step=screen['step']
            commits=STORAGE/method/'training/commits'
            entry=read(commits/f'{step:06d}.json')
            raw=entry['raw']
            if (not any(row['integrity']['status']=='fail' for row in screen['cases'])
                    or file_hash(raw['path'])!=raw['sha256']
                    or max(int(path.stem) for path in commits.glob('*.json'))!=step):
                raise ValueError(f'{method} failed to retain its exact stopped checkpoint')
            results[method]=dict(state='scientific-stop',updates=step,raw=raw,checkpoint_check=str(latest))
            continue
        if status['state']!='complete':
            results[method]=dict(state=status['state'], receipts=status['receipts'])
            continue
        directory=CACHE if method=='shared-best-stock' else STORAGE/'targets'/method
        cache=read(directory/'index.json'); identity=cache['identity']
        if (identity['manifest_sha256']!=manifest['sha256'] or identity['model']!=stock['identity']['model']
                or identity['variation']!=manifest['variation']
                or (cache['fingerprint']!=stock['fingerprint'] if method=='shared-best-stock' else identity['parent']!=stock['fingerprint'])
                or sum(s['count'] for s in cache['shards'])!=COUNTS.get(method,240)):
            raise ValueError(f'{method} did not use the registered shared teacher recipe')
        result=read(ROOT/'cohorts'/method/'result.json'); raw=result['raw']; raw_path=Path(raw['path'])
        sidecar=read(raw_path.with_suffix('.scoped.json'))
        report=read(ROOT/'tuning'/method/'boss-effect-tuning.json')
        if (result['step']!=192 or result['ema_decay'] is not None or result['cache']!=cache['fingerprint']
                or report['raw']!=raw or sidecar['completed_steps']!=192 or sidecar['selection']!='raw'
                or sidecar['ema_decay'] is not None or sidecar['options']['variation']!=manifest['variation']
                or raw_path.stat().st_size!=raw['bytes'] or file_hash(raw_path)!=raw['sha256']):
            raise ValueError(f'{method} does not have its completed RAW192 asset and matching scores')
        results[method]=dict(state='complete',raw=raw,cache=cache['fingerprint'],
                    qualifiers={condition:grades[method] for condition,grades in leaderboard['qualification'].items()})
    completed=sum(r['state']=='complete' for r in results.values())
    for url in URLs:
        with urlopen(url,timeout=10) as response:
            web=response.read().decode()
        if f'{completed} of 26 recipes with samples' not in web:
            raise ValueError('Hosted results have not caught up with completed training')
    return dict(teacher=receipt,counts=dict(Counter(r['state'] for r in results.values())),
                methods=results, urls=URLs, rankings=leaderboard['rankings'])


def main():
    state_path=ROOT/'campaign-state.json'
    while True:
        state=read(state_path)
        if not state['live'] and all(row['state'] in TERMINAL for row in state['recipes'].values()):
            break
        sleep(30)
    # Give the ongoing publisher and the one calibrated reread their normal publish window.
    sleep(60)
    leaderboard=read(STORAGE/'leaderboard.json')
    verdict=verify_files(state,leaderboard)
    write(STORAGE/'campaign-verified.json',verdict)
    print(json.dumps(verdict['counts']),flush=True)


if __name__=='__main__':
    main()
