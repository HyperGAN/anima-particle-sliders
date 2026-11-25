"""Keep both available cards working on the declared leaderboard cohort."""
import fcntl
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import sys
import time

import psutil

from control import CACHE, MANIFEST, PREVIOUS, ROOT, STORAGE
import optimization_worker as worker
worker.ROOT = ROOT
worker.STORAGE = STORAGE
from dispatch import classify, launch
from optimization_worker import pin, read, write

GIB = 2**30
SHARED = ('shared-best-stock', 'shared-best-middle', 'shared-best-opening', 'shared-best-neutral-path',
          'shared-best-mixed-path', 'shared-best-uniform-steps', 'shared-best-divergence')
RECOVERY_FILE = ROOT.parent / 'anima-final-boss-shared-seed-split-20261124/resume_supervisor.py'


def lane_is_free(gpu):
    names = ('vibe12-train', 'vibe12-metric-render', 'vibe12-native')
    for name in names:
        with (Path('/mnt/nvme/sliders/locks') / f'gpu-{gpu}-anima-{name}.lock').open('a+') as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
    return True


def prior_attempts(method, stage):
    pattern = re.compile(rf'dispatch-{stage}-{re.escape(method)}-a(\d+)\.log')
    attempts = [int(match[1]) for path in ROOT.glob(f'dispatch-{stage}-{method}-a*.log')
                if (match := pattern.fullmatch(path.name))]
    return max(attempts, default=0)


def check_stop(method, stage, outcome, result, recovery):
    if outcome == 'admission-refused':
        return True
    if outcome == 'scientific-stop':
        return False
    if not recovery.clean_stop(result):
        return False
    if stage == 'train':
        recovery.reconcile_pending(result, method, store=STORAGE)
    return True


def main():
    journal = ROOT / 'campaign-state.json'
    labels = read(ROOT / 'recipes.json')
    teacher_receipt = STORAGE / 'targets/receipt.json'
    protocol = ROOT / 'search-protocol.json'
    inputs = [MANIFEST, CACHE / 'index.json', teacher_receipt, protocol,
              *(ROOT / name for name in ('campaign.py', 'recipes.json', 'cohort.py', 'shared_data.py', 'recipe_loader.py',
                                         'optimization_guard.py', 'optimization_worker.py', 'control.py',
                                         'shared_capacity_guard.py')),
              *(PREVIOUS / name for name in ('optimization_guard.py', 'optimization_worker.py', 'tune.py', 'image-breakage-rule.json'))]
    provenance = [pin(path) for path in inputs]
    recovery_spec = spec_from_file_location('captured_vibe_resume', RECOVERY_FILE)
    recovery = module_from_spec(recovery_spec)
    recovery_spec.loader.exec_module(recovery)
    method_status = {method: dict(state='waiting', stage='train', receipts=[]) for method in labels}
    if journal.is_file():
        old = read(journal)
        if old['inputs'] != provenance or old['live']:
            raise RuntimeError('Reconcile the previous campaign and any live workers before restarting')
        method_status = old['recipes']
    active = {}
    last = None
    cooldowns = {}
    while True:
        for gpu, job in list(active.items()):
            result = job.result()
            if result is None:
                continue
            del active[gpu]
            outcome, clean = classify(job, result)
            datum = method_status[job.method]
            datum['receipts'].append(job.fields() | dict(outcome=outcome, cleanup_verified=clean, result=result))
            print(f'{job.method} {job.stage} {outcome}', flush=True)
            if outcome == 'complete':
                datum.update(state=('waiting' if job.stage == 'train' else 'complete'), stage='tune')
            elif outcome == 'scientific-stop':
                datum['state'] = 'scientific-stop'
            else:
                try:
                    again = clean and check_stop(job.method, job.stage, outcome, result, recovery)
                except RuntimeError as error:
                    datum['recovery_error'] = str(error)
                    again = False
                datum['state'] = 'waiting' if again else 'blocked'
                cooldowns[job.method] = time.monotonic() + 40
        runnable = [method for method, status in method_status.items() if status['state'] == 'waiting'
                    and (method in SHARED or (ROOT / 'ready' / f'{method}.json').is_file())
                    and time.monotonic() >= cooldowns.get(method, 0)]
        # Finish native comparison before starting a lower-priority recipe when a lane opens.
        runnable.sort(key=lambda method: (method_status[method]['stage'] != 'tune', labels_order[method]))
        for gpu in (0, 1):
            minimum = (55 if active else 50) * GIB
            if (not runnable or gpu in active or psutil.virtual_memory().available < minimum
                    or not lane_is_free(gpu) or recovery.gpu_free(gpu) < 15 * 1024):
                continue
            method = runnable.pop(0)
            datum = method_status[method]
            stage = datum['stage']
            attempt = prior_attempts(method, stage) + 1
            ready = ROOT / 'ready' / f'{method}.json'
            for item in provenance:
                if pin(Path(item['path'])) != item:
                    raise ValueError(f'Campaign source changed: {item["path"]}')
            folder = ROOT / 'waves' / f'{stage}-{method}-a{attempt}'
            folder.mkdir()
            write(folder / 'plan.json', dict(inputs=provenance, stage=stage, method=method, gpu=gpu, attempt=attempt,
                                             teacher=read(teacher_receipt), recipe=pin(ready) if ready.is_file() else None))
            active[gpu] = launch(stage, method, gpu, attempt)
            datum['state'] = 'running'
        current = dict(inputs=provenance, recipes=method_status, live={str(gpu): job.fields() for gpu, job in active.items()})
        json_repr = repr(current)
        if json_repr != last:
            write(journal, current)
            last = json_repr
        if all(status['state'] in ('complete', 'scientific-stop', 'blocked') for status in method_status.values()):
            return 0 if all(s['state'] != 'blocked' for s in method_status.values()) else 1
        time.sleep(5 if active else 15)


if __name__ == '__main__':
    labels_order = {method: i for i, method in enumerate(read(ROOT / 'recipes.json'))}
    with (ROOT / '.campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(main())
