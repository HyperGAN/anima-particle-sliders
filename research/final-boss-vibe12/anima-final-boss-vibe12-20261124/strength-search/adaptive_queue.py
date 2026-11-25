"""Run the same adaptive alpha search as each RAW192 recipe becomes ready."""

import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
from time import sleep

import psutil

from run import GIB, available_cards

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
RESULTS = HERE / 'leaderboard-retune-visible'
PRIORITY = ('shared-best-opening', 'soft45-background-orthogonal', 'late-cues', 'shared-best-stock',
            'shared-best-seed-split', 'shared-best-divergence', 'early-cues')
LOCKS = Path('/mnt/nvme/sliders/locks')


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(data, indent=2) + '\n')
    pending.replace(path)


def previous_attempt(method):
    files = HERE.glob(f'retune-visible-{method}-a*.*')
    numbers = [int(match[1]) for path in files if (match := re.search(r'-a(\d+)\.(?:log|terminal\.json)$', path.name))]
    return max(numbers, default=0)


def available_lease(gpu):
    with (LOCKS / f'gpu-{gpu}-anima-vibe12-alpha-search.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def run():
    active, faults = {}, {}
    state_path = RESULTS / 'queue.json'
    while True:
        for gpu, (process, method, attempt) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            del active[gpu]
            report = HERE / f'retune-visible-{method}-a{attempt}.terminal.json'
            receipt = read(report) if report.is_file() else {}
            if returncode == 0 and (RESULTS / method / 'result.json').is_file():
                print(f'Completed adaptive alpha search for {method}', flush=True)
            elif (receipt.get('status') == 'entry-refused' or (
                    (receipt.get('reason') or '').startswith('live floor crossed')
                    and receipt.get('remaining_pids') == [] and not receipt.get('remaining_gpu_mib'))):
                print(f'{method} yielded capacity and will resume its saved measurements', flush=True)
            else:
                faults[method] = {'returncode': returncode, 'attempt': attempt, 'receipt': receipt}
                print(f'{method} blocked on attempt {attempt}', flush=True)

        training = read(CAMPAIGN / 'campaign-state.json')['recipes']
        methods = list(dict.fromkeys((*PRIORITY, *training)))
        completed = [method for method in methods if (RESULTS / method / 'result.json').is_file()]
        running = [method for _, method, _ in active.values()]
        eligible = [method for method in methods if training[method]['state'] == 'complete'
                    and method not in completed and method not in running and method not in faults]
        if eligible and psutil.virtual_memory().available >= (54 if active else 48) * GIB:
            for gpu in available_cards():
                if gpu in active or not available_lease(gpu):
                    continue
                method = eligible.pop(0)
                attempt = previous_attempt(method) + 1
                child = subprocess.Popen([sys.executable, '-u', str(HERE / 'guard.py'), str(gpu), f'a{attempt}',
                                          method, 'leaderboard-visible'], cwd=CAMPAIGN)
                active[gpu] = (child, method, attempt)
                print(f'Adaptive search for {method} on GPU {gpu}', flush=True)
                break
        save(state_path, {'trained': [method for method in methods if training[method]['state'] == 'complete'],
                          'complete': completed, 'active': {gpu: method for gpu, (_, method, _) in active.items()},
                          'faults': faults, 'training_state': {method: row['state'] for method, row in training.items()}})
        waiting_for_training = any(row['state'] in ('running', 'waiting') for row in training.values())
        if not (eligible or active or waiting_for_training):
            return bool(faults)
        sleep(15)


if __name__ == '__main__':
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / 'queue.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(run())
