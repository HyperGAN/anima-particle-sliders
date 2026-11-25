"""Calibrate separately from the training queue under bounded shared-card admission."""
from collections import deque
from hashlib import file_digest
import json
from pathlib import Path
import re
import subprocess
import sys
from time import sleep

import psutil

ROOT=Path(__file__).resolve().parent
GIB=2**30


def plan():
    config=json.loads((ROOT/'plan.json').read_text())
    for method, entry in config['methods'].items():
        with Path(entry['definition']).open('rb') as stream:
            if file_digest(stream,'sha256').hexdigest()!=entry['sha256']:
                raise ValueError(f'Calibrated definition changed for {method}')
    queue=deque()
    for method in config['methods']:
        previous=[int(match[1]) for path in ROOT.glob(f'tune-{method}-a*.terminal.json')
                  if (match:=re.fullmatch(rf'tune-{re.escape(method)}-a(\d+)\.terminal\.json',path.name))]
        if not (ROOT/f'{method}.alpha-tune/tuning.json').is_file():
            queue.append((method,max(previous,default=0)+1))
    return queue


def available_cards():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
    candidates=[tuple(map(int,row.split(','))) for row in lines]
    return [gpu for gpu,memory,utilization in sorted(candidates,key=lambda r:(r[2],-r[1])) if memory>=18*1024]


def main():
    queue=plan(); active={}; outcomes={}
    while queue or active:
        for gpu,(process,method,attempt) in list(active.items()):
            status=process.poll()
            if status is None:
                continue
            del active[gpu]
            path=ROOT/f'tune-{method}-a{attempt}.terminal.json'
            receipt=json.loads(path.read_text()) if path.exists() else {}
            complete=(ROOT/f'{method}.alpha-tune/tuning.json').is_file() and status==0
            if complete:
                print(f'{method} finished alpha search',flush=True)
            elif attempt < 4 and ((receipt.get('status')=='entry-refused') or
                  (receipt.get('reason','') or '').startswith('live floor crossed') and receipt.get('remaining_pids')==[] and not receipt.get('remaining_gpu_mib')):
                queue.appendleft((method,attempt+1))
                print(f'{method} yielded shared capacity at attempt {attempt}',flush=True)
                continue
            else:
                print(f'{method} exited {status}; inspect {path.name}',flush=True)
            outcomes[method]=dict(complete=complete,receipt=receipt)
            pending=ROOT/'status.pending';pending.write_text(json.dumps(outcomes,indent=2)+'\n');pending.replace(ROOT/'status.json')
        if queue and psutil.virtual_memory().available>=(54 if active else 48)*GIB:
            for gpu in available_cards():
                if gpu in active or not queue:
                    continue
                method,attempt=queue.popleft()
                process=subprocess.Popen([sys.executable,'-u',str(ROOT/'guard.py'),str(gpu),f'a{attempt}',method],cwd=ROOT)
                active[gpu]=(process,method,attempt)
                print(f'{method} alpha search on GPU {gpu}',flush=True)
                break
        if queue or active:
            sleep(8)
    return 0 if all(v['complete'] for v in outcomes.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
