"""Small offline CPU subprocess with resident-memory and host-availability guards."""
import json, os, pathlib, resource, signal, subprocess, sys, time
prefix = pathlib.Path(sys.argv[1]).resolve()
command = sys.argv[2:]
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
environment.pop('PYTHONPATH', None)
def available():
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
minimum = available()
if minimum < 16 * 1024**3:
    raise SystemExit('Not started: available host memory below 16 GiB')
peak = 0
reason = None
start = time.monotonic()
with prefix.with_suffix('.log').open('w') as output:
    process = subprocess.Popen(command, cwd='/tmp', env=environment, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    while process.poll() is None:
        minimum = min(minimum, available())
        pending, seen, rss = [process.pid], set(), 0
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                status = pathlib.Path(f'/proc/{pid}/status').read_text().splitlines()
                rss += next((int(line.split()[1]) * 1024 for line in status if line.startswith('VmRSS:')), 0)
                pending.extend(int(x) for x in pathlib.Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
            except (FileNotFoundError, ProcessLookupError):
                pass
        peak = max(peak, rss)
        if minimum < 16 * 1024**3:
            reason = 'host available memory below 16 GiB'
        elif peak > 2 * 1024**3:
            reason = 'subprocess tree RSS above 2 GiB'
        elif time.monotonic() - start > 180:
            reason = 'bounded 180 second timeout'
        if reason:
            os.killpg(process.pid, signal.SIGKILL)
            break
        time.sleep(.05)
    code = process.wait()
receipt = dict(command=command, cwd='/tmp', returncode=code, elapsed_seconds=round(time.monotonic() - start, 3), sampled_tree_peak_rss_bytes=peak, child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024, minimum_host_available_bytes=minimum, guard_stop=reason, environment={key: environment[key] for key in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
prefix.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
print(prefix.with_suffix('.log').read_text()[-16000:])
raise SystemExit(code)
