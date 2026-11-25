"""Bounded GPU ownership for the isolated Anima experiment."""

import ctypes
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from optimization_targets import plan
from optimization_worker import ROOT, PRODUCT, write as save

PYTHON = "/mnt/nvme/sliders/experiments/anima-pg-default-empf-20261116/env-stock/bin/python"
GIB = 2**30


def nv(gpu, field):
    return subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-" + field, "--format=csv,noheader,nounits"],
        text=True, timeout=10,
    ).splitlines()


def available(gpu):
    return int(nv(gpu, "gpu=memory.free")[0]), psutil.virtual_memory().available


def owned_memory(gpu, pids):
    rows = [line.split(",") for line in nv(gpu, "compute-apps=pid,used_gpu_memory")]
    return sum(int(mib) for pid, mib in rows if int(pid) in pids)


def run(stage, gpu, attempt, arguments=()):
    parent = os.getpid()

    def death_guard():
        if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0 or os.getppid() != parent:
            os._exit(1)

    stem = ROOT / f"{stage}-{arguments[0]}-{attempt}"
    lock_root = Path("/mnt/nvme/sliders/locks")
    locks = [lock_root / f"gpu-{gpu}-anima-{name}.lock" for name in
             ("vibe12-train", "vibe12-metric-render", "vibe12-native")]
    from contextlib import ExitStack
    with ExitStack() as stack:
        leases = [stack.enter_context(path.open("a+")) for path in locks]
        for lease in leases:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entry_gpu, entry_host = available(gpu)
        entry_min = 13 * 1024 if stage == "train" else 11 * 1024
        if entry_gpu < entry_min or entry_host < 47 * GIB:
            save(stem.with_suffix(".terminal.json"), {
                "status": "entry-refused", "gpu": gpu, "gpu_free_mib": entry_gpu,
                "host_available": entry_host, "gpu_required_mib": entry_min})
            return 75
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   PYTHONPATH=f"{PRODUCT / 'product'}:{PRODUCT / 'runtime-overlay'}",
                   PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   ANIMA_WORKER_GUARD=str(parent))
        log = stack.enter_context(stem.with_suffix(".log").open("x", buffering=1))
        teacher = plan(arguments[0]).get("teacher") if stage in ("train", "prepare") else None
        worker = ROOT / {
            "explicit-boss-cues-v1": "strong_teacher_worker.py",
            "explicit-boss-cues-low-floor-v1": "lower_background_teacher_worker.py",
            "explicit-boss-cues-low-mix-v1": "low_mix_teacher_worker.py",
            "background-orthogonal-v1": "background_orthogonal_teacher_worker.py",
            "adapter-rollout-v1": "adapter_rollout_teacher_worker.py",
        }.get(teacher, "optimization_worker.py")
        child = subprocess.Popen([PYTHON, "-u", str(worker), stage, attempt, *arguments],
                                 env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, preexec_fn=death_guard,
                                 pass_fds=tuple(x.fileno() for x in leases))
        started = time.monotonic()
        peak_gpu = peak_rss = 0
        reason = None
        pids = {child.pid}
        print(f"{stage} attempt {attempt}, GPU {gpu}, owned PID {child.pid}", flush=True)
        try:
            while child.poll() is None:
                try:
                    process = psutil.Process(child.pid)
                    family = [process, *process.children(recursive=True)]
                    pids.update(p.pid for p in family)
                    rss = sum(p.memory_info().rss for p in family)
                except psutil.NoSuchProcess:
                    continue
                gpu_used = owned_memory(gpu, pids)
                free_gpu, free_host = available(gpu)
                peak_gpu, peak_rss = max(peak_gpu, gpu_used), max(peak_rss, rss)
                if free_gpu < 5 * 1024 or free_host < 47 * GIB:
                    reason = f"live floor crossed: GPU {free_gpu} MiB, host {free_host / GIB:.2f} GiB"
                elif gpu_used > 20 * 1024 or rss > 24 * GIB:
                    reason = f"owned cap crossed: GPU {gpu_used} MiB, RSS {rss / GIB:.2f} GiB"
                if reason:
                    break
                time.sleep(2)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            result = child.wait()
            gpu_remaining = owned_memory(gpu, pids)
            remaining = [pid for pid in pids if psutil.pid_exists(pid)]
            terminal = dict(gpu=gpu, pid=child.pid, returncode=result, reason=reason,
                            elapsed_s=time.monotonic() - started, peak_gpu_mib=peak_gpu,
                            peak_rss_bytes=peak_rss, remaining_pids=remaining,
                            remaining_gpu_mib=gpu_remaining)
            save(stem.with_suffix(".terminal.json"), terminal)
            print(terminal, flush=True)
    return (128 - result if result < 0 else result) or (1 if reason or remaining or gpu_remaining else 0)


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]))
