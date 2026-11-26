"""Run one worker under this campaign's GPU lease, entry floors and owned caps."""

import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from campaign import CODE, PARENT_STUDY, ROOT

GIB = 2**30
LOCKS = Path("/mnt/nvme/sliders/locks")


def nv(gpu, field):
    return subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-" + field, "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    ).splitlines()


def run(gpu, worker, arguments, product=None):
    parent = os.getpid()

    def death_guard():
        if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0 or os.getppid() != parent:
            os._exit(1)

    logs = ROOT / "runs"
    logs.mkdir(exist_ok=True)
    label = (arguments[0] if arguments else "all").replace(":", "-") + (f"+{len(arguments) - 1}" if len(arguments) > 1 else "")
    stem = logs / f"{Path(worker).stem}-{label}-{time.time_ns()}"
    # Training shares the equipped study's per-card leases. Inference-only calibration takes this
    # campaign's own lease and a higher host entry floor, so the study's companions keep room.
    training = Path(worker).name == "fit.py"
    # A second training of this campaign may share a card that has the room. It takes its own
    # lease and needs the first one's peak left free on entry, above the usual floor.
    second = training and os.environ.get("ANIMA_POST_TRAINING_SLOT") == "2"
    names = (
        ["post-training-second"]
        if second
        else ["adult-sliders-production", "duration-active", "post-training"]
        if training
        else ["post-training-companion"]
    )
    leases = [(LOCKS / f"gpu-{gpu}-anima-{name}.lock").open("a+") for name in names]
    try:
        for lease in leases:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"GPU {gpu} lease busy", flush=True)
        return 75
    free = int(nv(gpu, "gpu=memory.free")[0])
    host = psutil.virtual_memory().available
    if free < (21 if second else 13 if training else 11) * 1024 or host < (44 if training else 50) * GIB:
        print(f"Entry refused: GPU {free} MiB free, host {host / GIB:.1f} GiB", flush=True)
        return 75
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONPATH=str(product or ROOT / "product"),
        ANIMA_POST_TRAINING_ROOT=str(ROOT),
        ANIMA_RETEST_STUDY=str(ROOT),
        OMP_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONOPTIMIZE="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    with stem.with_suffix(".log").open("x", buffering=1) as log:
        child = subprocess.Popen(
            [str(PARENT_STUDY / "env/bin/python"), "-u", str(CODE / worker), *arguments],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=death_guard,
            pass_fds=tuple(lease.fileno() for lease in leases),
        )
        print(f"{worker} {arguments} on GPU {gpu}, PID {child.pid}, log {log.name}", flush=True)
        reason, peak, pids = None, 0, {child.pid}
        try:
            while child.poll() is None:
                try:
                    process = psutil.Process(child.pid)
                    family = [process, *process.children(recursive=True)]
                    pids.update(p.pid for p in family)
                    rss = sum(p.memory_info().rss for p in family)
                except psutil.NoSuchProcess:
                    continue
                rows = [line.split(",") for line in nv(gpu, "compute-apps=pid,used_gpu_memory")]
                used = sum(int(mib) for pid, mib in rows if int(pid) in pids)
                peak = max(peak, used)
                free = int(nv(gpu, "gpu=memory.free")[0])
                host = psutil.virtual_memory().available
                if free < 5 * 1024 or host < 36 * GIB:
                    reason = f"live floor crossed: GPU {free} MiB, host {host / GIB:.1f} GiB"
                elif used > 20 * 1024 or rss > 8 * GIB:
                    reason = f"owned cap crossed: GPU {used} MiB, RSS {rss / GIB:.1f} GiB"
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
            code = child.wait()
            terminal = dict(gpu=gpu, returncode=code, reason=reason, peak_gpu_mib=peak)
            stem.with_suffix(".terminal.json").write_text(json.dumps(terminal, indent=2) + "\n")
            print(terminal, flush=True)
    return (128 - code if code < 0 else code) or (1 if reason else 0)


if __name__ == "__main__":
    product = os.environ.get("ANIMA_POST_TRAINING_PRODUCT")
    sys.exit(run(int(sys.argv[1]), sys.argv[2], sys.argv[3:], product))
