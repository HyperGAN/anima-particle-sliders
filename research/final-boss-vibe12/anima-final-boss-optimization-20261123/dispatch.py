"""Adopt the running first wave and dispatch same-card follow-up work."""

from collections import deque
from dataclasses import dataclass
import fcntl
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

import psutil

from optimization_worker import ROOT, read, write


CARDS = {
    0: ("soft-contexts-65-blend50", 4178600, 4178471, "soft-floor45"),
    1: ("soft-projection-65-blend50", 4178601, 4178472, "soft-floor80"),
}
MAX_ADMISSIONS = 3
RETRY_DELAY = 60
STATUS = ROOT / "campaign-status.json"
TERMINAL = ROOT / "campaign-terminal.json"


@dataclass
class Job:
    stage: str
    method: str
    gpu: int
    attempt: int
    process: subprocess.Popen | None = None
    guardian: psutil.Process | None = None
    worker_pid: int | None = None

    @property
    def stem(self):
        return ROOT / f"{self.stage}-{self.method}-a{self.attempt}"

    def fields(self):
        return {"stage": self.stage, "method": self.method, "gpu": self.gpu,
                "attempt": f"a{self.attempt}",
                "guard_pid": self.process.pid if self.process else (self.guardian.pid if self.guardian else None),
                "inherited_worker_pid": self.worker_pid}

    def receipt(self):
        path = self.stem.with_suffix(".terminal.json")
        if not path.is_file():
            return None
        receipt = read(path)
        seen_worker = receipt.get("pid")
        if receipt.get("gpu") != self.gpu or (self.worker_pid is not None and seen_worker != self.worker_pid):
            raise ValueError(f"Mismatched GPU or worker identity in {path}")
        return receipt

    def result(self):
        if self.process is not None:
            code = self.process.poll()
            if code is None:
                return None
        else:
            code = None
            if self.guardian is not None and live(self.guardian):
                return None
        receipt = self.receipt()
        if receipt is not None:
            return receipt
        # A failed flock occurs before the guard starts the child or opens its log.
        dispatch_log = ROOT / f"dispatch-{self.stem.name}.log"
        before_worker = self.process is not None and not self.stem.with_suffix(".log").exists()
        if before_worker and dispatch_log.is_file():
            log = dispatch_log.read_text()
            if "fcntl.flock(" in log and "BlockingIOError: [Errno 11]" in log:
                return {"status": "entry-refused", "gpu": self.gpu, "returncode": code,
                        "reason": "GPU lease unavailable", "worker_started": False}
        return {"status": "missing-terminal-receipt", "gpu": self.gpu, "returncode": code,
                "reason": "Guard exited before publishing resource cleanup"}


def live(process):
    try:
        return process.is_running() and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False


def matches(process, filename, arguments):
    command = process.cmdline()
    if command[-len(arguments):] != list(arguments) or len(command) < len(arguments) + 2:
        return False
    source = Path(command[-len(arguments) - 1])
    if not source.is_absolute():
        source = Path(process.cwd()) / source
    return source.resolve() == (ROOT / filename).resolve()


def adopt(gpu, method, worker_pid, guard_pid):
    job = Job("train", method, gpu, 1, worker_pid=worker_pid)
    receipt = job.receipt()
    try:
        guardian = psutil.Process(guard_pid)
        if live(guardian):
            if not matches(guardian, "optimization_guard.py", ("train", str(gpu), "a1", method)):
                raise RuntimeError(f"PID {guard_pid} does not own the declared GPU {gpu} guard")
            job.guardian = guardian
        worker = psutil.Process(worker_pid)
        if live(worker) and (job.guardian is None or worker.ppid() != guard_pid or
                             not matches(worker, "optimization_worker.py", ("train", "a1", method))):
            raise RuntimeError(f"PID {worker_pid} does not match its declared training guard")
        if job.guardian is not None:
            return job
    except psutil.NoSuchProcess:
        if job.guardian is not None:
            return job
    if receipt is None and job.receipt() is None:
        raise RuntimeError(f"No verified running guard or terminal receipt for {method}")
    return job


def launch(stage, method, gpu, attempt):
    job = Job(stage, method, gpu, attempt)
    if job.stem.with_suffix(".log").exists() or job.stem.with_suffix(".terminal.json").exists():
        raise RuntimeError(f"Attempt already exists: {job.stem}")
    with (ROOT / f"dispatch-{job.stem.name}.log").open("x") as output:
        job.process = subprocess.Popen([sys.executable, "-u", str(ROOT / "optimization_guard.py"),
                                        stage, str(gpu), f"a{attempt}", method],
                                       cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    print(f"{method} {stage} started on GPU {gpu} under guard {job.process.pid}", flush=True)
    return job


def classify(job, receipt):
    if receipt.get("status") == "entry-refused":
        return "admission-refused", True
    clean = receipt.get("remaining_pids") == [] and receipt.get("remaining_gpu_mib") == 0
    if clean and receipt.get("returncode") == 0 and receipt.get("reason") is None:
        return "complete", True
    if clean and job.stage == "train":
        milestones = sorted((ROOT / "cohorts" / job.method / "milestones").glob("RAW*.json"))
        if milestones and any(case["integrity"]["status"] == "fail" for case in read(milestones[-1])["cases"]):
            return "scientific-stop", True
    return "operational-stop", clean


def dispatch():
    active = {gpu: adopt(gpu, method, worker_pid, guard_pid)
              for gpu, (method, worker_pid, guard_pid, _) in CARDS.items()}
    queue = {gpu: deque([("train", followup, 1)]) for gpu, (_, _, _, followup) in CARDS.items()}
    cooling = {gpu: 0 for gpu in CARDS}
    finished, skipped = [], []
    previous = None

    def snapshot():
        return {"live": [job.fields() for job in active.values()],
                "waiting": [{"stage": stage, "method": method, "gpu": gpu, "attempt": f"a{number}"}
                            for gpu, jobs in queue.items() for stage, method, number in jobs],
                "terminal": finished.copy(), "skipped": skipped.copy()}

    def save_status():
        nonlocal previous
        current = snapshot()
        if current != previous:
            write(STATUS, current)
            previous = current

    save_status()
    while active or any(queue.values()):
        for gpu, job in list(active.items()):
            result = job.result()
            if result is None:
                continue
            del active[gpu]
            outcome, safe = classify(job, result)
            finished.append(job.fields() | {"outcome": outcome, "cleanup_verified": safe, "result": result})
            print(f"{job.method} {job.stage} on GPU {gpu}: {outcome}", flush=True)
            if outcome == "complete" and job.stage == "train":
                queue[gpu].appendleft(("tune", job.method, 1))
            elif outcome == "admission-refused" and job.attempt < MAX_ADMISSIONS:
                queue[gpu].appendleft((job.stage, job.method, job.attempt + 1))
                cooling[gpu] = monotonic() + RETRY_DELAY
            elif not safe:
                while queue[gpu]:
                    stage, method, _ = queue[gpu].popleft()
                    skipped.append({"stage": stage, "method": method, "gpu": gpu,
                                    "reason": "Preceding guard did not verify resource cleanup"})
        for gpu, jobs in queue.items():
            if gpu not in active and jobs and monotonic() >= cooling[gpu]:
                stage, method, attempt = jobs.popleft()
                active[gpu] = launch(stage, method, gpu, attempt)
        save_status()
        if active or any(queue.values()):
            sleep(5)

    tuned = {item["method"] for item in finished if item["stage"] == "tune" and item["outcome"] == "complete"}
    expected = {method for first, _, _, second in CARDS.values() for method in (first, second)}
    closed = snapshot() | {"outcome": "complete" if tuned == expected else "finished-with-stops"}
    write(TERMINAL, closed)
    return int(tuned != expected)


def main():
    with (ROOT / ".dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if TERMINAL.is_file():
            return int(read(TERMINAL)["outcome"] != "complete")
        if STATUS.exists():
            raise RuntimeError("An existing dispatcher receipt requires reconciliation before another dispatch")
        return dispatch()


if __name__ == "__main__":
    sys.exit(main())
