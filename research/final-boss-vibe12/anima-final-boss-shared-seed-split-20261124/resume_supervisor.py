"""Exact-resume supervisor for our own shared-card training (never touches foreign jobs).

After a clean memory yield it waits for host headroom, reconciles the yield flag (only when the last own
training receipt is a clean yield with a contiguous journal), writes a fresh pinned resume wave and runs it.
It stops as soon as training completes (the wave then runs the shared-strength tune itself), or on anything
that is not a clean yield or an entry refusal.

  resume_supervisor.py --gpu 1 [--method shared-best-seed-split]
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parent
STORE = Path("/mnt/nvme/sliders/experiments") / ROOT.name
METHODS = ("shared-best-seed-split", "shared-best-stock")
FLAG = ROOT / "yield.flag"
GIB = 2**30
WAIT_HOST = 52.8 * GIB
WAIT_GPU_MIB = 13.5 * 1024
MAX_WAVES = 200


def gpu_free(gpu):
    return int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                                        "--format=csv,noheader,nounits"], text=True, timeout=10).split()[0])


def attempts(method, root=ROOT):
    found = [int(m.group(1)) for path in root.iterdir()
             if (m := re.fullmatch(rf"(?:dispatch-)?train-{re.escape(method)}-a(\d+)\.(?:log|terminal\.json)", path.name))]
    return max(found)


def last_step(method, store=STORE):
    commits = sorted((store / method / "training/commits").glob("*.json"))
    return int(commits[-1].stem)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_stop(receipt):
    reason = str(receipt.get("reason", ""))
    return ((receipt.get("returncode") == 75 and reason.startswith("yield floor")
             or receipt.get("returncode") == -15 and reason.startswith("live floor crossed"))
            and receipt.get("remaining_pids") == [] and receipt.get("remaining_gpu_mib") == 0)


def reconcile_pending(receipt, method, store=STORE):
    """Archive an in-flight update that our own guard interrupted, only when nothing of it was committed."""
    root = store / method / "training"
    pending = root / "pending-update.json"
    if not pending.exists():
        return None
    record = json.loads(pending.read_text())
    commits = sorted((root / "commits").glob("*.json"))
    steps = [int(path.stem) for path in commits]
    last = json.loads(commits[-1].read_text())
    base = Path(record["base"]["path"])
    run = root / f"run-{record['attempt']}"
    checks = {
        "own_worker_gone": receipt.get("remaining_pids") == [] and receipt.get("remaining_gpu_mib") == 0
                           and not psutil.pid_exists(receipt["pid"]),
        "journal_contiguous": steps == list(range(len(steps))),
        "pending_is_next_step": record["step"] == steps[-1] + 1,
        "base_is_last_commit": last["full"]["path"] == record["base"]["path"]
                               and last["full"]["sha256"] == record["base"]["sha256"],
        "base_bytes_intact": base.is_file() and sha(base) == record["base"]["sha256"],
        "no_artifact_of_pending_step": not (run / f"FULL{record['step']:06d}.pt").exists()
                                       and not (root / "commits" / f"{record['step']:06d}.json").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"In-flight update cannot be reconciled automatically: {checks}")
    folder = root / "reconciled"
    folder.mkdir(exist_ok=True)
    target = folder / f"pending-update-step{record['step']:03d}-{record['attempt']}.json"
    with target.open("x") as stream:
        json.dump({"pending": record, "interrupted_by": receipt, "checks": checks,
                   "resolution": "update never completed; no artifact; resume recomputes it from the last commit"},
                  stream, indent=2)
        stream.write("\n")
    pending.unlink()
    print(f"reconciled interrupted update {record['step']} -> {target}", flush=True)
    return target


def wait_headroom(gpu):
    good = 0
    while good < 3:
        ok = psutil.virtual_memory().available >= WAIT_HOST and gpu_free(gpu) >= WAIT_GPU_MIB
        good = good + 1 if ok else 0
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--method", choices=METHODS, default=METHODS[0])
    args = parser.parse_args()
    method = args.method
    for _ in range(MAX_WAVES):
        if (ROOT / "cohorts" / method / "result.json").exists():
            print("training already complete", flush=True)
            return 0
        last = attempts(method)
        receipt = json.loads((ROOT / f"train-{method}-a{last}.terminal.json").read_text())
        refused = receipt.get("status") == "entry-refused"
        if not (clean_stop(receipt) or refused):
            print(f"last own attempt a{last} is not an own memory-floor stop: {receipt}", flush=True)
            return 2
        if clean_stop(receipt):
            reconcile_pending(receipt, method)
        wait_headroom(args.gpu)
        if FLAG.exists():
            FLAG.unlink()
        step, attempt = last_step(method), last + 1
        wave = f"waves/resume-{method}-{step:03d}-a{attempt}"
        print(time.strftime("%H:%M:%SZ", time.gmtime()), f"resuming {method} from step {step} as a{attempt} on GPU {args.gpu}", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / "write_launch_plan.py"), "--gpu", str(args.gpu),
                        "--wave", wave, "--attempt", str(attempt), "--method", method], cwd=ROOT, check=True)
        with (ROOT / wave / "wave.out").open("x") as out:
            code = subprocess.run([sys.executable, "-u", str(ROOT / "control.py"), str(ROOT / wave / "launch-plan.json")],
                                  cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
        print(time.strftime("%H:%M:%SZ", time.gmtime()), f"wave {wave} exit {code}, step {last_step(method)}", flush=True)
        if code != 0:
            return code
    return 3


if __name__ == "__main__":
    sys.exit(main())
