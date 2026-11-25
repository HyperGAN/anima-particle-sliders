"""Run the constant-alpha retest to completion from the equipped Anima study directory."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import argparse, json, os, subprocess, sys, time

from locations import CODE, STUDY

settings = argparse.ArgumentParser(description=__doc__)
settings.add_argument(
    "--metric", choices=("distinct_boss_score",), default="distinct_boss_score"
)
settings.add_argument(
    "--study",
    type=Path,
    default=STUDY,
    help="equipped study containing product, native-prod and the environment",
)
args = settings.parse_args()
metric = args.metric
study = args.study.resolve()
root = study / "cohort-retest"
root.mkdir(parents=True, exist_ok=True)
panel = root / "panels.json"
source = (CODE / "panels.json").read_bytes()
if panel.exists() and panel.read_bytes() != source:
    raise ValueError("Retained panel differs from the registered cases and rubric")
if not panel.exists():
    panel.write_bytes(source)
if not __debug__:
    raise RuntimeError("The evidence validator requires Python assertions")
env = dict(
    os.environ,
    ANIMA_RETEST_STUDY=str(study),
    PYTHONPATH=str(study / "product"),
    PYTHONDONTWRITEBYTECODE="1",
    PYTHONOPTIMIZE="0",
)
python = str(study / "env/bin/python")


def execute(script, *arguments):
    subprocess.run(
        [python, str(CODE / script), *arguments], cwd=study, env=env, check=True
    )


execute("inventory.py")
execute("pack_source.py")
roster = json.loads((root / "inventory.json").read_text())
historical = json.loads(
    (study / "product/docs/final-boss-raw192/vibe12/standings.json").read_text()
)["methods"]
all_boss = {n for n, r in roster.items() if r["run"] not in ("anime", "appeal-v2")}
stage = "retest-board"


def shard(job):
    name, gpu = job
    if name == "neighbor-effect-board":
        key = ("anime", "appeal-v2")[gpu]
        expected = [root / "neighbor-effect-results" / key / "result.json"]
        arguments = [key]
    else:
        names = sorted(historical if name == stage else all_boss - set(historical))[
            gpu::2
        ]
        expected = [root / "methods" / n / metric / "result.json" for n in names]
        arguments = [str(gpu), "2", metric]
    attempts = 0
    while not all(p.exists() for p in expected):
        attempts += 1
        label = f"run-{metric}-{time.time_ns()}"
        guard = CODE / "gpu_guard.py"
        proc = subprocess.run(
            [sys.executable, str(guard), name, str(gpu), label, *arguments],
            cwd=study,
            env=env,
        )
        if proc.returncode and proc.returncode != 75:
            receipt = json.loads(
                (
                    study
                    / "native-prod"
                    / f"{name}-{arguments[0]}-{label}.terminal.json"
                ).read_text()
            )
            if not receipt.get("reason"):
                raise RuntimeError(
                    f"Worker error in {name} on GPU {gpu}, attempt {label}"
                )
        if attempts >= 60:
            raise RuntimeError(f"Repeated allocation refusal for {name} on GPU {gpu}")
        if proc.returncode:
            time.sleep(30)


with ThreadPoolExecutor(6) as pool:
    futures = [
        pool.submit(shard, (name, gpu))
        for name in (stage, "extra-board", "neighbor-effect-board")
        for gpu in (0, 1)
    ]
    while any(not f.done() for f in futures):
        execute("publish.py", metric)
        for f in futures:
            if f.done():
                f.result()
        pending = [f for f in futures if not f.done()]
        if pending:
            wait(pending, timeout=30, return_when=FIRST_COMPLETED)
    for f in futures:
        f.result()
execute("recover_ranking.py", metric)
execute("publish.py", metric)
# Reserve the completed board before opening the independent production scenes.
pin = root / "finalist-check/selection.json"
finished = False
if pin.is_file():
    names = [r["method"] for r in json.loads(pin.read_bytes())["entries"]]
    finished = bool(names) and all(
        (pin.parent / name / "result.json").is_file() for name in names
    )
if not finished:
    for attempt in range(60):
        tag = f"final-{time.time_ns()}"
        gpu = attempt % 2
        proc = subprocess.run(
            [
                sys.executable,
                str(CODE / "gpu_guard.py"),
                "finalist-check",
                str(gpu),
                tag,
                "leader",
            ],
            cwd=study,
            env=env,
        )
        if proc.returncode == 0:
            break
        report = json.loads(
            (
                study / "native-prod" / f"finalist-check-leader-{tag}.terminal.json"
            ).read_bytes()
        )
        if proc.returncode != 75 and not report.get("reason"):
            raise RuntimeError(f"Finalist confirmation failed in {tag}")
        time.sleep(30)
    else:
        raise RuntimeError("No capacity to confirm the reserved finalists")
execute("publish.py", metric)
execute("verify_export.py", "constant-retest", "--complete")
execute("snapshot.py")
execute("publish.py", metric)
execute("verify_export.py", "constant-retest", "--complete")
execute("markdown_samples.py")
