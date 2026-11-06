"""Compile complete unsealed train/dev coverage without Torch/model imports."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen_studio.contracts import atomic_json
from lumen_studio.coverage import compile_coverage


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--dev-manifest", type=Path, required=True)
    ap.add_argument("--variation", default="candlelit")
    ap.add_argument("--mode", choices=("production", "pilot"), default="production")
    ap.add_argument("--pilot-row", action="append", default=None,
                    help="explicit row IDs from each split; pilot mode only")
    ap.add_argument("--pilot-train-seed", action="append", type=int, default=None)
    ap.add_argument("--pilot-dev-seed", action="append", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("fresh output required; existing source/cohort evidence is immutable")
    manifests = {s: json.loads(p.read_text()) for s, p in
                 (("train", a.train_manifest), ("dev", a.dev_manifest))}
    if a.pilot_row is not None:
        declared = {r["id"] for m in manifests.values() for r in m["rows"]}
        if not set(a.pilot_row) <= declared:
            ap.error("pilot contains undeclared row IDs")
    contracts = {}
    for split, manifest in manifests.items():
        rows = a.pilot_row
        if a.mode == "pilot" and rows is not None:
            declared = {r["id"] for r in manifest["rows"]}
            rows = [r for r in rows if r in declared]
        contracts[split] = compile_coverage(manifest, a.variation, mode=a.mode,
            pilot_rows=rows, pilot_seeds=getattr(a, "pilot_" + split + "_seed"))
    if not set(contracts["train"]["characters"]).isdisjoint(contracts["dev"]["characters"]):
        raise ValueError("train/development character overlap")
    atomic_json(a.out, dict(schema="anima-pg095-fullcoverage-preparation-plan-v1",
        status="METADATA_ONLY_NO_NATIVE_ADMISSION", mode=a.mode, contracts=contracts,
        production_rate_eligible=a.mode == "production",
        session_weight_EMA=None, materialized_EMA_shadows=False, selection="raw",
        fresh_cohort_required=True, old128_resume_forbidden=True,
        prospective_horizon=1600, prospective_effective_batch=8,
        hardware_admission_and_exact_source_review_required=True))
    print(json.dumps(dict(out=str(a.out), mode=a.mode,
        fields={s: c["field_count"] for s, c in contracts.items()},
        examples={s: c["example_count"] for s, c in contracts.items()},
        Torch_imported="torch" in sys.modules)))


if __name__ == "__main__":
    main()
