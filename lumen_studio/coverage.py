"""Pure metadata coverage contracts for fresh production PG0.9.5 cohorts.

The historical two-shard pilot remains reproducible. It cannot satisfy this
production contract or publish a production learning-rate recommendation.
"""
from collections import Counter
import math
from pathlib import PurePosixPath
import re

from .contracts import digest
from .dataset import validate_manifest

PATHS = ("neutral", "positive")
POSITIONS = tuple(range(10))
_NAME = re.compile(r"[A-Za-z0-9_-]+\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def compile_coverage(manifest, variation, *, mode="production", pilot_rows=None,
                     pilot_seeds=None):
    """Derive every shard/field from a validated complete declared manifest.

    Pilot restrictions are explicit selections, never a first-N implicit cut.
    A pilot remains ineligible even if its selection happens to cover all rows.
    """
    validate_manifest(manifest)
    _require(manifest["split"] in ("train", "dev"), "only unsealed train/dev coverage")
    _require(mode in ("production", "pilot"), "unknown coverage mode")
    declared = [r for r in manifest["rows"] if r["variation"] == variation]
    _require(declared, "variation has no declared rows")
    for row in declared:
        _require(_NAME.fullmatch(row["id"]), "row ID must be a safe filename component")
        _require(row["seeds"] and all(type(s) is int for s in row["seeds"]),
                 "declared seeds must be exact integers")
        _require(len(set(row["seeds"])) == len(row["seeds"]), "duplicate declared seed")
    ids = [r["id"] for r in declared]
    _require(len(set(ids)) == len(ids), "duplicate declared row ID")
    if mode == "production":
        _require(pilot_rows is None and pilot_seeds is None,
                 "production rejects pilot row/seed restrictions")
        selected = declared
        seeds = {r["id"]: list(r["seeds"]) for r in selected}
    else:
        _require(isinstance(pilot_rows, list) and pilot_rows
                 and len(set(pilot_rows)) == len(pilot_rows),
                 "pilot requires explicit unique row IDs")
        _require(set(pilot_rows) <= set(ids), "pilot contains undeclared rows")
        _require(isinstance(pilot_seeds, list) and pilot_seeds
                 and all(type(s) is int for s in pilot_seeds)
                 and len(set(pilot_seeds)) == len(pilot_seeds),
                 "pilot requires explicit unique integer seeds")
        selected = [r for r in declared if r["id"] in pilot_rows]
        _require(all(set(pilot_seeds) <= set(r["seeds"]) for r in selected),
                 "pilot contains undeclared seeds")
        seeds = {r["id"]: [s for s in r["seeds"] if s in pilot_seeds] for r in selected}
    shards = [dict(path=f"{r['id']}-{s}.pt", row=r["id"], seed=s, count=20)
              for r in selected for s in seeds[r["id"]]]
    fields = [[s["row"], s["seed"], path, position]
              for s in shards for path in PATHS for position in POSITIONS]
    result = dict(schema="anima-pg095-declared-coverage-v1", mode=mode,
        production_rate_eligible=(mode == "production"), split=manifest["split"],
        variation=variation, manifest_sha256=manifest["sha256"],
        row_ids=[r["id"] for r in selected],
        row_sha256={r["id"]: digest(r) for r in selected}, seeds_by_row=seeds,
        characters=sorted({r["character"] for r in selected}),
        definitions=sorted({r["definition"] for r in selected}),
        bare_rows=[r["id"] for r in selected if r.get("bare")],
        shards=shards, field_keys=fields, field_count=len(fields),
        example_count=len(shards) * 2, positions=list(POSITIONS), paths=list(PATHS),
        example_key="row_id,seed,trajectory; character is a separate stratum",
        scope="metadata coverage only; no tensor/native qualification inferred")
    result["sha256"] = digest(result)
    return result


def validate_contract(contract):
    _require(contract.get("schema") == "anima-pg095-declared-coverage-v1",
             "unknown coverage schema")
    _require(digest({k: v for k, v in contract.items() if k != "sha256"})
             == contract.get("sha256"), "coverage checksum differs")
    _require(contract.get("mode") in ("production", "pilot")
             and contract.get("production_rate_eligible") is (contract["mode"] == "production"),
             "coverage mode/eligibility differs")


def validate_index(index, contract):
    """Exact complete shard membership; payload authentication is separate."""
    validate_contract(contract)
    identity = index["identity"]
    for key in ("split", "variation", "manifest_sha256"):
        _require(identity.get(key) == contract[key], "index coverage differs: " + key)
    _require(identity.get("coverage_sha256") == contract["sha256"],
             "index omits exact coverage contract")
    actual = index.get("shards", [])
    for shard in actual:
        path = PurePosixPath(shard["path"])
        _require(len(path.parts) == 1 and path.name not in (".", ".."),
                 "unsafe shard path")
        _require(isinstance(shard.get("sha256"), str) and _SHA.fullmatch(shard["sha256"]),
                 "missing actual shard file checksum")
        _require(type(shard.get("seed")) is int and type(shard.get("count")) is int,
                 "shard seed/count require exact integers")
    keys = ("path", "row", "seed", "count")
    _require([{k: s[k] for k in keys} for s in actual] == contract["shards"],
             "exact declared shard order/membership/count required")
    _require(index.get("input_fingerprint") == digest(identity), "input fingerprint differs")
    _require(index.get("fingerprint") == digest(dict(input=digest(identity), shards=actual)),
             "complete index fingerprint differs")
    return dict(shards=len(actual), fields=sum(s["count"] for s in actual),
                coverage_sha256=contract["sha256"], payloads_verified=False)


def field_metadata(row, seed, trajectory, position, *, index):
    """Distinct prompt pairs must never collapse into a character-only example."""
    _require(type(seed) is int and trajectory in PATHS and type(position) is int
             and position in POSITIONS and type(index) is int, "invalid native field coordinate")
    return dict(index=index, row_id=row["id"], subject=row["character"], seed=seed,
                trajectory=trajectory, position=position,
                example=[row["id"], str(seed), trajectory])


def native_field_sse(prediction, neutral, positive, metadata, torch):
    """The frozen experiment's operand-first FP64 native response arithmetic."""
    prediction, neutral, positive = [x.detach().cpu().double().flatten()
                                     for x in (prediction, neutral, positive)]
    _require(prediction.shape == neutral.shape == positive.shape, "field shapes differ")
    _require(all(torch.isfinite(x).all() for x in (prediction, neutral, positive)),
             "nonfinite native field; no student-based exclusion")
    target = positive - neutral
    reverse = neutral - target
    return dict(metadata, numel=target.numel(),
        error_sse=float((prediction - positive).square().sum()),
        teacher_sse=float(target.square().sum()),
        oracle_SSE=dict(OFF=float((neutral - positive).square().sum()),
                        teacher=float((positive - positive).square().sum()),
                        reverse=float((reverse - positive).square().sum())))


def validate_field_membership(fields, contract):
    validate_contract(contract)
    _require(len({f["index"] for f in fields}) == len(fields), "duplicate field index")
    actual = []
    for f in fields:
        _require(type(f.get("index")) is int and type(f.get("seed")) is int
                 and type(f.get("position")) is int and f.get("trajectory") in PATHS,
                 "field coordinates require exact integer types and declared paths")
        _require(f["example"] == [f["row_id"], str(f["seed"]), f["trajectory"]],
                 "example must identify row, not only character")
        actual.append((f["row_id"], f["seed"], f["trajectory"], f["position"]))
    _require(Counter(actual) == Counter(tuple(x) for x in contract["field_keys"]),
             "exact declared row/seed/path/all-position membership required")
    return dict(fields=len(fields), examples=len({tuple(f["example"]) for f in fields}),
                coverage_sha256=contract["sha256"])


def require_production_rate_coverage(manifests, contracts, indexes, fields):
    """Necessary coverage gate; training/source/RAW/final gates remain separate."""
    _require(set(manifests) == set(contracts) == set(indexes) == set(fields) == {"train", "dev"},
             "both complete production train/dev splits required")
    _require(set(contracts["train"]["characters"]).isdisjoint(contracts["dev"]["characters"]),
             "training and development subjects overlap")
    for split in ("train", "dev"):
        c = contracts[split]
        validate_contract(c)
        _require(c["mode"] == "production" and c["production_rate_eligible"] is True,
                 "bounded pilot cannot publish a production rate")
        _require(c == compile_coverage(manifests[split], c["variation"]),
                 "production contract differs from complete declared manifest")
        validate_index(indexes[split], c)
        validate_field_membership(fields[split], c)
    return dict(status="PASS_FULL_DECLARED_TRAIN_DEV_COVERAGE",
                scope="complete authenticated metadata membership, not native parity or quality",
                train_coverage_sha256=contracts["train"]["sha256"],
                dev_coverage_sha256=contracts["dev"]["sha256"],
                train_cache_fingerprint=indexes["train"]["fingerprint"],
                dev_cache_fingerprint=indexes["dev"]["fingerprint"],
                actual_native_or_rate_qualification=False)


def _teacher_fields(fields, contract):
    validate_field_membership(fields, contract)
    result = []
    for f in fields:
        _require(type(f.get("teacher_sse")) in (int, float)
                 and math.isfinite(f["teacher_sse"]) and f["teacher_sse"] >= 0,
                 "invalid teacher SSE")
        result.append({k: f[k] for k in ("index", "row_id", "seed", "trajectory",
                                        "position", "example", "teacher_sse")})
    return result


def freeze_teacher_membership(fields, contract):
    """Teacher-only exact-zero membership, produced before any student exists."""
    teachers = _teacher_fields(fields, contract)
    groups = {}
    for f in teachers:
        groups.setdefault(tuple(f["example"]), []).append(f["teacher_sse"])
    valid, excluded = [], []
    for example, values in sorted(groups.items()):
        energy = math.fsum(values)
        row = dict(example=list(example), teacher_energy=energy, fields=len(values))
        (valid if energy > 0 else excluded).append(row)
    _require(valid, "all declared examples have exact-zero teacher energy")
    return dict(schema="anima-fullcoverage-frozen-teacher-membership-v1",
                coverage_sha256=contract["sha256"], teacher_fields_sha256=digest(teachers),
                valid=valid, excluded=excluded, zero_threshold=0.,
                student_based_exclusion_forbidden=True)


def pool_primary(fields, contract, *, teacher_membership):
    """The existing operand-FP64 field SSEs, pooled by distinct prompt example.

    Scalar inputs must come from authenticated saved fields/actual native source.
    This function changes grouping authority, not tensor arithmetic or the metric.
    """
    validate_field_membership(fields, contract)
    _require(teacher_membership == freeze_teacher_membership(fields, contract),
             "actual teachers/membership differ from pristine pre-student freeze")
    grouped = {}
    for f in fields:
        _require(all(type(f.get(k)) in (int, float) and math.isfinite(f[k]) and f[k] >= 0
                     for k in ("error_sse", "teacher_sse")), "invalid field SSE")
        grouped.setdefault(tuple(f["example"]), []).append(f)
    examples = []
    for key, items in sorted(grouped.items()):
        energy = math.fsum(x["teacher_sse"] for x in items)
        sse = math.fsum(x["error_sse"] for x in items)
        examples.append(dict(example=list(key), fields=len(items), SSE=sse,
                             teacher_energy=energy, relative_SSE=None if energy == 0 else sse / energy))
    valid = [x for x in examples if x["teacher_energy"] > 0]
    return dict(fields=len(fields), examples=examples, valid_examples=len(valid),
                excluded_examples=len(examples) - len(valid),
                equal_example_pooled_relative_SSE=math.fsum(x["relative_SSE"] for x in valid) / len(valid))


def score_native_fields(fields, contract, *, teacher_membership):
    """Primary plus unchanged OFF/teacher/reverse tensor oracles."""
    primary = pool_primary(fields, contract, teacher_membership=teacher_membership)
    oracles = {}
    for name, expected in (("OFF", 1.), ("teacher", 0.), ("reverse", 4.)):
        values = [dict(f, error_sse=f["oracle_SSE"][name]) for f in fields]
        oracles[name] = pool_primary(values, contract, teacher_membership=teacher_membership)
        _require(oracles[name]["equal_example_pooled_relative_SSE"] == expected,
                 "actual native oracle differs: " + name)
    return dict(primary=primary, oracles=oracles)
