"""Metadata regression controls: real declared manifests, no Torch/model calls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from lumen_studio.contracts import digest
from lumen_studio.coverage import (compile_coverage, field_metadata, pool_primary,
    require_production_rate_coverage, validate_field_membership, validate_index,
    freeze_teacher_membership)

ROOT = Path(__file__).resolve().parents[1]


def index_for(contract):
    identity = {k: contract[k] for k in ("split", "variation", "manifest_sha256")}
    identity["coverage_sha256"] = contract["sha256"]
    shards = [dict(s, sha256="a" * 64) for s in contract["shards"]]
    return dict(identity=identity, input_fingerprint=digest(identity), shards=shards,
                fingerprint=digest(dict(input=digest(identity), shards=shards)))


def metadata_for(contract, manifest):
    rows = {r["id"]: r for r in manifest["rows"]}
    return [dict(field_metadata(rows[r], s, path, pos, index=i),
                 error_sse=2., teacher_sse=2.)
            for i, (r, s, path, pos) in enumerate(contract["field_keys"])]


class ProductionCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifests = {s: json.loads((ROOT / "data" / (s + ".json")).read_text())
                         for s in ("train", "dev")}
        cls.contracts = {s: compile_coverage(m, "candlelit") for s, m in cls.manifests.items()}
        cls.indexes = {s: index_for(c) for s, c in cls.contracts.items()}
        cls.fields = {s: metadata_for(c, cls.manifests[s]) for s, c in cls.contracts.items()}

    def test_full_actual_catalog_coverage_and_no_torch(self):
        self.assertEqual([self.contracts[s]["field_count"] for s in ("train", "dev")], [7680, 480])
        self.assertEqual([self.contracts[s]["example_count"] for s in ("train", "dev")], [768, 48])
        self.assertEqual([len(self.contracts[s]["definitions"]) for s in ("train", "dev")], [6, 8])
        self.assertEqual(len(self.contracts["dev"]["bare_rows"]), 4)
        require_production_rate_coverage(self.manifests, self.contracts, self.indexes, self.fields)
        self.assertNotIn("torch", sys.modules)

    def test_original_two_shard_pilot_refused(self):
        index = deepcopy(self.indexes["train"])
        index["shards"] = [index["shards"][0], index["shards"][16]]
        index["fingerprint"] = digest(dict(input=index["input_fingerprint"], shards=index["shards"]))
        with self.assertRaisesRegex(ValueError, "membership"):
            validate_index(index, self.contracts["train"])

    def test_missing_seed_duplicate_or_extra_shard_refused(self):
        for mutate in (lambda s: s.pop(), lambda s: s.append(s[0]),
                       lambda s: s.__setitem__(0, dict(s[0], seed=999999))):
            index = deepcopy(self.indexes["dev"])
            mutate(index["shards"])
            index["fingerprint"] = digest(dict(input=index["input_fingerprint"], shards=index["shards"]))
            with self.assertRaisesRegex(ValueError, "membership"):
                validate_index(index, self.contracts["dev"])

    def test_production_has_no_subset_switch(self):
        with self.assertRaisesRegex(ValueError, "rejects pilot"):
            compile_coverage(self.manifests["train"], "candlelit",
                             pilot_rows=["candlelit-train-00"], pilot_seeds=[1000])

    def test_explicit_pilot_never_production_eligible(self):
        c = compile_coverage(self.manifests["train"], "candlelit", mode="pilot",
                             pilot_rows=["candlelit-train-00", "candlelit-train-01"], pilot_seeds=[1000])
        self.assertEqual(c["field_count"], 40)
        self.assertIs(c["production_rate_eligible"], False)
        contracts = dict(self.contracts, train=c)
        with self.assertRaisesRegex(ValueError, "pilot"):
            require_production_rate_coverage(self.manifests, contracts, self.indexes, self.fields)

    def test_rehashed_fake_production_subset_refused(self):
        c = compile_coverage(self.manifests["train"], "candlelit", mode="pilot",
                             pilot_rows=["candlelit-train-00"], pilot_seeds=[1000])
        c.update(mode="production", production_rate_eligible=True)
        c["sha256"] = digest({k: v for k, v in c.items() if k != "sha256"})
        contracts = dict(self.contracts, train=c)
        with self.assertRaisesRegex(ValueError, "complete declared manifest"):
            require_production_rate_coverage(self.manifests, contracts, self.indexes, self.fields)

    def test_character_only_example_collapses_full_dev_and_refuses(self):
        fields = deepcopy(self.fields["dev"])
        for f in fields:
            f["example"][0] = f["subject"]
        with self.assertRaisesRegex(ValueError, "not only character"):
            validate_field_membership(fields, self.contracts["dev"])

    def test_missing_or_duplicate_position_refused(self):
        fields = deepcopy(self.fields["dev"])
        fields[-1]["position"] = 8
        with self.assertRaisesRegex(ValueError, "all-position"):
            validate_field_membership(fields, self.contracts["dev"])

    def test_metric_keeps_distinct_prompt_examples_and_teacher_only_zero(self):
        fields = deepcopy(self.fields["dev"])
        example = fields[0]["example"]
        for f in fields:
            if f["example"] == example:
                f.update(teacher_sse=0., error_sse=3.)
        frozen = freeze_teacher_membership(fields, self.contracts["dev"])
        score = pool_primary(fields, self.contracts["dev"], teacher_membership=frozen)
        self.assertEqual(score["valid_examples"], 47)
        self.assertEqual(score["excluded_examples"], 1)
        self.assertEqual(score["equal_example_pooled_relative_SSE"], 1.)
        fields[-1]["error_sse"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid field SSE"):
            pool_primary(fields, self.contracts["dev"], teacher_membership=frozen)

    def test_student_cannot_change_frozen_teacher_membership(self):
        fields = deepcopy(self.fields["dev"])
        frozen = freeze_teacher_membership(fields, self.contracts["dev"])
        fields[0]["teacher_sse"] = 0.
        with self.assertRaisesRegex(ValueError, "pre-student freeze"):
            pool_primary(fields, self.contracts["dev"], teacher_membership=frozen)

    def test_native_preparer_refuses_before_torch_or_output(self):
        from lumen_studio.fullcoverage_prepare import CORE, CPU_PROFILE, prepare_fullcoverage
        class Provider:
            def identity(self):
                return dict(profile=CPU_PROFILE, CPU_encoder_unloaded=True)
            def pair(self, row):
                raise AssertionError("no conditioning payload access before admission")
        authority = dict(core_commit=CORE, core_version="0.13.0", PG="0.9.5",
                         session_weight_EMA=None, materialized_EMA_shadows=False, selection="raw")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "unadmitted"
            with self.assertRaisesRegex(ValueError, "admission refused"):
                prepare_fullcoverage(None, self.manifests, out, variation="candlelit",
                    conditioning_provider=Provider(), owner_check=lambda: None,
                    admission=lambda request: False, source_authority=authority, native_profile={})
            self.assertFalse(out.exists())
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
