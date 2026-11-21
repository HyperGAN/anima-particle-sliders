"""Explicitly broad concept edits; separate from lighting-only pair contracts."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from .contracts import Shared, digest, file_hash

SCHEMA = "psychological-horror-v1"
SOURCE = Path(__file__).resolve().parents[1] / "configs/anima/candidates/uncanny-v1.json"


def definitions():
    spec = json.loads(SOURCE.read_text())
    if (spec["id"] != "uncanny-v1" or spec["identity_preservation"] is not False
            or spec["edit_scope"] != "expression_pose_framing_atmosphere"
            or [d["split"] for d in spec["definitions"]] != ["train"] * 6 + ["eval"] * 2):
        raise ValueError("Uncanny requires an explicit broad-edit contract and six train/two eval definitions")
    required = {"split", "expression", "pose", "framing", "atmosphere"}
    if any(set(d) != required or any(not v.strip() for v in d.values()) for d in spec["definitions"]):
        raise ValueError("Invalid psychological horror definition")
    return spec["definitions"]


def compile_manifest(split):
    if split not in ("train", "dev"):
        raise ValueError("Experimental pilots use training and development characters only")
    from .dataset import compile_manifest as legacy_manifest
    source = legacy_manifest(split)
    clauses = definitions()
    templates = [r for r in source["rows"] if r["variation"] == "candlelit"]
    rows = []
    for i, template in enumerate(templates):
        di = int(template["definition"].rsplit("-", 1)[1]) - 1
        definition = clauses[di]
        neutral = Shared(**template["shared"])
        marker = "neutral expression, closed mouth"
        if marker not in neutral.character:
            raise ValueError("Expected the neutral expression in the original character catalog")
        positive = Shared(**dict(asdict(neutral),
            character=neutral.character.replace(marker, definition["expression"], 1),
            pose=definition["pose"], framing=definition["framing"]))
        rows.append(dict(template, id=f"uncanny-{split}-{i:02}", variation="uncanny",
            definition=f"uncanny-{di+1:02}", definition_split=definition["split"],
            shared=asdict(neutral), positive_shared=asdict(positive),
            changed_fields=["character.expression", "pose", "framing", "atmosphere"],
            identity_preservation=False, edit_scope="expression_pose_framing_atmosphere",
            neutral_lighting="", positive_lighting=definition["atmosphere"],
            neutral=neutral.prompt(), positive=positive.prompt() + ", " + definition["atmosphere"]))
    if split == "train" and (len(rows) != 24
            or set(Counter(r["character"] for r in rows).values()) != {2}
            or set(Counter(r["definition"] for r in rows).values()) != {4}):
        raise ValueError("Unbalanced concept training coverage")
    manifest = dict(schema=1, extension=SCHEMA, split=split, variation="uncanny",
        source_spec_sha256=file_hash(SOURCE), compiler_sha256=file_hash(__file__),
        shared_source_sha256=source["sha256"], characters_sha256=source["characters_sha256"],
        definitions_sha256=digest(clauses), identity_preservation=False,
        edit_scope="expression_pose_framing_atmosphere", rows=rows)
    manifest["sha256"] = digest(manifest)
    return manifest
