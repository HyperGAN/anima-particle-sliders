"""Resolve recipe-specific frozen target transforms from verified declarations."""
from importlib.util import module_from_spec, spec_from_file_location
from functools import lru_cache
from pathlib import Path

from control import ROOT
from lumen_studio.contracts import file_hash
from optimization_worker import read


@lru_cache
def prepared(method):
    path = ROOT / "ready" / f"{method}.json"
    entry = read(path)
    module = Path(entry["implementation"]["path"])
    if file_hash(module) != entry["implementation"]["sha256"]:
        raise ValueError(f"Recipe implementation no longer matches {path}")
    for item in entry.get("inputs", []):
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError(f"Immutable recipe input no longer matches {path}")
    spec = spec_from_file_location(f"recipe_{method.replace('-', '_')}", module)
    result = module_from_spec(spec)
    spec.loader.exec_module(result)
    return result, entry
