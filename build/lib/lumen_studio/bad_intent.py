"""Validate the exact archived manifests behind the public Bad Intent release."""
import json
from pathlib import Path
from .contracts import digest
from .psychological_horror import compile_manifest

ARCHIVES = {
    'train': ('ea3e8808797c6792c5ec111b6cbfd897b5f5fdd9914e1aee5591e02880f823ad',
              '80c79fc3e6bf599e665826a6b203d095410aee33e88a0540e33135e293cb0b86'),
    'dev': ('e7b2fd89c0c3ec07e9541d708657365a361c13fd4a06fd0e2390235509af0b33',
            '26b209e3659e73c0ef04c73a5976e77c9302c605be36499d20875460064762a2'),
}


def validate_manifest(manifest):
    split = manifest.get('split')
    if split not in ARCHIVES:
        raise ValueError('Bad Intent uses archived training/development manifests only')
    expected_hash, source_hash = ARCHIVES[split]
    expected = compile_manifest(split)
    # The original Studio had revised unrelated Theatrical definitions. The
    # Candlelit-derived shared rows are exactly equal; retain the full-catalog
    # provenance recorded at target preparation, without changing old releases.
    expected['shared_source_sha256'] = source_hash
    expected['sha256'] = digest({k:v for k,v in expected.items() if k != 'sha256'})
    if expected['sha256'] != expected_hash or manifest != expected:
        raise ValueError('Experimental manifest changed shared content, coverage, or provenance')


def manifest_for(split):
    if split not in ARCHIVES:
        raise ValueError('Bad Intent uses archived training/development manifests only')
    manifest = json.loads((Path(__file__).resolve().parents[1]/f'data/bad-intent-{split}.json').read_text())
    validate_manifest(manifest)
    return manifest
