"""Public naming must not change the psychological-horror training pairs."""
import copy
import json
from pathlib import Path
import pytest
from lumen_studio.dataset import validate_manifest
from lumen_studio.bad_intent import manifest_for as compile_manifest

ROOT=Path(__file__).resolve().parents[1]

def test_archived_horror_manifests_and_split_isolation():
    train,dev=[compile_manifest(s) for s in ('train','dev')]
    for manifest in (train,dev):
        archived=json.loads((ROOT/f"data/bad-intent-{manifest['split']}.json").read_text())
        assert archived==manifest
        validate_manifest(manifest)
        assert manifest['variation']=='uncanny'
        assert manifest['identity_preservation'] is False
    assert not {r['character'] for r in train['rows']} & {r['character'] for r in dev['rows']}
    assert len(train['rows'])==24 and len(dev['rows'])==12
    bad=copy.deepcopy(train);bad['rows'][0]['positive']='changed'
    with pytest.raises(ValueError,match='Experimental manifest changed'):validate_manifest(bad)
    with pytest.raises(ValueError):compile_manifest('test')
