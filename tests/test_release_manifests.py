"""Reproduction must reject edited or test-split training manifests."""
import copy
import pytest
from lumen_studio.release_manifests import manifest_for,validate_manifest


@pytest.mark.parametrize('name',['final-form','afterimage','dusk'])
def test_archived_manifest_integrity(name):
    for split in ('train','dev'):
        manifest=manifest_for(name,split)
        changed=copy.deepcopy(manifest)
        changed['rows'][0]['positive']+=' altered'
        with pytest.raises(ValueError,match='changed'):
            validate_manifest(changed)
    with pytest.raises(ValueError,match='training/development'):
        manifest_for(name,'test')
