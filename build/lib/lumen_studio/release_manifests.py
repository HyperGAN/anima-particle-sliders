"""Read the hash-bound train/development manifests of the expanded release."""
import json
from pathlib import Path
from .contracts import digest
from .dataset import load_characters

ROOT=Path(__file__).resolve().parents[1]


def validate_manifest(manifest):
    plan=json.loads((ROOT/'data/expansion-plan.json').read_text())
    entry=next((e for e in plan['sliders'] if e['training_id']==manifest.get('variation')),None)
    split=manifest.get('split')
    if entry is None or split not in ('train','dev'):
        raise ValueError('Expected a released training/development manifest')
    expected=entry['manifests'][split]
    if manifest.get('sha256')!=expected or digest({k:v for k,v in manifest.items() if k!='sha256'})!=expected:
        raise ValueError('Archived release manifest changed')
    characters={c['id'] for c in load_characters() if c['split']==split}
    if any(r['character'] not in characters or r['split']!=split or r['variation']!=entry['training_id']
           for r in manifest['rows']):
        raise ValueError('Character split leakage')


def manifest_for(public_id,split):
    if split not in ('train','dev'):
        raise ValueError('Release reproduction uses training/development only')
    manifest=json.loads((ROOT/f'data/{public_id}-{split}.json').read_text())
    validate_manifest(manifest)
    return manifest
