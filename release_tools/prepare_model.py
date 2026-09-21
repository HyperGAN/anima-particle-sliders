"""Convert pinned official files and verify the released portable model identity."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lumen_studio.prepare import convert
from lumen_studio.contracts import atomic_json,file_hash
from lumen_studio.provenance import model_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('artifacts/model'))
    args=p.parse_args()
    target=convert(args.output)
    expected=json.loads((ROOT/'configs/anima/converted-model.lock.json').read_text())
    for name,sha in expected['files'].items():
        if name=='modular_model_index.json':continue
        if file_hash(target/name)!=sha:raise ValueError(f'Converted file differs: {name}')
    # This field records the original conversion environment, not the active
    # inference environment (which is hashed separately by model_identity).
    expected['files']['modular_model_index.json']=file_hash(target/'modular_model_index.json')
    before=(target/'anima-lock.json').read_bytes()
    atomic_json(target/'anima-lock.json',expected)
    released=json.loads((ROOT/'configs/anima/released-identity.json').read_text())
    if model_identity(target)!=released:
        (target/'anima-lock.json').write_bytes(before)
        raise ValueError('Conversion does not reproduce the released portable identity')
    print('Verified',target)

if __name__=='__main__':main()
