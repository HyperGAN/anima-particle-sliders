"""Queue matched release samples through the existing Studio worker."""
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
API = 'http://127.0.0.1:8876/api'
OUT = ROOT.parent / 'samples-jobs.json'

def main():
    from lumen_studio.sampling import selection_rows
    if OUT.exists():
        print('Using existing', OUT)
        return
    catalog = requests.get(API + '/catalog').json()['checkpoints']
    result = []
    for variation in ('candlelit', 'moonlit'):
        checkpoint = next(c for c in catalog if c['variation'] == variation and c['metadata']['step'] == 1600)
        rows = selection_rows(variation)
        for case, row in enumerate([rows[0], rows[2], rows[4], rows[-1]]):
            for strength in range(6):
                payload = dict(prompt=row['neutral'], seed=row['seeds'][0], width=768, height=768,
                    steps=10, energy=strength, mix={variation:1.}, checkpoints={variation:checkpoint['sha256']})
                response = requests.post(API + '/generate', json=payload)
                response.raise_for_status()
                result.append(dict(variation=variation, case=case, strength=strength, payload=payload,
                    jobs=response.json(), source_row=row))
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print('Queued',len(result),'matched renders:',OUT)

if __name__ == '__main__':
    main()
