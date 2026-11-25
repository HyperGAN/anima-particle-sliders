"""Bind a Jev-only alpha search to each of the saved native RAW192 adapters."""
from hashlib import file_digest
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent
PRIOR=CAMPAIGN.parent/'anima-final-boss-research-raw192-20261123'
METHODS=('shared-best-seed-split','shared-best-stock','shared-best-divergence')
QUESTION='How well does the candidate convey ‘Final Boss’ without breaking the image?'


def sha(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def main():
    source=PRIOR/'prompt-only-alpha-tune/prompt-only-tuned.json'
    template=json.loads(source.read_text())
    panel={row['label']:row for row in json.loads((PRIOR/'panel.json').read_text())}
    cases=[]
    for case in template['cases']:
        name=case['id']; context=panel[name]
        off=json.loads((PRIOR/'shared-off'/name/'render.json').read_text())['image']
        image=Path(off['path'])
        if sha(image)!=off['sha256']:
            raise ValueError(f'The reference image changed: {name}')
        off=dict(path=str(image),sha256=off['sha256'],bytes=image.stat().st_size)
        cases.append(case|dict(request=dict(prompt=context['prompt'],seed=context['seed']),off=off))
    definitions={}
    for method in METHODS:
        result=json.loads((CAMPAIGN/'cohorts'/method/'result.json').read_text()); raw=result['raw']
        path=Path(raw['path']); sidecar=json.loads(path.with_suffix('.scoped.json').read_text())
        if (result['step']!=192 or result['ema_decay'] is not None or sidecar['native_RAW_sha256']!=raw['sha256']
                or path.stat().st_size!=raw['bytes'] or sha(path)!=raw['sha256']):
            raise ValueError(f'{method} lacks its immutable RAW192 adapter')
        definition=template | dict(checkpoint=raw,concept=QUESTION,cases=cases,
             profiles=dict(early=[.25,.4,.55,2/3,.85,1.05,1.25],constant=[.35,.5,.6375,.8,1.,1.2,1.45]),
             refinement_rounds=1)
        path=ROOT/f'{method}.json'
        with path.open('x') as stream:
            json.dump(definition,stream,indent=2,ensure_ascii=False);stream.write('\n')
        definitions[method]={'definition':str(path),'sha256':sha(path),'checkpoint':raw['sha256']}
    with (ROOT/'plan.json').open('x') as stream:
        json.dump(dict(methods=definitions,original=sha(source),concept=QUESTION),stream,indent=2);stream.write('\n')

if __name__=='__main__':
    main()
