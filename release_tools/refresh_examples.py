"""Apply recorded development-example replacements while retaining old assets."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def apply_refreshes(catalog):
    path = ROOT/'data/example-refreshes.json'
    if not path.exists():
        return catalog
    for spec in json.loads(path.read_text())['replacements']:
        entry = next((e for e in catalog['sliders'] if e['id'] == spec['slider']), None)
        if entry is None:
            continue
        request = spec['request']
        manifest = json.loads((ROOT/spec['source_manifest']).read_text())
        row = next(r for r in manifest['rows'] if r['id'] == spec['source_row_id'])
        assert row['split'] == 'dev' and row['character'] == spec['character']
        assert spec['final_test_used'] is False and request['seed'] in row['seeds']
        assert spec['prompt_origin'] in ('manifest-neutral', 'authored-action-scene')
        if spec['prompt_origin'] == 'manifest-neutral':
            assert request['prompt'] == row['neutral']
        existing = next((i for i,r in enumerate(entry['render_requests'])
                         if r['case'] == request['case']), None)
        if existing is not None:
            assert entry['render_requests'][existing] == request
            assert entry['example_refreshes'][request['case']] == spec
            continue
        index = next(i for i,r in enumerate(entry['render_requests']) if r['case'] == spec['replaces'])
        archived = copy.deepcopy(dict(request=entry['render_requests'][index],
                                      comparison=entry['comparisons'][index]))
        if archived not in entry.setdefault('example_archive', []):
            entry['example_archive'].append(archived)
        samples = []
        for fmt, strength in [('particles',1), ('lora',1), ('off',0)]:
            image = f"samples/{spec['version']}/{entry['id']}/{request['case']}/{fmt}.png"
            samples.append(dict(case=request['case'], format=fmt, strength=strength,
                image=image, metadata=str(Path(image).with_suffix('.json'))))
        entry['render_requests'][index] = request
        entry['comparisons'][index] = dict(case=request['case'], samples=samples,
            asset=f"assets/{spec['version']}-{entry['id']}-{request['case']}.jpg")
        entry['samples'] = [s for c in entry['comparisons'] for s in c['samples'] if s['format'] != 'lora']
        entry['distilled_samples'] = [s for c in entry['comparisons'] for s in c['samples'] if s['format'] == 'lora']
        entry.setdefault('example_refreshes', {})[request['case']] = spec
        if index == 0:
            entry['featured_comparison'] = entry['comparisons'][0]
            entry['featured_refresh'] = spec
    return catalog


def stage(folder, candidates):
    from release_tools.build import copy as copy_file, sha
    from release_tools.calibrated_release import write_json
    catalog = apply_refreshes(json.loads((folder/'catalog.json').read_text()))
    report = dict(passed=True, nominal_strength=1, final_test_used=False, samples=[])
    for spec in json.loads((ROOT/'data/example-refreshes.json').read_text())['replacements']:
        entry = next(e for e in catalog['sliders'] if e['id'] == spec['slider'])
        comparison = next(c for c in entry['comparisons'] if c['case'] == spec['request']['case'])
        for sample in comparison['samples']:
            source = candidates/'samples'/spec['slider']/spec['candidate_case']/(sample['format']+'.png')
            metadata = json.loads(source.with_suffix('.json').read_text())
            assert metadata['image_sha256'] == sha(source)
            assert all(metadata[k] == spec['request'][k] for k in ('prompt','seed','width','height','steps'))
            expected = entry['particle'] if sample['format']=='particles' else entry['native_lora'] if sample['format']=='lora' else None
            assert metadata['adapter'] == expected
            assert metadata['adapter_sha256'] == (sha(folder/expected) if expected else None)
            assert metadata['strength'] == sample['strength']
            metadata.update(case=sample['case'], request_provenance=spec)
            copy_file(source, folder/sample['image'])
            write_json(folder/sample['metadata'], metadata)
            report['samples'].append(dict(image=sample['image'], sha256=sha(source),
                candidate_case=spec['candidate_case'], exact_candidate_file=True))
    write_json(folder/'catalog.json', catalog)
    copy_file(ROOT/'data/example-refreshes.json', folder/'data/example-refreshes.json')
    write_json(ROOT/'validation/example-refreshes.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    args = parser.parse_args()
    stage(args.folder, args.candidates)
