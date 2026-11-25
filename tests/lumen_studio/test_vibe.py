from contextlib import contextmanager
import json
from pathlib import Path

import pytest

pytest.importorskip('particle_sliders_vibe')
pytest.importorskip('nanoodle')

from lumen_studio.dataset import validate_manifest
from lumen_studio.vibe import AnimaVibe, training_manifest, workflow_graphs
from particle_sliders_vibe import collect
from particle_sliders_vibe.nanoodle import NanoodleWorkflows


class Workflows:
    identity = {'provider': 'fixture'}
    def propose(self, vibe, count, history, **kw):
        return [dict(neutral=f'knight {len(history)} {i}', positive=f'boss knight {len(history)} {i}') for i in range(count)]
    def score(self, vibe, output, **kw):
        return {'score': 0.9 if output['prompt'].startswith('boss') else 0.2}


class Mixer:
    @contextmanager
    def scales(self, values):
        assert values == {}
        yield


class Runtime:
    identity = {'model': 'render-fixture-not-pretrained'}
    mixer = Mixer()
    def render(self, prompt, seed, **kwargs):
        from PIL import Image
        return Image.new('RGB', (16, 16), 'red')
    def close(self):
        pass


def test_collect_to_native_manifest_and_cache_validation(tmp_path):
    product = AnimaVibe(tmp_path, Workflows(), runtime=Runtime())
    state = collect(product, vibe='fantasy rpg final boss, epic battle scene', directory=tmp_path / 'run',
                    pool_size=2, replace=1, best_of=2, rounds=2)
    manifest = training_manifest(state)
    validate_manifest(manifest)
    assert len(manifest['rows']) == 4
    assert len({r['id'] for r in manifest['rows']}) == 4
    assert all(r['split'] == 'train' and r['seeds'] for r in manifest['rows'])
    assert manifest['collection_identity']['selection'] == 'raw'
    # The existing target-cache entrypoint must admit the new manifest directly.
    from lumen_studio.cache import prepare_targets
    class StopBeforeModelMath:
        identity = product.identity['model']
        @property
        def mixer(self):
            raise RuntimeError('reached native cache after validation')
    with pytest.raises(RuntimeError, match='reached native cache'):
        prepare_targets(StopBeforeModelMath(), manifest, manifest['variation'], tmp_path / 'targets')
    media = Path(state['selected'][0]['positive']['output']['path'])
    media.write_bytes(b'changed')
    with pytest.raises(ValueError, match='media changed'):
        training_manifest(state)


def test_no_cpu_rendering(tmp_path):
    with pytest.raises(ValueError, match='CUDA'):
        AnimaVibe(tmp_path, Workflows(), device='cpu', runtime=Runtime())


@pytest.mark.parametrize('resolution', [256, 448, 513, 1600, 2048, True])
def test_resolution_refuses_before_model_identity_or_loading(tmp_path, resolution):
    with pytest.raises(ValueError, match='native 512–1536'):
        AnimaVibe(tmp_path / 'missing-model', Workflows(), resolution=resolution)


def test_no_qualified_outputs_is_explicit(tmp_path):
    workflows = Workflows()
    workflows.score = lambda *a, **kw: {'score': 0.2}
    state = collect(AnimaVibe(tmp_path, workflows, runtime=Runtime()), vibe='boss',
                    directory=tmp_path / 'run', pool_size=1, replace=1)
    with pytest.raises(ValueError, match='no qualifying'):
        training_manifest(state)
    assert (tmp_path / 'run' / 'collection.json').is_file()


def test_best_qualifying_context_is_retained_even_when_largest_gain_is_below_threshold(tmp_path):
    workflows = Workflows()
    def score(vibe, output, **kw):
        positive = output['prompt'].startswith('boss')
        return {'score': (.7 if positive else .05) if output['seed'] == 7 else (.9 if positive else .4)}
    workflows.score = score
    state = collect(AnimaVibe(tmp_path, workflows, runtime=Runtime()), vibe='boss',
                    directory=tmp_path / 'run', pool_size=1, replace=1, best_of=2)
    assert state['selected'][0]['seed'] == 7
    manifest = training_manifest(state)
    assert manifest['rows'][0]['seeds'] == [8]


@pytest.mark.parametrize('provider', ['nanogpt', 'typesafe'])
@pytest.mark.parametrize('concept', ['Final Boss', 'Candlelit'])
def test_nanoodle_graphs_run_real_executor_with_injected_transport(tmp_path, monkeypatch, provider, concept):
    """Exercise actual Nanoodle graph wiring, media encoding and typed decision parsing."""
    from nanoodle.transport import HttpResponse
    requests = []
    def http(method, url, **kwargs):
        requests.append((method, url, kwargs))
        if url.endswith('/chat/completions'):
            data = {'choices': [{'message': {'content': json.dumps([dict(neutral='knight', positive='boss knight')])}}]}
        elif url.endswith('/api/v1/decisions') or url == 'https://api.typesafe.ai/v1/systemone':
            data = {'answers': {'answer': {'type': 'score', 'score': 2.67,
                    'probabilities': {'0': 0, '1': 0, '2': .33, '3': .67}, 'confidence': .67}},
                    'usage': {'cost': .001}}
        else:
            raise AssertionError(url)
        return HttpResponse(200, {'content-type': 'application/json'}, json.dumps(data).encode())
    monkeypatch.setenv('NANOGPT_API_KEY', 'test-key-never-real')
    monkeypatch.setenv('TYPESAFE_API_KEY', 'fixture-typesafe')
    pgraph, sgraph = workflow_graphs('fixture-writer', 'fixture-jev', concept)
    workflows = NanoodleWorkflows(pgraph, sgraph, judge_provider=provider)
    workflows.prompts.http = http
    if provider == 'typesafe':
        monkeypatch.setattr('nanoodle.transport.default_http', http)
    else:
        workflows.judge.http = http
    assert workflows.propose(concept, 1, [], request_id='request')[0]['positive'] == 'boss knight'
    from PIL import Image
    image = tmp_path / 'image.png'
    Image.new('RGB', (16, 16)).save(image)
    score = workflows.score(concept, dict(path=str(image)), request_id='score')
    assert score['score'] == pytest.approx(.89)
    assert score['decision']['model'] == 'fixture-jev'
    assert len(requests) == 2
    body = json.loads(requests[-1][2]['body'])
    question = body['questions']['answer']
    assert question['instructions'] == f'How well does the candidate convey ‘{concept}’ without breaking the image?'
    assert question['criteria'] == [
        f'No convincing {concept} effect, or the image is visibly broken.',
        f'A weak suggestion of {concept} in a coherent image.',
        f'Clearly conveys {concept} in a coherent, intact image.',
        f'Unmistakably conveys {concept} in a coherent, intact image.',
    ]
    if provider == 'typesafe':
        _, url, kwargs = requests[-1]
        assert url == 'https://api.typesafe.ai/v1/systemone'
        assert kwargs['headers']['Authorization'] == 'Bearer fixture-typesafe'
        assert 'x-api-key' not in kwargs['headers']
        body = json.loads(kwargs['body'])
        assert body['model'] == 'fixture-jev'
        assert body['state'][0] == concept
        assert body['state'][1] in body['assets']
        assert score['raw']['answers']['answer']['score'] == 2.67


def test_typesafe_rejection_is_reported_as_typesafe_not_nanogpt(tmp_path, monkeypatch):
    from nanoodle.transport import HttpResponse
    from particle_sliders_vibe import ProviderRejected
    monkeypatch.setattr('nanoodle.transport.default_http', lambda *a, **k: HttpResponse(402, {}, b'{}'))
    monkeypatch.setenv('TYPESAFE_API_KEY', 'fixture-typesafe')
    workflows = NanoodleWorkflows(*workflow_graphs('writer', 'jev-latest', 'boss'), judge_provider='typesafe')
    from PIL import Image
    image = tmp_path / 'image.png'
    Image.new('RGB', (16, 16)).save(image)
    with pytest.raises(ProviderRejected, match='TypeSafe.*402'):
        workflows.score('boss', {'path': str(image)}, request_id='fixture')


def test_missing_provider_score_is_not_treated_as_zero(tmp_path, monkeypatch):
    from nanoodle.transport import HttpResponse
    payload = {'answers': {'answer': {'type': 'score', 'probabilities': {'0': 1, '1': 0, '2': 0, '3': 0}}}}
    monkeypatch.setattr('nanoodle.transport.default_http', lambda *a, **k: HttpResponse(200, {}, json.dumps(payload).encode()))
    monkeypatch.setenv('TYPESAFE_API_KEY', 'fixture-typesafe')
    workflows = NanoodleWorkflows(*workflow_graphs('writer', 'jev-latest', 'boss'), judge_provider='typesafe')
    from PIL import Image
    image = tmp_path / 'image.png'
    Image.new('RGB', (16, 16)).save(image)
    with pytest.raises(ValueError, match='no valid measured score'):
        workflows.score('boss', {'path': str(image)}, request_id='fixture')


def test_manifest_does_not_allow_cache_path_escape(tmp_path):
    from lumen_studio.contracts import digest
    state = collect(AnimaVibe(tmp_path, Workflows(), runtime=Runtime()), vibe='boss',
                    directory=tmp_path / 'run', pool_size=1, replace=1)
    manifest = training_manifest(state)
    manifest['rows'][0]['id'] = '../escaped'
    manifest['sha256'] = digest({k: v for k, v in manifest.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='row identity'):
        validate_manifest(manifest)


def test_finished_target_cache_is_left_byte_identical(tmp_path):
    from conftest import TinyRuntime
    from lumen_studio.contracts import file_hash
    from lumen_studio.vibe import prepare_snapshot

    runtime = TinyRuntime()
    product = AnimaVibe(tmp_path, Workflows(), runtime=Runtime())
    state = collect(product, vibe='boss', directory=tmp_path / 'run', pool_size=1, replace=1, best_of=1)
    manifest = training_manifest(state)
    targets = tmp_path / 'snapshot' / 'training' / 'targets'
    loads = []

    def load():
        loads.append(1)
        return runtime

    first = prepare_snapshot(load, manifest, targets, model=runtime.identity, resolution=64)
    saved = file_hash(targets / 'index.json')
    again = prepare_snapshot(load, manifest, targets, model=runtime.identity, resolution=64)
    assert again == first and file_hash(targets / 'index.json') == saved and len(loads) == 1
    # A damaged or foreign cache is not taken as finished; the pinned preparer decides.
    shard = targets / first['shards'][0]['path']
    shard.write_bytes(shard.read_bytes() + b'x')
    prepare_snapshot(load, manifest, targets, model=runtime.identity, resolution=64)
    assert len(loads) == 2
    assert prepare_snapshot(load, manifest, targets, model=dict(runtime.identity, sha256='b' * 64),
                            resolution=64) is not None and len(loads) == 3
