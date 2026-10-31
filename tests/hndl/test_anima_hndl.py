"""Installed opt-in HyperGAN/HNDL cohort; tiny CPU models, no pretrained loads."""
from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import sysconfig

import pytest
import torch
from torch import nn

if version('particle-sliders-core') != '0.12.0':
    pytest.skip('separate core0.12/HNDL0.10 candidate environment required', allow_module_level=True)

from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig, UnsupportedOperation, open_session
from particle_sliders.pg09 import _hash, validate_pg09_domain
from anima_hndl import lifecycle as candidate
from lumen_studio.particles import Branch
from lumen_studio.pg09_training import NativeFieldCritic
from test_image_lifecycle import configuration
from test_pg09_training import equal, runtime_fixture as legacy_runtime_fixture
from test_input_admission import stage_model_admission_fixture, stage_owned_runtime as staged_model


def runtime_fixture():
    # Actual candidate runtime class, using generated Cosmos state without any
    # loader. Native identity must retain this class's real source identity.
    original = legacy_runtime_fixture()
    runtime = candidate.LocalTurboRuntime.__new__(candidate.LocalTurboRuntime)
    runtime.__dict__.update(original.__dict__)
    return runtime


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def affine(destination, source):
    with torch.no_grad():
        destination.weight.copy_(source.weight)
        if source.bias is not None:
            destination.bias.copy_(source.bias)


@pytest.mark.parametrize('shape', [(2, 5), (2, 3, 5), (2, 3, 4, 5)])
def test_actual_product_branch_output_input_and_bank_gradient_parity(shape):
    torch.manual_seed(21)
    old = Branch(5, 6)
    builder, identity = candidate.verified_builder()
    new = candidate.Branch(nn.Linear(5, 6), builder, identity)
    nn.init.normal_(old.up.weight, std=.2)
    affine(new.graph['down'], old.down)
    affine(new.graph['up'], old.up)
    for prefix, layers in [('router', old.routed.router), ('bridge', old.routed.net)]:
        for index in (0, 2, 4, 6):
            affine(new.graph[prefix + str(index)], layers[index])
    x = torch.randn(shape, requires_grad=True)
    bank = torch.randn(128, 4, requires_grad=True)
    other_x, other_bank = x.detach().clone().requires_grad_(), bank.detach().clone().requires_grad_()
    expected, actual = old(x, bank), new(other_x, other_bank)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    old_grad = torch.autograd.grad(expected.square().sum(), (x, bank))
    new_grad = torch.autograd.grad(actual.square().sum(), (other_x, other_bank))
    for actual_grad, expected_grad in zip(new_grad, old_grad):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=3e-5)
    assert new_grad[1].abs().sum() > 0


@pytest.mark.parametrize('spatial', [(1, 2, 2), (1, 3, 4)])
def test_actual_patch_critic_output_and_penalty_second_derivative_parity(spatial):
    old, new = NativeFieldCritic(4, 12, 8), candidate.NativeFieldCritic(4, 12, spatial, 8)
    for key, layer in [('field0', old.field[0]), ('field2', old.field[2]),
                       ('context0', old.context[0]), ('logits0', old.logits[0]), ('logits2', old.logits[2])]:
        affine(new.graph[key], layer)
    x = torch.randn(2, 4, *spatial, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    condition = torch.randn(2, 12)
    expected, actual = old(x, condition), new(other, condition)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    a = torch.autograd.grad(expected.sum(), x, create_graph=True)[0]
    b = torch.autograd.grad(actual.sum(), other, create_graph=True)[0]
    torch.testing.assert_close(b, a, atol=2e-7, rtol=2e-5)
    aa = torch.autograd.grad(a.square().sum(), old.field[0].weight)[0]
    bb = torch.autograd.grad(b.square().sum(), new.graph['field0'].weight)[0]
    torch.testing.assert_close(bb, aa, atol=2e-7, rtol=3e-5)


def config_for(tmp_path, runtime):
    old = configuration(tmp_path, runtime)
    return SessionConfig('anima-hndl', 'anima-hypergan-hndl-v2-cpu', **old.options)


def test_candidate_and_shared_core_import_from_installed_wheels():
    import particle_sliders
    for module in (candidate, particle_sliders):
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())


def test_real_shared_session_resume_ownership_ema_and_native_only_inference(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    config = config_for(tmp_path, runtime)
    frozen = deepcopy(runtime.transformer.state_dict())
    rng = torch.get_rng_state().clone()
    with open_session(config, runtime=runtime) as session:
        assert isinstance(session.game.critic, candidate.NativeFieldCritic)
        assert isinstance(session.game.backend.adapter, candidate.ParticleAdapter)
        adapter = session.game.backend.adapter
        leaves = [p for group in session.game.opt_g.param_groups for p in group['params']]
        assert len(leaves) == len({id(p) for p in leaves})
        assert {id(p) for p in leaves} == {id(p) for p in adapter.parameters()}
        assert session.game.prior.z is adapter.particles
        bank = adapter.particles.detach().clone()
        session.train(stop_after=2)
        assert not torch.equal(bank, adapter.particles)
        packet = session.checkpoint(tmp_path / 'training.pt')
        assert session.game.state_dict()['config']['pg_version'] == '0.9.4'
        assert 'hypergan-hndl-v2' in json.dumps(packet, default=str)
        before = session.checkpoint()
        bad = deepcopy(before)
        bad['manifest']['config']['cohort'] = 'historical-anima-cohort'
        bad['sha256'] = _hash({key: value for key, value in bad.items() if key != 'sha256'})
        with pytest.raises(CheckpointError):
            session.restore(bad)
        assert equal(before, session.checkpoint())
        session.game.network.zero_grad(set_to_none=True)
        session.game.critic.zero_grad(set_to_none=True)
        with session.game.network.scaled(.5):
            response = session.game.backend.generated(session.game.rows[0])
            response.float().sum().backward()
        assert adapter.particles.grad is not None and adapter.particles.grad.abs().sum() > 0
        assert all(p.grad is None for p in runtime.transformer.parameters())
        assert all(p.grad is None for p in session.game.critic.parameters())
        request = dict(latent=session.game.rows[0].row.latent.clone(),
                       embedding=session.game.rows[0].row.neutral.clone(),
                       timestep=float(session.game.rows[0].row.timestep))
        with runtime.mixer.scales({}):
            base = runtime.predict(**request)
        assert torch.equal(session.evaluate(strength=0.), base)
        expected_inference = {}
        raw = deepcopy(session.game.network.state_dict())
        for selection in ('raw', 'ema'):
            session.export_inference(tmp_path / (selection + '.pt'), selection=selection)
            session.game.network.load_state_dict(raw if selection == 'raw' else session.ema.values)
            expected_inference[selection] = session.evaluate(strength=.5)
            session.game.network.load_state_dict(raw)
        expected_metrics = session.train()
        expected = session.checkpoint()
        assert equal(frozen, runtime.transformer.state_dict())
    assert not runtime.mixer.adapters and not runtime.mixer.handles
    assert torch.equal(rng, torch.get_rng_state())
    other = runtime_fixture()
    resume_rng = torch.get_rng_state().clone()
    with open_session(config, resume=tmp_path / 'training.pt', runtime=other) as session:
        assert equal(packet, session.checkpoint())
        assert equal(expected_metrics, session.train())
        assert equal(expected, session.checkpoint())
    assert torch.equal(resume_rng, torch.get_rng_state())
    runtime.mixer.close()
    other.mixer.close()
    shutil.rmtree(config.options['cache_dir'])
    options, loaded = staged_model(tmp_path, monkeypatch)
    stage_model_admission_fixture(options['model_dir'], monkeypatch)
    from particle_sliders.pg09 import ConditionalPG09
    def forbidden(*args, **kwargs):
        pytest.fail('native-only inference constructed training state')
    monkeypatch.setattr(ConditionalPG09, '__init__', forbidden)
    monkeypatch.setattr(candidate.NativeFieldCritic, '__init__', forbidden)
    wrong_shape = dict(request, latent=torch.zeros(1, 4, 1, 3, 2))
    with pytest.raises(ValueError, match='fixed H/W'):
        open_inference(tmp_path / 'raw.pt', options=options,
                       operation='response', native_request=wrong_shape)
    assert not loaded
    inference_rng = torch.get_rng_state().clone()
    for selection in ('raw', 'ema'):
        with open_inference(tmp_path / (selection + '.pt'), options=options,
                            operation='response', native_request=request) as native:
            assert native.selection == selection
            adapter = loaded[-1].mixer.adapters['finalform']
            assert isinstance(adapter, candidate.ParticleAdapter)
            assert {id(p) for p in native.binding.network.parameters()} == {id(p) for p in adapter.parameters()}
            assert native.binding.identity['protocol'] == candidate.PROTOCOL
            assert 'anima_hndl.lifecycle' in native.binding.identity['sources']
            assert 'product_sources_sha256' in json.dumps(native.binding.identity)
            assert torch.equal(native.response(strength=0., **request), base)
            assert torch.equal(native.response(strength=.5, **request), expected_inference[selection])
            with pytest.raises(UnsupportedOperation):
                native.train()
        assert not loaded[-1].mixer.adapters and not loaded[-1].mixer.handles
        assert not hasattr(loaded[-1], 'pipe')
    assert torch.equal(inference_rng, torch.get_rng_state())


def test_exact_builder_refusal_before_attachment_and_preserved_domain_guard(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    config = config_for(tmp_path, runtime)
    import particle_sliders.fibo_hndl as shared
    original = shared._sha
    monkeypatch.setattr(shared, '_sha', lambda p: '0' * 64 if str(p).endswith('hndl_networks.py') else original(p))
    with pytest.raises(ValueError, match='builder closure'):
        open_session(config, runtime=runtime)
    assert not runtime.mixer.adapters and not runtime.mixer.handles
    with pytest.raises(ValueError, match='width >=170'):
        validate_pg09_domain((1, 170), response_domain='continuous_vector')
    runtime.mixer.close()


def test_candidate_accelerator_request_refuses_before_builder_or_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('accelerator request reached builder/model work')
    monkeypatch.setattr(candidate, 'verified_builder', forbidden)
    with pytest.raises(UnsupportedOperation, match='requires CPU'):
        candidate.BACKEND.open(SessionConfig('anima-hndl', 'refused', device='cuda:0'))


@pytest.mark.parametrize('relative', ['lumen_studio/particles.py', 'lumen_studio/cache.py',
                                    'lumen_studio/pg09_training.py', 'particle_sliders/recipe.py'])
def test_complete_installed_source_closure_rejects_actual_file_changes(relative):
    path = Path(sysconfig.get_paths()['purelib']) / relative
    original = path.read_bytes()
    try:
        path.write_bytes(original + b'\n# source-closure rejection fixture\n')
        with pytest.raises(ValueError, match='complete .* source closure differs'):
            candidate.verified_builder()
    finally:
        path.write_bytes(original)


def test_exact_core_provenance_and_isolation_refusals(monkeypatch):
    from anima_hndl import source_validation as validation
    actual_distribution = validation.distribution
    core = actual_distribution('particle-sliders-core')
    class WrongProvenance:
        version = core.version
        def read_text(self, name):
            if name == 'direct_url.json':
                return json.dumps({'url': 'https://github.com/HyperGAN/particle-sliders.git',
                    'subdirectory': 'packages/particle-sliders-core',
                    'vcs_info': {'vcs': 'git', 'commit_id': '0' * 40}})
            return core.read_text(name)
    with monkeypatch.context() as scoped:
        scoped.setattr(validation, 'distribution', lambda name: WrongProvenance()
                       if name == 'particle-sliders-core' else actual_distribution(name))
        with pytest.raises(ValueError, match='Git/wheel provenance'):
            candidate.verified_builder()
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, 'path', sys.path + ['/tmp/unqualified/site-packages'])
        with pytest.raises(RuntimeError, match='shared dependency'):
            candidate.verified_builder()
    import numpy
    with monkeypatch.context() as scoped:
        scoped.setattr(numpy, '__file__', '/tmp/own-shadow-fixture/numpy.py')
        with pytest.raises(RuntimeError, match='loaded dependency origin'):
            candidate.verified_builder()


def test_model_load_cannot_replace_configured_cache_index(tmp_path, monkeypatch):
    runtime = runtime_fixture()
    config = config_for(tmp_path, runtime)
    options, loaded = staged_model(tmp_path, monkeypatch)
    stage_model_admission_fixture(options['model_dir'], monkeypatch)
    config = SessionConfig(config.backend, config.cohort, **dict(config.options, model_dir=options['model_dir']))
    original = candidate.LocalTurboRuntime.__init__
    def changed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        (Path(config.options['cache_dir']) / 'index.json').write_text('{}')
    monkeypatch.setattr(candidate.LocalTurboRuntime, '__init__', changed)
    with pytest.raises(ValueError, match='index differs'):
        open_session(config)
    assert not loaded[-1].mixer.adapters and not loaded[-1].mixer.handles
    assert not hasattr(loaded[-1], 'pipe')
    runtime.mixer.close()
