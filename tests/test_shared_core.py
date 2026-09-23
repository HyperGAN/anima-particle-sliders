"""The product trains the shared gmix stamp and pins particle-sliders-core."""
import json
from pathlib import Path

import torch

import particle_sliders.reference as shared_reference
from particle_sliders import fit_routed_down, winning_formulation
from lumen_studio.game import STAMP, TRAINING_CONFIG
from lumen_studio.provenance import runtime_source
from lumen_studio.vendor import reference

ROOT = Path(__file__).resolve().parents[1]
PIN = (
    "particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@"
    "4340e28bed388d50800c469525b460a108091da0#subdirectory=packages/particle-sliders-core"
)


def test_winning_formulation_is_gmix_and_pinned():
    stamp = winning_formulation()
    assert stamp.architecture_id == "gmix"
    assert stamp.formulation_id == "particle-gmix-1600-v2"
    assert stamp.spec["noise_hold_ratio"] == 1.3
    assert PIN in (ROOT / "requirements.txt").read_text()
    assert "concept-slider-core" not in (ROOT / "requirements.txt").read_text()
    lock = json.loads((ROOT / "core.lock.json").read_text())
    assert lock["repository"] == "https://github.com/HyperGAN/particle-sliders"
    assert lock["commit"] == "4340e28bed388d50800c469525b460a108091da0"
    assert "files" not in lock
    stamp.require(TRAINING_CONFIG)
    assert TRAINING_CONFIG["g_lr"] == 2e-5
    assert STAMP is stamp
    assert reference.RoutedMLP.__module__ == "particle_sliders.reference"
    assert reference.GlobalMixErrorCritic.__module__ == "particle_sliders.reference"
    hashed = runtime_source("vendor/reference.py")
    assert hashed.resolve() == Path(shared_reference.__file__).resolve()
    released = json.loads((ROOT / "configs/anima/released-identity.json").read_text())
    assert released["model"] == "circlestone-labs/Anima:turbo-v1.1"


def test_published_checkpoint_identity_remains_loadable():
    from lumen_studio.provenance import checkpoint_identity_accepted, released_identity
    released = released_identity()
    current = dict(released, runtime_sha256="particle-sliders-runtime")
    assert checkpoint_identity_accepted(released, current)
    assert checkpoint_identity_accepted(current, current)
    assert not checkpoint_identity_accepted(dict(released, model="other"), current)


def test_shared_fitting_is_exact_original_recipe():
    torch.manual_seed(817)
    torch.set_num_threads(1)
    for rows, width in ((24, 48), (64, 12)):
        x, y = torch.randn(rows, width), torch.randn(rows, 8)
        gram = x @ x.T
        ridge = 1e-2 * gram.diag().mean().clamp_min(1e-8)
        gram.diagonal().add_(ridge)
        expected = torch.linalg.solve(gram, y).T @ x
        actual, actual_ridge = fit_routed_down(x, y)
        assert torch.equal(actual, expected)
        assert torch.equal(actual_ridge, ridge)
