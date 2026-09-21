"""The extraction must preserve released identities and the fitting result."""
import json
from pathlib import Path

import torch

from concept_slider_core import fit_routed_down
from lumen_studio.contracts import digest, file_hash
from lumen_studio.provenance import runtime_source
from lumen_studio.vendor import reference

ROOT = Path(__file__).resolve().parents[1]


def test_actual_core_matches_released_runtime_identity():
    names = ("runtime.py", "backends/anima.py", "contracts.py", "particles.py", "vendor/reference.py")
    expected = json.loads((ROOT / "configs/anima/released-identity.json").read_text())
    assert digest({name: file_hash(runtime_source(name)) for name in names}) == expected["runtime_sha256"]
    assert reference.RoutedMLP.__module__ == "concept_slider_core.reference"
    assert reference.GlobalMixErrorCritic.__module__ == "concept_slider_core.reference"


def test_training_engine_is_byte_identical():
    names = ("training.py", "game.py", "cache.py", "metrics.py", "execution.py", "vendor/grad_regularizers.py")
    assert digest({name: file_hash(ROOT / "lumen_studio" / name) for name in names}) == \
        "53f20872a7fa68cc1db0ab133bb77af5f80f286ad351bce5c04d2d2520ac8f0f"


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
