"""Restrict the stronger Final Boss cue's coarse exterior contrast."""

from functools import cache

import torch


# Daubechies 4 reconstruction low pass in the periodized sampling convention.
LOW_PASS = (0.2303778133088965, 0.7148465705529157, 0.6308807679298589, -0.027983769416859854,
            -0.18703481171909309, 0.030841381835560764, 0.0328830116668852, -0.010597401785069032)


@cache
def low_pass_basis(size, levels, device):
    if size < 8 or size % 2 or levels < 1:
        raise ValueError("The native latent shape does not support this wavelet level")
    rows = torch.arange(size // 2, device=device)[:, None]
    taps = torch.arange(len(LOW_PASS), device=device)
    columns = (2 * rows + taps - 3) % size
    values = torch.tensor(LOW_PASS, dtype=torch.float64, device=device)
    basis = torch.zeros((size // 2, size), dtype=torch.float64, device=device)
    basis.scatter_(1, columns, values.expand_as(columns))
    if levels == 1:
        return basis
    return low_pass_basis(size // 2, levels - 1, device) @ basis


def low_band(field, level):
    height, width = field.shape[-2:]
    y = low_pass_basis(height, level, field.device)
    x = low_pass_basis(width, level, field.device)
    return y.T @ (y @ field @ x.T) @ x


def coarse_exterior_target(neutral, baseline, stronger, exterior):
    strong_edit = stronger.double() - neutral.double()
    added = stronger.double() - baseline.double()
    exterior_edit = torch.where(exterior, strong_edit, 0.)
    original_norm = exterior_edit.norm()
    if not original_norm > 0:
        return stronger
    exterior_cue = torch.where(exterior, added, 0.)
    coarse = low_band(exterior_cue, 3)
    revised = torch.where(exterior, exterior_edit - coarse, 0.)
    norm = revised.norm()
    if not norm > 0:
        raise ValueError("The exterior waveform has no remaining edit")
    target = torch.where(exterior, (neutral.double() + revised * (original_norm / norm)).to(stronger.dtype),
                         stronger)
    full_ratio = (target.double() - neutral.double()).norm() / strong_edit.norm()
    if not torch.isfinite(target).all() or abs(full_ratio - 1) > 1e-6:
        raise ValueError("Removing the coarse exterior cue changed the edit dose")
    return target
