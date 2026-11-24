"""Select exterior training support from the frozen Anima cue response."""

import torch
from torch.nn.functional import max_pool2d


POSITIONS = (3, 4, 5, 6)
CUTOFF = 0.01
REFERENCE_RADIUS = 2
MAX_RADIUS = 4


def dilate(matte, radius):
    return max_pool2d(matte.squeeze(2), 2 * radius + 1, stride=1, padding=radius).unsqueeze(2)


def response_halo(matte, teacher_deltas):
    """Redistribute exterior support while retaining the fixed halo's area and opacities."""
    fixed = dilate(matte, REFERENCE_RADIUS)
    outside = matte <= CUTOFF
    exterior_weights = fixed[outside & (fixed > CUTOFF)].sort(descending=True).values
    allowed = (outside & (dilate(matte, MAX_RADIUS) > CUTOFF)).flatten()
    # One response magnitude per latent site and teacher field.
    response = teacher_deltas.double().square().sum(dim=2, keepdim=True).sqrt()
    response = response / response.mean(dim=(-2, -1), keepdim=True)
    evidence = response.clamp(max=3).mean(dim=0).flatten()
    order = evidence.masked_fill(~allowed, -torch.inf).argsort(descending=True, stable=True)
    selected = order[:exterior_weights.numel()]
    if not (allowed[selected].all() and torch.isfinite(evidence).all()):
        raise ValueError("The teacher response cannot support the area-matched halo")
    inferred = fixed.masked_fill(outside, 0)
    inferred.view(-1)[selected] = exterior_weights
    return fixed, inferred
