"""Move subject edit strength from middle to late Anima teacher positions."""

from math import isfinite, sqrt

import torch


MIDDLE = range(3, 7)
LATE = range(7, 10)
LATE_ENERGY_SHARE = 0.01


def trajectory_scales(energies):
    if len(energies) != 10 or any(not isfinite(value) or value <= 0 for value in energies):
        raise ValueError("Expected ten finite, positive subject edit energies")
    middle = sum(energies[position] for position in MIDDLE)
    late = sum(energies[position] for position in LATE)
    transfer = LATE_ENERGY_SHARE * sum(energies)
    if transfer >= middle:
        raise ValueError("There is insufficient middle-position energy for the transfer")
    return {"middle": sqrt(1 - transfer / middle), "late": sqrt(1 + transfer / late)}


def redistribute_subject_target(neutral, target, matte, position, scales):
    if position in MIDDLE:
        scale = scales["middle"]
    elif position in LATE:
        scale = scales["late"]
    else:
        return target
    subject = (matte > 0.01).expand_as(target)
    anchor = neutral[subject].double()
    delta = target[subject].double() - anchor
    replacement = (anchor + delta * scale).to(target.dtype)
    return torch.masked_scatter(target, subject, replacement)
