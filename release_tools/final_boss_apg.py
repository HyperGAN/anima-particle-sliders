"""Apply the APG projection to the subject's frozen teacher contrast."""

import torch


def project_subject_target(latent, sigma, neutral, conditional, target, matte):
    support = matte > 0.01
    clean = latent.double() - sigma * conditional.double()
    axis = torch.where(support, clean, 0.)
    axis_norm = axis.norm()
    if not torch.isfinite(axis_norm) or axis_norm.item() <= 1e-10:
        raise ValueError("Degenerate conditional clean-latent axis")
    axis = axis / axis_norm

    original = target.double() - neutral.double()
    local = torch.where(support, original, 0.)
    local_norm = local.norm()
    parallel = (local * axis).sum()
    orthogonal = local - parallel * axis
    orthogonal_norm = orthogonal.norm()
    if not torch.isfinite(orthogonal_norm) or orthogonal_norm.item() <= 1e-10:
        raise ValueError("Degenerate teacher contrast after the subject projection")

    rescaled = orthogonal * (local_norm / orthogonal_norm)
    projected = torch.where(support, (neutral.double() + rescaled).to(target.dtype), target)
    actual = projected.double() - neutral.double()
    local_actual = torch.where(support, actual, 0.)
    if (not torch.isfinite(projected).all() or abs(local_actual.norm() / local_norm - 1.) > 1e-6
            or abs((local_actual * axis).sum()) / local_norm > 1e-6):
        raise ValueError("The subject projection changed the edit norm or retained a parallel component")
    return projected
