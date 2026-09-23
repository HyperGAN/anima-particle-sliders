"""Compatibility imports from the pinned particle-sliders core.

Runtime provenance hashes ``particle_sliders.reference``, not this shim.
The shared package owns RoutedMLP, the global-mix critic, and the losses.
"""
from particle_sliders.reference import (
    REFERENCE, RoutedMLP, GlobalMixErrorCritic, mlp, bound_score,
    register_paired_error_norm, particle_vic, noise_std, rp_d_loss, rp_g_loss,
)
