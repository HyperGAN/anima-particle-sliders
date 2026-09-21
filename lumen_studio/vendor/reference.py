"""Compatibility imports from the pinned shared algorithm package.

The extracted implementation is byte-identical to the released reference.
Runtime provenance hashes that implementation, not this import shim.
"""
from concept_slider_core.reference import (
    REFERENCE, RoutedMLP, GlobalMixErrorCritic, mlp, bound_score,
    register_paired_error_norm, particle_vic, noise_std, rp_d_loss, rp_g_loss,
)
