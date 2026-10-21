"""Archived gmix benchmark; no native dispatch under the PG09 proposal."""
from .training_policy import legacy_training_unavailable


def benchmark(root, variation, gpu="0", should_stop=None):
    legacy_training_unavailable("legacy benchmark")
