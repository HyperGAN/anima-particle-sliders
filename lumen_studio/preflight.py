"""Archived gmix resume proof; prior proofs do not qualify PG09 training."""
from .training_policy import legacy_training_unavailable


def verify_gpu_resume(root, configuration, gpu="0", should_stop=None, profile_update=False):
    legacy_training_unavailable("legacy GPU resume preflight")
