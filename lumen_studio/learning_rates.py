"""Explicit PG095 learning rates shared by the API and configured sessions."""
import math


def recipe_learning_rates(*, lr=None, d_lr_mult=None, prior_lr_mult=None):
    """Omitted values retain the authoritative ParticleGAN recipe defaults.

    Return only explicit overrides so old configurations keep their exact law.
    The shared game records the resolved recipe and rejects LR changes on resume.
    """
    values = dict(lr=lr, d_lr_mult=d_lr_mult, prior_lr_mult=prior_lr_mult)
    for name, value in values.items():
        if value is not None and (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(name + ' must be finite and positive')
    return {name: float(value) for name, value in values.items() if value is not None}
