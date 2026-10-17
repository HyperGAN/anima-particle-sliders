"""Validate a complete checkpoint tensor schema without changing live modules."""
from collections.abc import Mapping

import torch


def validated_tensor_state(state, expected, *, scalar_keys=()):
    """Return module weights after checking all keys, shapes, dtypes and values.

    Extra serialized alpha scalars use FP64, as written by the native exporters.
    They are validated with the weights, then omitted from the returned module
    state. Device differences are allowed; implicit dtype conversions are not.
    """
    if not isinstance(state, Mapping):
        raise ValueError('Checkpoint tensor state must be a mapping')
    schema = {name: (tuple(value.shape), value.dtype) for name, value in expected.items()}
    for name in scalar_keys:
        if name in schema:
            raise ValueError('Checkpoint scalar collides with a module tensor')
        schema[name] = ((), torch.float64)
    if set(state) != set(schema):
        missing, unexpected = sorted(set(schema) - set(state)), sorted(set(state) - set(schema))
        raise ValueError(f'Checkpoint tensor keys differ: missing={missing}, unexpected={unexpected}')
    for name, (shape, dtype) in schema.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f'Checkpoint tensor shape or dtype differs: {name}')
        if not torch.isfinite(value).all():
            raise ValueError(f'Checkpoint tensor must be finite: {name}')
    return {name: state[name] for name in expected}
