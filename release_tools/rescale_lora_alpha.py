"""Create an immutable ordinary-LoRA derivative by scaling only stored alpha."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def rescale_alpha(source, destination, gain):
    source, destination = Path(source), Path(destination)
    if isinstance(gain, bool) or not math.isfinite(gain) or gain <= 0:
        raise ValueError('Gain must be finite and positive')
    if destination.exists():
        raise FileExistsError('Use a new derived filename; source exports are immutable')
    state = load_file(str(source))
    expected = set()
    for key, value in list(state.items()):
        for down, up in (('.lora_A.weight', '.lora_B.weight'),
                         ('.lora_down.weight', '.lora_up.weight')):
            if key.endswith(down):
                prefix = key.removesuffix(down)
                if value.ndim != 2 or value.shape[0] < 1:
                    raise ValueError('Invalid down matrix')
                other = state[prefix+up]
                if other.ndim != 2 or other.shape[1] != value.shape[0]:
                    raise ValueError('Inconsistent LoRA rank')
                expected.add(prefix+'.alpha')
                # Older native exports omit alpha, meaning alpha == rank.
                state.setdefault(prefix+'.alpha', torch.tensor(float(value.shape[0]), dtype=torch.float64))
    if not expected or {k for k in state if k.endswith('.alpha')} != expected:
        raise ValueError('Expected ordinary LoRA matrices with no unmatched alpha keys')
    for key in expected:
        value = state[key]
        if value.ndim != 0 or not value.is_floating_point() or not 0 < float(value) < math.inf:
            raise ValueError('Alpha must be a finite positive floating scalar')
        value = float(value)*gain
        if not math.isfinite(value):
            raise ValueError('Scaled alpha must be finite')
        state[key] = torch.tensor(value, dtype=torch.float64)
    with safe_open(source, framework='pt', device='cpu') as handle:
        metadata = dict(handle.metadata() or {})
    metadata.update(alpha_parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    alpha_gain=repr(float(gain)), alpha_scaling='alpha/rank',
                    alpha_derivation='Only alpha tensors changed; learned matrices preserved')
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(destination), metadata=metadata)
    return dict(path=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                parent_sha256=metadata['alpha_parent_sha256'], gain=gain,
                projections=len(expected), alphas=sorted({float(state[k]) for k in expected}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    p.add_argument('--gain', type=float, required=True)
    args = p.parse_args()
    print(json.dumps(rescale_alpha(args.source, args.destination, args.gain), indent=2))
