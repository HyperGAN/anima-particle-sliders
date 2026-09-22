"""Portable alpha/rank scaling for native particle adapter inference exports.

The pinned v1 training format stays unchanged. V2 stores one scalar alpha per
branch, and this loader applies it inside the branch, before ordinary mixing.
These remain routed particle adapters, not conventional two-matrix LoRAs.
"""
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .contracts import canonical, file_hash
from .particles import ARCHITECTURE, FORMAT, ParticleAdapter

ALPHA_FORMAT = "anima-turbo-routed-particles-alpha-v2"
SCALING = "alpha/rank"


def read_metadata(path):
    with safe_open(path, framework="pt", device="cpu") as handle:
        return json.loads(handle.metadata()["anima"])


def alpha_values(metadata, state):
    if metadata.get("format") != ALPHA_FORMAT or metadata.get("scaling") != SCALING:
        raise ValueError("Unsupported adapter alpha format")
    if metadata.get("architecture") != ARCHITECTURE:
        raise ValueError("Incompatible particle architecture")
    names = metadata["targets"]
    expected = {f"branches.{i}.alpha" for i in range(len(names))}
    if {k for k in state if k.endswith(".alpha")} != expected:
        raise ValueError("Missing or unexpected branch alpha")
    values = []
    for i in range(len(names)):
        value = state[f"branches.{i}.alpha"]
        if value.ndim != 0 or not value.is_floating_point():
            raise ValueError("Branch alpha must be a floating scalar")
        alpha = float(value)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Branch alpha must be finite and positive")
        values.append(alpha)
    if any(a != metadata.get("network_alpha") for a in values):
        raise ValueError("Branch alpha differs from the export metadata")
    return values


def export_with_alpha(source, destination, *, alpha, provenance=None):
    """Create a new immutable file. Every learned tensor remains bit-identical."""
    if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Alpha must be finite and positive")
    metadata = read_metadata(source)
    if metadata.get("format") != FORMAT or metadata.get("architecture") != ARCHITECTURE:
        raise ValueError("Alpha export requires an original v1 particle checkpoint")
    if metadata.get("weights") != "ema":
        raise ValueError("Alpha exports require immutable EMA weights")
    state = load_file(str(source))
    if any(k.endswith(".alpha") for k in state):
        raise ValueError("Source already contains alpha")
    for i in range(len(metadata["targets"])):
        # Rank is a power of two. FP64 preserves the calibrated Python scalar
        # exactly through alpha = gain * rank, then gain = alpha / rank.
        state[f"branches.{i}.alpha"] = torch.tensor(alpha, dtype=torch.float64)
    metadata.update(format=ALPHA_FORMAT, scaling=SCALING, network_alpha=float(alpha),
                    source_checkpoint_sha256=file_hash(source),
                    alpha_provenance=provenance or {})
    alpha_values(metadata, state)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Exports are immutable; use a new checkpoint path")
    temporary = destination.with_suffix(".tmp")
    save_file(state, str(temporary), metadata={"anima": canonical(metadata)})
    temporary.replace(destination)
    return file_hash(destination)


class AlphaParticleAdapter(ParticleAdapter):
    """V1 and alpha-v2 reader, usable with the original Mixer outside Studio."""

    def __init__(self, transformer):
        super().__init__(transformer)
        self.alphas = [float(ARCHITECTURE["rank"])] * len(self.branches)
        for index, branch in enumerate(self.branches):
            def apply_alpha(module, inputs, output, index=index):
                scale = self.alphas[index] / ARCHITECTURE["rank"]
                return output if scale == 1. else output * scale
            branch.register_forward_hook(apply_alpha)

    def load_export(self, path, *, model_identity):
        metadata = read_metadata(path)
        if metadata["format"] == FORMAT:
            result = super().load_export(path, model_identity=model_identity)
            self.alphas = [float(ARCHITECTURE["rank"])] * len(self.branches)
            return result
        if metadata.get("targets") != self.names or metadata.get("model_identity") != model_identity:
            raise ValueError("Alpha checkpoint targets or model runtime differ")
        state = load_file(str(path))
        alphas = alpha_values(metadata, state)
        weights = {k: v for k, v in state.items() if not k.endswith(".alpha")}
        self.load_state_dict(weights, strict=True)
        self.alphas = alphas
        return metadata

    def export(self, *args, **kwargs):
        # Inference-only: never accidentally serialize alpha-loaded weights as v1.
        raise ValueError("Use export_with_alpha with the original training checkpoint")


def checkpoint_alphas(checkpoints, catalog):
    """Image provenance only. Actual scaling is read from tensors by the loader."""
    result = {}
    for name, sha in checkpoints.items():
        metadata = catalog[sha]["metadata"]
        if metadata.get("format") == ALPHA_FORMAT:
            result[name] = dict(checkpoint_sha256=sha, alpha=metadata["network_alpha"],
                                rank=metadata["architecture"]["rank"], scaling=SCALING)
    return result
