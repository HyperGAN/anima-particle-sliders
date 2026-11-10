"""Anima explicit fresh95-v2 callable API. Old lumen_studio routes stay unchanged."""
PROFILE = "anima-explicit-fresh95-v2"

def open_training(*args, **kwargs):
    from .training import FreshTrainer
    return FreshTrainer(*args, **kwargs)

def open_inference(*args, **kwargs):
    from .inference import InferenceOwner
    return InferenceOwner(*args, **kwargs)
