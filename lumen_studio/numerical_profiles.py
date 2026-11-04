"""Explicit native numerical contracts; activation precedes CUDA initialization."""
import hashlib
import os
from pathlib import Path
import sys


NAME = 'deterministic-math-cuda-v1'


def activate_numerical_profile(name):
    """Opt in for one isolated process, before any CUDA context is created.

    The default product never calls this. An experiment explicitly selects and
    records the combined profile; these switches are not separate causal claims.
    """
    if name != NAME:
        raise ValueError('unknown Anima numerical profile')
    loaded = sys.modules.get('torch')
    if loaded is not None and loaded.cuda.is_initialized():
        raise RuntimeError('activate Anima numerical profile before CUDA initialization')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(False)
    return require_numerical_profile(name)


def require_numerical_profile(name):
    """Verify actual flags without changing them or allocating a model."""
    if name != NAME:
        raise ValueError('unknown Anima numerical profile')
    import torch
    actual = dict(cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),
        cudnn_sdp=(torch.backends.cuda.cudnn_sdp_enabled()
                   if hasattr(torch.backends.cuda, 'cudnn_sdp_enabled') else False))
    expected = dict(cublas_workspace=':4096:8',deterministic=True,warn_only=False,
        tf32_matmul=False,tf32_cudnn=False,cudnn_benchmark=False,
        cudnn_deterministic=True,flash_sdp=False,memory_efficient_sdp=False,
        math_sdp=True,cudnn_sdp=False)
    if actual != expected:
        raise ValueError('Anima numerical profile differs from its actual process flags')
    return dict(schema='anima-native-numerical-profile-v1',name=NAME,flags=actual,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
