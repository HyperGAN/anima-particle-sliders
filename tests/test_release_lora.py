"""The released LoRA hook must preserve the explicit two-matrix operation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from safetensors.torch import save_file
from release_tools.distill import attach_lora


def test_lora_delta_and_zero(tmp_path):
    torch.manual_seed(71)
    model=nn.Module();model.projection=nn.Linear(12,16,bias=False)
    down=torch.randn(8,12);up=torch.randn(16,8)
    file=tmp_path/'lora.safetensors'
    save_file({'projection.lora_A.weight':down,'projection.lora_B.weight':up},str(file))
    x=torch.randn(2,3,12);base=model.projection(x)
    for strength in (0,1,3,5):
        handles=attach_lora(model,file,strength)
        actual=model.projection(x)
        expected=base+strength*((x@down.T)@up.T)
        assert torch.equal(actual,expected)
        for h in handles:h.remove()
        assert torch.equal(model.projection(x),base)
