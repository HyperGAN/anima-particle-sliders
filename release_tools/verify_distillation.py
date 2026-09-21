"""CPU numerical checks for the exported ordinary-LoRA distillation primitive."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from release_tools.distill import fit

torch.manual_seed(71);torch.set_num_threads(1)
# Identifiable linear teacher: fitting should recover its output on new inputs.
down=torch.randn(8,12)*.1;up=torch.randn(16,8)*.1
x=torch.randn(128,12);v=torch.randn(64,12)
branch=nn.Module();branch.up=nn.Linear(8,16,bias=False);branch.up.weight.data.copy_(up)
class Fixture:pass
adapter=Fixture();adapter.branches=[branch];adapter.names=['linear-recovery']
with torch.no_grad():
 fitted,report=fit(adapter,[(x,x@down.T)],[(v,v@down.T)])
assert report['heldout_projection_relative_mse']<.001,report
result=dict(passed=True,fixture='held-out recovery of a known rank-8 linear teacher',
            relative_mse=report['heldout_projection_relative_mse'],ridge_fraction=.01)
(ROOT/'validation/distillation-primitive.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
