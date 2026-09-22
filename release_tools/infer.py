"""Render a fixed-prompt particle or ordinary-LoRA sample."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lumen_studio import execution
from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.alpha_adapter import AlphaParticleAdapter
from single_particle import adapter_class
from lumen_studio.contracts import file_hash
from release_tools.distill import attach_lora


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--format',choices=['particles','lora'],default='particles')
    parser.add_argument('--prompt',required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--strength',type=float,default=1.)
    parser.add_argument('--width',type=int,default=768)
    parser.add_argument('--height',type=int,default=768)
    parser.add_argument('--steps',type=int,choices=[8,10,12],default=10)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 0<=args.strength<=5: parser.error('Strength must be between 0 and 5')
    runtime=TurboRuntime(args.model,'cuda:0')
    handles=[]
    try:
        if args.format=='particles':
            adapter=adapter_class(args.adapter)(runtime.transformer).to(runtime.device)
            adapter.load_export(args.adapter,model_identity=runtime.identity)
            adapter.requires_grad_(False).eval()
            runtime.mixer.add('release',adapter)
            runtime.mixer.strengths={'release':args.strength}
        else:
            handles=attach_lora(runtime.transformer,args.adapter,args.strength)
        image=runtime.render(args.prompt,args.seed,args.width,args.height,args.steps)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        image.save(args.output)
        record={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        record.update(adapter_sha256=file_hash(args.adapter),model_identity=runtime.identity,cfg=1.)
        args.output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    finally:
        for h in handles:h.remove()
        runtime.close()

if __name__=='__main__':main()
