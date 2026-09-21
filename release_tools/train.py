"""Reproduce a released particle training run using the frozen recipe."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lumen_studio import execution
from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.cache import TargetCache,prepare_targets
from lumen_studio.dataset import compile_manifest
from lumen_studio.training import Trainer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variation',choices=['candlelit','moonlit'])
    p.add_argument('--root',type=Path,default=Path('artifacts/reproduction'))
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--prepare-targets',action='store_true')
    p.add_argument('--until',type=int,default=1600)
    args=p.parse_args()
    if not 1<=args.until<=1600:p.error('--until must be between 1 and 1600')
    runtime=TurboRuntime(args.model,'cuda:0',checkpointing=False)
    trainer=None
    try:
        if args.prepare_targets:
            for split in ('train','dev'):
                prepare_targets(runtime,compile_manifest(split),args.variation,
                    args.root/'targets'/args.variation/split,
                    progress=lambda done,total:print('targets',done,total,flush=True))
        train=TargetCache(args.root/'targets'/args.variation/'train',pin_memory=True)
        dev=TargetCache(args.root/'targets'/args.variation/'dev',pin_memory=True)
        trainer=Trainer(runtime,train,args.root/'runs'/args.variation,args.variation,
                        seed=7,microbatch=1,dev_cache=dev)
        while trainer.step<args.until:
            result=trainer.update()
            print('update',result['step'],result['g_adv'],flush=True)
            if trainer.step%100==0:trainer.probe()
        trainer.save(snapshot=True)
    finally:
        if trainer:trainer.close()
        runtime.close()

if __name__=='__main__':main()
