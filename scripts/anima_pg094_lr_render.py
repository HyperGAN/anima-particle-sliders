"""Render matched heldout controls from portable PG094 EMA exports only.

Use in a separate admitted GPU window after training has unloaded. This loader
constructs no critic, optimizer, sampler, training game, or training cache.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anima_pg094_lr_pilot import load_response_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--heldout-cache', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-memory-gib', type=float, required=True)
    parser.add_argument('--seed', type=int, default=29001)
    args = parser.parse_args()
    import torch
    from lumen_studio.cache import TargetCache
    from lumen_studio.contracts import atomic_json, file_hash
    from lumen_studio.particles import ParticleAdapter
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.max_memory_gib*2**30/
        torch.cuda.get_device_properties(0).total_memory)
    args.output.mkdir(parents=True, exist_ok=True)

    def guard():
        while True:
            result = subprocess.run(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],
                capture_output=True,text=True)
            if result.returncode == 0 and int(result.stdout.strip()) < 5120:
                print('Own renderer exiting: global free VRAM below5GiB',flush=True)
                os._exit(75)
            time.sleep(3)
    threading.Thread(target=guard, daemon=True).start()
    runtime = load_response_runtime(args.model_dir)
    cache = TargetCache(args.heldout_cache)
    if cache.index['identity']['split'] != 'dev' or cache.index['identity']['model'] != runtime.identity:
        raise ValueError('heldout cache/model identity differs')
    first = cache.index['shards'][0]
    data = torch.load(cache.directory/first['path'],map_location='cpu',weights_only=True)
    runtime.pipe.load_components(names=['vae'],dtype=runtime.dtype)
    runtime.pipe.vae.requires_grad_(False).eval().to(runtime.device)
    runtime.pipe.vae.enable_slicing()
    adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
    runtime.mixer.add('pilot-render',adapter)
    report = dict(schema='anima-pg094-matched-native-render-v1',
        source_sha256=file_hash(__file__),source_commit=os.environ.get('ANIMA_EXPERIMENT_SOURCE_COMMIT'),
        model=runtime.identity,heldout_index_sha256=file_hash(cache.directory/'index.json'),
        row=data['row'],seed=args.seed,width=512,height=512,steps=10,images=[],artifacts=[])

    @torch.no_grad()
    def render(name,embedding,strength):
        with runtime.mixer.scales({'pilot-render':strength}):
            latent,scheduler = runtime.noise(args.seed,512,512),runtime.scheduler(10)
            for timestep in scheduler.timesteps:
                velocity=runtime.predict(latent,timestep,embedding)
                latent=runtime.step(scheduler,velocity,timestep,latent)
            image=runtime.decode(latent)
        path=args.output/(name+'.png')
        if path.exists():
            raise FileExistsError('render outputs are immutable')
        image.save(path)
        report['images'].append(dict(name=name,sha256=file_hash(path),strength=strength))
        atomic_json(args.output/'report.json',report)
        print(json.dumps(report['images'][-1]),flush=True)
        return image

    try:
        images=[render('base-neutral',data['embedding'],0),
                render('positive-teacher',data['positive_embedding'],0)]
        for artifact in args.artifacts:
            metadata=adapter.load_export(artifact,model_identity=runtime.identity)
            report['artifacts'].append(dict(path=str(artifact),sha256=file_hash(artifact),metadata=metadata))
            for strength in (.5,1.):
                images.append(render(artifact.parent.name+'-strength-'+str(strength),data['embedding'],strength))
        from PIL import Image,ImageDraw
        labels=[item['name'] for item in report['images']]
        grid=Image.new('RGB',(512*len(images),544),'white')
        draw=ImageDraw.Draw(grid)
        for i,(image,label) in enumerate(zip(images,labels,strict=True)):
            grid.paste(image,(512*i,32))
            draw.text((512*i+8,8),label,fill='black')
        grid.save(args.output/'matched-grid.png')
        report.update(status='complete',grid_sha256=file_hash(args.output/'matched-grid.png'),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            qualification='matched native visualization; no common sampler/render capability promotion')
        atomic_json(args.output/'report.json',report)
    finally:
        runtime.close()


if __name__=='__main__':
    main()
