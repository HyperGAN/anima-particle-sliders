"""Compare nearby particle/LoRA alphas at nominal strength one on development cases."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from lumen_studio.alpha_adapter import AlphaParticleAdapter
from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.cache import TargetCache
from lumen_studio.contracts import atomic_json, file_hash
from lumen_studio.coordinator import GpuLease
from release_tools.audit_distill import edit_metrics
from release_tools.distill import attach_lora


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new audit directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    particle = args.folder/'weights/bad-intent.safetensors'
    lora = args.folder/'distilled/native/bad-intent.safetensors'
    requests = [s for s in json.loads((ROOT/'data/bad-intent-samples.json').read_text()) if s['strength'] == 1]
    cache = TargetCache(args.root/'targets/uncanny/dev')
    assert cache.index['identity']['split'] == 'dev'
    indices = [shard*20+t for shard in (0,2,4,6) for t in range(10)]
    records = [cache[i] for i in indices]
    report = dict(split='dev', final_test_used=False, cache_sha256=cache.index['fingerprint'], cache_indices=indices,
                  particle_sha256=file_hash(particle), lora_sha256=file_hash(lora), nominal_strength=1,
                  measurements=[], samples=[], pairs=[])
    runtime = TurboRuntime(args.root/'model', 'cuda:0')
    assert runtime.identity == cache.index['identity']['model']
    report['model_identity'] = runtime.identity
    started = time.monotonic()

    def save():
        report['seconds'] = time.monotonic()-started
        atomic_json(args.output/'audit.json', report)

    def predict():
        return [runtime.predict(r['latent'].to(runtime.device),r['timestep'],r['embedding']).cpu() for r in records]

    def render_samples(fmt, alpha, source):
        for sample in requests:
            payload = sample['payload']
            name = f"{fmt}-alpha{alpha:g}-case{sample['case']}"
            image = runtime.render(payload['prompt'],payload['seed'],payload['width'],payload['height'],payload['steps'])
            image.save(args.output/(name+'.png'))
            metadata = dict(payload,case=sample['case'],format=fmt,alpha=alpha,rank=8,nominal_strength=1,
                source_strength=alpha/8,source_adapter_sha256=file_hash(source),split='dev',model_identity=runtime.identity)
            atomic_json(args.output/(name+'.json'),metadata)
            report['samples'].append(dict(path=name+'.png',**metadata))
            print('rendered',name,flush=True)
            save()

    try:
        base = predict()
        report['baseline_max_abs'] = max(float((p-r['neutral']).abs().max()) for p,r in zip(base,records))
        adapter = AlphaParticleAdapter(runtime.transformer).to(runtime.device).eval().requires_grad_(False)
        adapter.load_export(particle,model_identity=runtime.identity)
        runtime.mixer.add('particle',adapter)
        particle_predictions = {}
        for alpha in (8.,10.,12.,16.):
            # Apply alpha inside the particle branch, using nominal strength one.
            adapter.alphas = [alpha]*len(adapter.branches)
            with runtime.mixer.scales({'particle':1.}):
                pred = predict()
                particle_predictions[alpha] = pred
                metrics = edit_metrics(pred,particle_predictions[8.],base)
                report['measurements'].append(dict(format='particles',alpha=alpha,**metrics))
                render_samples('particles',alpha,particle)
        runtime.mixer.close()
        del adapter
        for alpha in (20.,22.,24.):
            handles = attach_lora(runtime.transformer,lora,alpha/8.)
            try:
                pred = predict()
                metrics = edit_metrics(pred,particle_predictions[8.],base)
                report['measurements'].append(dict(format='lora',alpha=alpha,**metrics))
                for pa, pp in particle_predictions.items():
                    report['pairs'].append(dict(particle_alpha=pa,lora_alpha=alpha,**edit_metrics(pred,pp,base)))
                render_samples('lora',alpha,lora)
            finally:
                for h in handles:h.remove()
        save()
        print(json.dumps(report['pairs']),flush=True)
    finally:
        runtime.close()


if __name__ == '__main__':
    with GpuLease(Path('/tmp'),os.environ['ANIMA_GPU_LEASE']):
        main()
