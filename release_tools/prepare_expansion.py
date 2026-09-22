"""Calibrate new releases and fit ordinary LoRAs using development-only audits."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from lumen_studio.alpha_adapter import AlphaParticleAdapter, export_with_alpha, read_metadata
from single_particle import adapter_class, ORIGINAL_FORMAT as SINGLE_FORMAT, export_with_alpha as single_export
from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.cache import TargetCache
from lumen_studio.contracts import atomic_json, file_hash
from lumen_studio.coordinator import GpuLease
from release_tools.audit_distill import edit_metrics
from release_tools.distill import collect, fit, save_loras, attach_lora
from release_tools.rescale_lora_alpha import rescale_alpha


def render(runtime, request, dest, **provenance):
    metadata = dict(request, split='dev', cfg=1, model_identity=runtime.identity,
                    postprocessing='none', **provenance)
    if dest.exists():
        old = json.loads(dest.with_suffix('.json').read_text())
        assert all(old[k] == v for k,v in metadata.items())
        assert old['image_sha256'] == file_hash(dest)
        return
    image = runtime.render(request['prompt'], request['seed'], request['width'], request['height'], request['steps'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    atomic_json(dest.with_suffix('.json'), dict(metadata, image_sha256=file_hash(dest)))
    print('rendered', dest, flush=True)


def load_particle(runtime, path):
    adapter = adapter_class(path)(runtime.transformer).to(runtime.device).eval().requires_grad_(False)
    adapter.load_export(path, model_identity=runtime.identity)
    runtime.mixer.add('particle', adapter)
    return adapter


@torch.no_grad()
def particle_sweep(runtime, studio, output, entry):
    folder = output/entry['id']/'particle-sweep'
    for request in entry['render_requests']:
        render(runtime, request, folder/f"{request['case']}-off.png", format='off', strength=0, alpha=None)
    for step in entry['checkpoint_candidates']:
        source = studio/entry['run_directory']/f'ema-{step:06}.safetensors'
        adapter = load_particle(runtime, source)
        try:
            for alpha in entry['particle_alpha_candidates']:
                adapter.alphas = [alpha]*len(adapter.branches)
                with runtime.mixer.scales({'particle':1.}):
                    for request in entry['render_requests']:
                        render(runtime, request, folder/f"step{step}-alpha{alpha:g}-{request['case']}.png",
                            format='particles', strength=1, alpha=alpha, rank=8, checkpoint_step=step,
                            source_adapter_sha256=file_hash(source))
        finally:
            runtime.mixer.close()


@torch.no_grad()
def fit_lora(runtime, studio, output, entry, selection):
    folder = output/entry['id']
    source = studio/entry['run_directory']/f"ema-{selection['step']:06}.safetensors"
    teacher = folder/'particle.safetensors'
    if not teacher.exists():
        exporter=single_export if read_metadata(source)['format']==SINGLE_FORMAT else export_with_alpha
        exporter(source, teacher, alpha=selection['particle_alpha'], provenance=selection)
    metadata = read_metadata(teacher)
    assert metadata['source_checkpoint_sha256']==file_hash(source)
    assert metadata['network_alpha']==selection['particle_alpha']
    adapter = load_particle(runtime, teacher)
    report_path = folder/'distilled/evaluation.json'
    try:
        if report_path.exists():
            assert json.loads(report_path.read_text())['teacher_sha256']==file_hash(teacher)
            return
        train_cache = TargetCache(studio/'targets'/entry['training_id']/'train')
        dev_cache = TargetCache(studio/'targets'/entry['training_id']/'dev')
        assert train_cache.index['identity']['split']=='train' and dev_cache.index['identity']['split']=='dev'
        for split,cache in [('train',train_cache),('dev',dev_cache)]:
            assert cache.index['identity']['manifest_sha256']==entry['manifests'][split]
        assert metadata['manifest_sha256']==entry['manifests']['train']
        with runtime.mixer.scales({'particle':1.}):
            train = collect(runtime,adapter,train_cache,training=True)
            dev = collect(runtime,adapter,dev_cache,training=False)
        downs, report = fit(adapter,train,dev)
        save_loras(adapter,downs,folder/'distilled',entry['id'],teacher)
        report.update(variation=entry['training_id'], export_name=entry['id'], teacher_sha256=file_hash(teacher),
            method='rank8-ridge-full-input', ridge_fraction=.01, final_test_used=False,
            train_cache_sha256=train_cache.index['fingerprint'],dev_cache_sha256=dev_cache.index['fingerprint'])
        atomic_json(report_path,report)
        print('fit complete',entry['id'],report['heldout_projection_relative_mse'],flush=True)
    finally:
        runtime.mixer.close()


@torch.no_grad()
def lora_sweep(runtime, studio, output, entry, selection):
    folder = output/entry['id']
    teacher = folder/'particle.safetensors'
    lora = folder/'distilled/native'/f"{entry['id']}.safetensors"
    cache = TargetCache(studio/'targets'/entry['training_id']/'dev')
    seen=set(); indices=[]
    for i,shard in enumerate(cache.index['shards']):
        record=cache[i*20]
        character=record['row']['character']
        if character in seen:continue
        seen.add(character)
        indices.extend(i*20+t for t in range(10))
    assert len(seen)==4 and cache.index['identity']['split']=='dev'
    records=[cache[i] for i in indices]
    def predict():
        return [runtime.predict(r['latent'].to(runtime.device),r['timestep'],r['embedding']).cpu() for r in records]
    base=predict()
    adapter=load_particle(runtime,teacher)
    try:
        with runtime.mixer.scales({'particle':1.}):target=predict()
    finally:
        runtime.mixer.close()
    report=dict(teacher_sha256=file_hash(teacher),lora_sha256=file_hash(lora),cache_indices=indices,
        cache_sha256=cache.index['fingerprint'],split='dev',final_test_used=False,measurements=[])
    for gain in entry['lora_gain_candidates']:
        handles=attach_lora(runtime.transformer,lora,gain)
        try:
            metrics=edit_metrics(predict(),target,base)
            report['measurements'].append(dict(gain=gain,alpha=gain*selection['particle_alpha'],**metrics))
            for request in entry['render_requests']:
                render(runtime,request,folder/'lora-sweep'/f"gain{gain:g}-{request['case']}.png",
                    format='lora',strength=1,alpha=gain*selection['particle_alpha'],rank=8,
                    source_adapter_sha256=file_hash(lora),source_strength=gain)
            print('lora match',entry['id'],gain,metrics,flush=True)
            atomic_json(folder/'lora-sweep/audit.json',report)
        finally:
            for h in handles:h.remove()


@torch.no_grad()
def final_samples(runtime, studio, output, entry, selection):
    folder=output/entry['id'];teacher=folder/'particle.safetensors'
    for fmt in ('native','comfyui'):
        source=folder/'distilled'/fmt/f"{entry['id']}.safetensors"
        dest=folder/'final'/fmt/f"{entry['id']}-unit-alpha.safetensors"
        if not dest.exists():rescale_alpha(source,dest,selection['lora_alpha']/selection['particle_alpha'])
    adapter=load_particle(runtime,teacher)
    try:
        for fmt,strength in [('particles',1),('lora',1),('off',0)]:
            path=teacher if fmt=='particles' else folder/'final/native'/f"{entry['id']}-unit-alpha.safetensors" if fmt=='lora' else None
            handles=attach_lora(runtime.transformer,path,1.) if fmt=='lora' else []
            try:
                with runtime.mixer.scales({'particle':1.} if fmt=='particles' else {}):
                    for request in entry['render_requests']:
                        render(runtime,request,folder/'final/samples'/request['case']/f'{fmt}.png',
                            format=fmt,strength=strength,energy=strength,
                            alpha=selection['particle_alpha'] if fmt=='particles' else selection['lora_alpha'] if fmt=='lora' else None,
                            rank=8 if path else None,adapter_sha256=file_hash(path) if path else None)
            finally:
                for h in handles:h.remove()
    finally:
        runtime.mixer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['particles','fit','loras','final'])
    p.add_argument('--studio',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path,default=ROOT/'data/expansion-plan.json')
    p.add_argument('--selection',type=Path,default=ROOT/'data/expansion-selection.json')
    args=p.parse_args()
    plan=json.loads(args.plan.read_text())
    selections=json.loads(args.selection.read_text()) if args.selection.exists() else {}
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35)
    runtime=TurboRuntime(args.studio/'model','cuda:0')
    try:
        for entry in plan['sliders']:
            if args.action=='particles':particle_sweep(runtime,args.studio,args.output,entry)
            else:
                fn={'fit':fit_lora,'loras':lora_sweep,'final':final_samples}[args.action]
                fn(runtime,args.studio,args.output,entry,selections[entry['id']])
    finally:
        runtime.close()


if __name__=='__main__':
    with GpuLease(Path('/tmp'),os.environ['ANIMA_GPU_LEASE']):main()
