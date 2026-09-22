"""Compare complete teacher/student edits and render a development alpha sweep.

Run with the pinned anima_cuda_host.py launcher and ANIMA_GPU_LEASE set to the
physical GPU UUID. Strength g here is equivalent to multiplying every stored
LoRA alpha by g and sampling that derived file at nominal strength one.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import load_file
from lumen_studio.alpha_adapter import AlphaParticleAdapter
from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.cache import TargetCache
from lumen_studio.contracts import atomic_json, file_hash
from lumen_studio.coordinator import GpuLease
from release_tools.distill import attach_lora


def edit_metrics(predictions, teacher, baselines):
    """Measure the entire denoiser's edit, excluding its shared base velocity."""
    student = torch.cat([(p-b).float().flatten() for p, b in zip(predictions, baselines)])
    target = torch.cat([(p-b).float().flatten() for p, b in zip(teacher, baselines)])
    ss, tt, st = student.square().sum(), target.square().sum(), (student*target).sum()
    return dict(relative_edit_mse=float((student-target).square().sum()/tt),
                edit_cosine=float(st/(ss*tt).sqrt()),
                edit_rms_ratio=float((ss/tt).sqrt()),
                aligned_edit_gain=float(st/tt))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--student', type=Path, required=True)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--gains', type=float, nargs='+', default=[1, 2, 3, 5, 8])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new audit output directory')
    if any(not 0 < g < float('inf') for g in args.gains):
        raise ValueError('Gains must be finite and positive')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    started = time.monotonic()
    cache = TargetCache(args.root/'targets/uncanny/dev')
    assert cache.index['identity']['split'] == 'dev'
    indices = [shard*20+t for shard in (0, 2, 4, 6) for t in range(10)]
    records = [cache[i] for i in indices]
    assert len({r['row']['character'] for r in records}) == 4
    report = dict(teacher_sha256=file_hash(args.teacher), student_sha256=file_hash(args.student),
                  cache_sha256=cache.index['fingerprint'], cache_indices=indices,
                  split='dev', final_test_used=False, measurements=[], samples=[])
    state = load_file(str(args.student))
    report['stored_alphas'] = sorted({float(v) for k, v in state.items() if k.endswith('.alpha')})
    del state
    runtime = TurboRuntime(args.root/'model', 'cuda:0')
    report['model_identity'] = runtime.identity
    assert runtime.identity == cache.index['identity']['model']

    def save():
        report['seconds'] = time.monotonic()-started
        atomic_json(args.output/'audit.json', report)

    def predict():
        return [runtime.predict(r['latent'].to(runtime.device), r['timestep'], r['embedding']).cpu()
                for r in records]

    try:
        baselines = predict()
        report['cached_baseline_max_abs'] = max(float((p-r['neutral']).abs().max())
                                                for p, r in zip(baselines, records))
        adapter = AlphaParticleAdapter(runtime.transformer).to(runtime.device).eval().requires_grad_(False)
        adapter.load_export(args.teacher, model_identity=runtime.identity)
        runtime.mixer.add('teacher', adapter)
        with runtime.mixer.scales({'teacher': 1.}):
            teacher = predict()
        runtime.mixer.close()
        del adapter
        print('Measured base and teacher on 40 shared development states', flush=True)
        for gain in args.gains:
            handles = attach_lora(runtime.transformer, args.student, gain)
            try:
                result = dict(gain=gain, effective_alphas=[gain*a for a in report['stored_alphas']],
                              **edit_metrics(predict(), teacher, baselines))
            finally:
                for h in handles:
                    h.remove()
            report['measurements'].append(result)
            save()
            print(json.dumps(result), flush=True)
        samples = [s for s in json.loads(args.samples.read_text()) if s['strength'] == 1]
        for sample in samples:
            payload = sample['payload']
            for gain in args.gains:
                handles = attach_lora(runtime.transformer, args.student, gain)
                try:
                    image = runtime.render(payload['prompt'], payload['seed'], payload['width'],
                                           payload['height'], payload['steps'])
                finally:
                    for h in handles:
                        h.remove()
                name = f"case{sample['case']}-gain{gain:g}"
                image.save(args.output/(name+'.png'))
                metadata = dict(payload, case=sample['case'], gain=gain, split='dev',
                                adapter_sha256=report['student_sha256'],
                                teacher_sha256=report['teacher_sha256'], model_identity=runtime.identity)
                atomic_json(args.output/(name+'.json'), metadata)
                report['samples'].append(dict(path=name+'.png', **metadata))
                save()
                print('rendered', name, flush=True)
    finally:
        runtime.close()


if __name__ == '__main__':
    with GpuLease(Path('/tmp'), os.environ['ANIMA_GPU_LEASE']):
        main()
