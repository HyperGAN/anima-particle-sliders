"""Bounded real Turbo LR experiment; produces a new cohort, never mutates caches.

Run only inside a parent-admitted GPU role slot. Historical native cache latents
are inputs; both teachers are freshly recomputed under the exact current runtime.
Heldout characters/seeds remain separated. This is a provisional response-fit
screen, not permission to publish weights or a claim of model-wide quality.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_response_runtime(model_dir):
    """Use the exact native predictor, loading only transformer and scheduler.

    Prompt embeddings are already frozen inputs. VAE/text modules have no role
    in this response experiment and consume other workers' shared VRAM.
    """
    from collections import OrderedDict
    import torch
    from diffusers import ModularPipeline
    from lumen_studio.backends.anima import TurboRuntime
    from lumen_studio.contracts import file_hash
    from lumen_studio.particles import Mixer
    from lumen_studio.provenance import model_identity, verify_environment
    verify_environment()
    runtime = TurboRuntime.__new__(TurboRuntime)
    runtime.model_dir, runtime.device = Path(model_dir), torch.device('cuda:0')
    runtime.dtype = torch.bfloat16
    runtime.lock = json.loads((runtime.model_dir/'anima-lock.json').read_text())
    if runtime.lock['model'] != 'circlestone-labs/Anima:turbo-v1.1':
        raise ValueError('expected original pinned Turbo model')
    for name, expected in runtime.lock['files'].items():
        if file_hash(runtime.model_dir/name) != expected:
            raise ValueError('native model file hash differs: '+name)
    runtime.identity = model_identity(runtime.model_dir)
    runtime.pipe = ModularPipeline.from_pretrained(str(runtime.model_dir), local_files_only=True)
    runtime.pipe.load_components(names=['transformer', 'scheduler'], dtype=runtime.dtype)
    runtime.transformer = runtime.pipe.transformer.requires_grad_(False).eval().to(runtime.device)
    runtime.transformer.enable_gradient_checkpointing()
    runtime.mixer = Mixer(runtime.transformer)
    runtime.embeddings, runtime.embedding_limit = OrderedDict(), 256
    runtime.scale, runtime.channels = runtime.pipe.vae_scale_factor, runtime.transformer.config.in_channels
    return runtime


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--historical-targets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variation', default='candlelit')
    parser.add_argument('--lrs', type=float, nargs='+', default=[3e-5, 1e-4, 3e-4, 1e-3])
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--eval-rows', type=int, default=12)
    parser.add_argument('--max-memory-gib', type=float, required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--fixed-d-lr', type=float)
    parser.add_argument('--fixed-prior-lr', type=float)
    parser.add_argument('--identity-only', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--render', action='store_true')
    return parser.parse_args()


def main():
    from particle_sliders.pg09 import PG_VERSION
    if PG_VERSION != '0.9.4':
        raise RuntimeError('Historical PG094 pilot requires its original pinned source environment; use a new PG095 cohort')
    args = parse_args()
    import torch
    from lumen_studio.cache import TargetCache, save_tensor_file
    from lumen_studio.contracts import atomic_json, digest, file_hash
    from lumen_studio.pg09_training import PG09Trainer
    from particle_sliders.pg09 import _hash

    if args.steps < 2 or args.shards < 2 or args.eval_rows < 2:
        raise ValueError('at least two steps/shards/evaluation rows required')
    if any(not math.isfinite(lr) or lr <= 0 for lr in args.lrs):
        raise ValueError('positive finite learning rates required')
    for name in ('fixed_d_lr','fixed_prior_lr','max_memory_gib'):
        value = getattr(args,name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(name+' must be finite and positive')
    args.output.mkdir(parents=True, exist_ok=True)
    def global_memory_guard():
        while True:
            result = subprocess.run(['nvidia-smi', '--id='+os.environ['CUDA_VISIBLE_DEVICES'],
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                text=True, capture_output=True)
            if result.returncode == 0 and int(result.stdout.strip()) < 5120:
                print('Aborting own Anima run: global free VRAM below 5GiB', flush=True)
                os._exit(75)
            time.sleep(3)
    threading.Thread(target=global_memory_guard, daemon=True).start()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.max_memory_gib * 2**30
        / torch.cuda.get_device_properties(0).total_memory, 0)
    started = time.monotonic()
    runtime = load_response_runtime(args.model_dir)
    report = dict(schema='anima-pg094-native-lr-screen-v1', status='running',
        model=runtime.identity, source_sha256=file_hash(__file__),
        source_commit=os.environ.get('ANIMA_EXPERIMENT_SOURCE_COMMIT'),
        gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
        settings=dict(lrs=args.lrs, steps=args.steps, d_lr_mult=1.5, prior_lr_mult=2.5,
                      fixed_d_lr=args.fixed_d_lr, fixed_prior_lr=args.fixed_prior_lr,
                      comparison=('generator-only screen; fixed D/P absolute rates'
                        if args.fixed_d_lr and args.fixed_prior_lr else 'coupled base-rate screen'),
                      batch=2, seed=args.seed, max_memory_gib=args.max_memory_gib), candidates=[])
    (args.output/'pilot-source.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(args.output / 'report.json', report)

    def memory(stage):
        record = dict(stage=stage, allocated_gib=torch.cuda.memory_allocated()/2**30,
                      peak_gib=torch.cuda.max_memory_allocated()/2**30,
                      seconds=time.monotonic()-started)
        print(json.dumps(record), flush=True)
        return record

    def rebuild(split):
        old = TargetCache(args.historical_targets / args.variation / split, verify=False)
        selected, seen = [], set()
        for shard in old.index['shards']:
            data = torch.load(old.directory / shard['path'], map_location='cpu', weights_only=True)
            row = data['row']
            if row['id'] in seen:
                continue
            seen.add(row['id'])
            if row['split'] != split:
                raise ValueError('historical target split differs')
            if file_hash(old.directory / shard['path']) != shard['sha256']:
                raise ValueError('historical input shard differs from manifest')
            selected.append((shard, data))
            if len(selected) == args.shards:
                break
        identity = dict(model=runtime.identity, split=split, variation=args.variation,
            manifest_sha256=digest(dict(original=old.index['identity']['manifest_sha256'],
                selected=[dict(path=s['path'], sha256=s['sha256']) for s, _ in selected])),
            source_cache_fingerprint=old.index['fingerprint'], teacher='fresh-current-runtime',
            layout='both-trajectories/all-positions/full-velocity/shared-state-v1')
        directory = args.output / 'fresh-targets' / split
        directory.mkdir(parents=True, exist_ok=True)
        fingerprint, shards = digest(identity), []
        with torch.no_grad(), runtime.mixer.scales({}):
            for shard, data in selected:
                path = directory / shard['path']
                if path.exists():
                    fresh = torch.load(path, map_location='cpu', weights_only=True)
                    if fresh['fingerprint'] != fingerprint:
                        raise ValueError('fresh target cohort changed')
                else:
                    if 'positive_embedding' not in data:
                        raise ValueError('original positive embedding unavailable')
                    records = []
                    for record in data['records']:
                        latent = record['latent'].to(runtime.device)
                        neutral = runtime.predict(latent, record['timestep'], data['embedding']).cpu()
                        positive = runtime.predict(latent, record['timestep'], data['positive_embedding']).cpu()
                        records.append(dict(record, neutral=neutral, positive=positive))
                    save_tensor_file(path, dict(data, fingerprint=fingerprint, records=records))
                shards.append(dict(shard, sha256=file_hash(path)))
        index = dict(identity=identity, input_fingerprint=fingerprint,
            fingerprint=digest(dict(input=fingerprint, shards=shards)), shards=shards)
        atomic_json(directory / 'index.json', index)
        return TargetCache(directory)

    trainer = None
    try:
        memory('loaded-model')
        train, dev = rebuild('train'), rebuild('dev')
        train_people = {train[i]['row']['character'] for i in range(len(train))}
        dev_people = {dev[i]['row']['character'] for i in range(len(dev))}
        if train_people & dev_people:
            raise ValueError('heldout character leakage')
        report['caches'] = dict(train=train.index, dev=dev.index)
        # Every new teacher was computed here with all adapters disabled. Verify
        # the fresh cache once more at update admission, not by a boolean flag.
        parity = []
        with torch.no_grad(), runtime.mixer.scales({}):
            for index in (0, len(train)-1):
                record = train[index]
                actual = runtime.predict(record['latent'].to(runtime.device),
                    record['timestep'], record['embedding']).cpu()
                error = float((actual-record['neutral']).abs().max())
                if error != 0:
                    raise ValueError('fresh neutral teacher parity failed')
                parity.append(dict(index=index, max_abs=error))
        report['teacher_parity'] = parity
        frozen = {name: value._version for name, value in runtime.transformer.named_parameters()}

        def admitted(request):
            return (request['model_identity'] == runtime.identity
                and request['cache_fingerprint'] == train.index['fingerprint']
                and all(p['max_abs'] == 0 for p in parity))

        @torch.no_grad()
        def evaluate(adapter):
            # Fixed heldout native fields; error relative to a neutral/no-edit
            # baseline avoids rewarding simply small velocity magnitudes.
            errors, baseline, cosines, gains = [], [], [], []
            indices = torch.linspace(0, len(dev)-1, min(args.eval_rows, len(dev))).long().tolist()
            with runtime.mixer.scales({args.variation: 1.}):
                for index in indices:
                    row = dev[index]
                    prediction = runtime.predict(row['latent'].to(runtime.device),
                        row['timestep'], row['embedding']).cpu()
                    student = (prediction-row['neutral']).float().flatten()
                    teacher = (row['positive']-row['neutral']).float().flatten()
                    errors.append((student-teacher).square().mean())
                    baseline.append(teacher.square().mean())
                    cosines.append(torch.nn.functional.cosine_similarity(student, teacher, dim=0))
                    gains.append((student*teacher).sum()/teacher.square().sum().clamp_min(1e-12))
            return dict(relative_mse=float(torch.stack(errors).mean()/torch.stack(baseline).mean()),
                edit_cosine=float(torch.stack(cosines).mean()), edit_gain=float(torch.stack(gains).mean()),
                rows=indices)

        for lr in args.lrs:
            directory = args.output / f'lr-{lr:.8g}'
            candidate = dict(lr=lr, status='running', steps=[])
            d_mult = args.fixed_d_lr / lr if args.fixed_d_lr is not None else 1.5
            p_mult = args.fixed_prior_lr / lr if args.fixed_prior_lr is not None else 2.5
            report['candidates'].append(candidate)
            atomic_json(args.output / 'report.json', report)
            try:
                trainer = PG09Trainer(runtime, train, directory, args.variation, total_steps=args.steps,
                    lr=lr, d_lr_mult=d_mult, prior_lr_mult=p_mult, seed=args.seed,
                    native_admission=admitted)
                initial = trainer.game.state_dict()
                initial_path = directory/'initial-state.pt'
                if initial_path.exists():
                    raise FileExistsError('initial state is immutable; use a new trial directory')
                save_tensor_file(initial_path, initial)
                candidate.update(absolute_lrs=dict(g=lr, d=lr*d_mult, prior=lr*p_mult),
                    recipe=initial['config']['recipe'], recipe_sha256=_hash(initial['config']['recipe']),
                    config_sha256=_hash(initial['config']), initial_state_sha256=_hash(initial),
                    initial_capture='actual step-zero packet before any update',
                    initial_file_sha256=file_hash(initial_path),
                    initial_network_sha256=_hash(initial['network']),
                    initial_critic_sha256=_hash(initial['critic']),
                    initial_rng_sha256=_hash({name:initial[name] for name in
                        ('streams','model_cpu_rng','model_cuda_rng')}))
                candidate['optimizer_groups_at_step_zero'] = [
                    dict(role=('prior' if any(parameter is trainer.adapter.particles
                                              for parameter in group['params']) else role),
                         lr=group['lr'], parameter_count=sum(p.numel() for p in group['params']))
                    for role, optimizer in (('g',trainer.game.opt_g),('d',trainer.game.opt_d))
                    for group in optimizer.param_groups]
                if args.identity_only:
                    candidate['status']='identity-only'
                    continue
                ema = {name: value.detach().clone() for name, value in trainer.adapter.state_dict().items()}
                candidate['baseline'] = evaluate(trainer.adapter)
                end = 2 if args.probe_only else args.steps
                while trainer.step < end:
                    before = time.monotonic()
                    previous = {name:value.detach().clone()
                                for name,value in trainer.adapter.named_parameters()}
                    metrics = trainer.update()
                    for name, value in trainer.adapter.state_dict().items():
                        ema[name].lerp_(value.detach(), .1)
                    entry = dict(step=trainer.step, seconds=time.monotonic()-before,
                                 **{k:v for k,v in metrics.items()
                                    if k != 'step' and isinstance(v,(int,float))})
                    entry['adapter_update_ratio'] = float(torch.stack([
                        (value.detach()-previous[name]).square().sum()
                        for name,value in trainer.adapter.named_parameters()]).sum().sqrt()
                        / torch.stack([value.square().sum() for value in previous.values()]).sum().sqrt().clamp_min(1e-12))
                    if any(not math.isfinite(v) for v in entry.values() if isinstance(v,float)):
                        raise ValueError('nonfinite training metrics')
                    candidate['steps'].append(entry)
                    print(json.dumps(dict(lr=lr, **entry)), flush=True)
                    if trainer.step == end//2:
                        trainer.save()
                        saved = trainer.game.state_dict()
                        trainer.close()
                        trainer = PG09Trainer(runtime, train, directory, args.variation,
                            total_steps=args.steps, lr=lr, d_lr_mult=d_mult, prior_lr_mult=p_mult, seed=args.seed,
                            native_admission=admitted)
                        if trainer.step != saved['completed_steps']:
                            raise ValueError('resume step mismatch')
                        candidate['resumed_at'] = trainer.step
                trainer.save()
                candidate['final_state_sha256'] = _hash(trainer.game.state_dict())
                candidate['final_packet_sha256'] = file_hash(directory/'pg09-resume.pt')
                candidate['raw'] = evaluate(trainer.adapter)
                raw = deepcopy(trainer.adapter.state_dict())
                trainer.adapter.load_state_dict(ema)
                candidate['ema'] = evaluate(trainer.adapter)
                path = directory / f'ema-{trainer.step:06}.safetensors'
                if not path.exists():
                    trainer.adapter.export(path, model_identity=runtime.identity,
                        normalization=dict(method='pg09-native-field-scalar-scale', response_scale=1.,
                                           cache_fingerprint=train.index['fingerprint']),
                        manifest_hash=train.index['identity']['manifest_sha256'], step=trainer.step, ema=ema)
                trainer.adapter.load_export(path, model_identity=runtime.identity)
                candidate['export_sha256'] = file_hash(path)
                candidate['export_reload'] = evaluate(trainer.adapter)
                if candidate['ema'] != candidate['export_reload']:
                    raise ValueError('portable EMA export changed heldout response')
                if args.render:
                    # Decoder loads only after training has completed; cached
                    # heldout embedding avoids loading unused text encoders.
                    runtime.pipe.load_components(names=['vae'], dtype=runtime.dtype)
                    runtime.pipe.vae.requires_grad_(False).eval().to(runtime.device)
                    runtime.pipe.vae.enable_slicing()
                    embedding = dev[0]['embedding']
                    with runtime.mixer.scales({args.variation: 1.}):
                        with torch.no_grad():
                            latent, scheduler = runtime.noise(29001, 512, 512), runtime.scheduler(10)
                            for timestep in scheduler.timesteps:
                                velocity = runtime.predict(latent, timestep, embedding)
                                latent = runtime.step(scheduler, velocity, timestep, latent)
                            image = runtime.decode(latent)
                    image.save(directory/'heldout-ema.png')
                    candidate['render_sha256'] = file_hash(directory/'heldout-ema.png')
                    runtime.pipe.vae.to('cpu')
                trainer.adapter.load_state_dict(raw)
                if any(value._version != frozen[name]
                       for name,value in runtime.transformer.named_parameters()):
                    raise ValueError('frozen backbone changed')
                candidate.update(status='complete', memory=memory('candidate-complete'))
            except Exception as exc:
                candidate.update(status='failed', error=type(exc).__name__+': '+str(exc),
                                 memory=memory('candidate-failed'))
                atomic_json(args.output/'report.json', report)
                raise
            finally:
                if trainer is not None:
                    trainer.close()
                    trainer = None
                # Mixer retains hooks after the first adapter; a later trainer
                # safely reuses their equivalent module targets.
                torch.cuda.empty_cache()
            atomic_json(args.output/'report.json', report)
            if args.probe_only:
                break
        complete = [c for c in report['candidates'] if c['status']=='complete']
        report.update(status='complete', seconds=time.monotonic()-started,
            provisional_best_lr=(min(complete,key=lambda c:c['ema']['relative_mse'])['lr']
                                 if complete else None),
            qualification='native bounded response screen; wider heldout/render review required')
        atomic_json(args.output/'report.json', report)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
