"""Fit ordinary rank-8 LoRAs to native particle branches on cached train activations."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lumen_studio import execution
import torch
from safetensors.torch import save_file, load_file
from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.particles import ParticleAdapter
from lumen_studio.cache import TargetCache
from lumen_studio.contracts import file_hash


def comfy_name(name):
    name = name.replace('transformer_blocks.', 'blocks.').replace('.attn1.', '.self_attn.').replace('.attn2.', '.cross_attn.')
    for a,b in (('to_q','q_proj'),('to_k','k_proj'),('to_v','v_proj'),('to_out.0','output_proj')):
        name = name.replace(a,b)
    return name


def save_loras(adapter, downs, output, variation, teacher):
    native, comfy = {}, {}
    for name, branch, down in zip(adapter.names, adapter.branches, downs):
        up = branch.up.weight.detach().cpu().contiguous()
        down = down.cpu().contiguous()
        native[name + '.lora_A.weight'] = down
        native[name + '.lora_B.weight'] = up
        prefix = 'diffusion_model.' + comfy_name(name)
        comfy[prefix + '.lora_down.weight'] = down
        comfy[prefix + '.lora_up.weight'] = up
        comfy[prefix + '.alpha'] = torch.tensor(8.)
    metadata = dict(format='anima-distilled-rank8-v1', teacher_sha256=file_hash(teacher),
                    method='ridge regression of routed bottleneck output on full projection input', rank='8')
    for fmt, values in (('native',native),('comfyui',comfy)):
        folder = output / fmt
        folder.mkdir(parents=True, exist_ok=True)
        save_file(values, str(folder / f'{variation}.safetensors'), metadata=metadata)


def attach_lora(transformer, path, strength):
    state = load_file(str(path))
    modules = dict(transformer.named_modules())
    handles = []
    for key in state:
        if not key.endswith('.lora_A.weight'): continue
        name = key.removesuffix('.lora_A.weight')
        device=modules[name].weight.device
        down, up = state[key].to(device), state[name+'.lora_B.weight'].to(device)
        def hook(module,args,output,down=down,up=up):
            if strength == 0: return output
            x = args[0].float()
            delta = (x @ down.to(x.device).T) @ up.to(x.device).T
            return output + (strength * delta).to(output.dtype)
        handles.append(modules[name].register_forward_hook(hook))
    return handles


@torch.no_grad()
def collect(runtime, adapter, cache, *, training):
    inputs=[[] for _ in adapter.names]
    targets=[[] for _ in adapter.names]
    modules = dict(runtime.transformer.named_modules())
    handles=[]
    for i,name in enumerate(adapter.names):
        def hook(module,args,output,i=i):
            x=args[0].detach().float().reshape(-1,args[0].shape[-1])
            idx=torch.linspace(0,len(x)-1,min(8,len(x)),device=x.device).long()
            x=x[idx]
            branch=adapter.branches[i]
            y=branch.routed(branch.down(x),adapter.particles)
            inputs[i].append(x.cpu())
            targets[i].append(y.cpu())
        handles.append(modules[name].register_forward_hook(hook))
    seen=set(); records=[]
    for si,shard in enumerate(cache.index['shards']):
        if shard['row'] in seen: continue
        seen.add(shard['row'])
        for t in ([0,2,4,6,9] if training else [1,5,8]):
            # First seed per row. Train: neutral 0/2/4/6, positive 9; dev: positive 1/5, neutral 8.
            records.append(si*20 + t + (10 if (si+t)%2 else 0))
    try:
        for n,idx in enumerate(records):
            rec=cache[idx]
            runtime.predict(rec['latent'].to(runtime.device),rec['timestep'],rec['embedding'])
            if n%20==0: print('capture',training,n,len(records),flush=True)
    finally:
        for h in handles: h.remove()
    return [(torch.cat(x),torch.cat(y)) for x,y in zip(inputs,targets)]


@torch.no_grad()
def fit(adapter, train, dev):
    downs=[]; reports=[]
    # Fixed ridge chosen before held-out evaluation. One solve per projection.
    for i,(branch,(xc,yc),(xv,yv)) in enumerate(zip(adapter.branches,train,dev)):
        device=branch.up.weight.device
        x,y=xc.to(device),yc.to(device)
        gram=x@x.T
        ridge=1e-2*gram.diag().mean().clamp_min(1e-8)
        gram.diagonal().add_(ridge)
        down=(torch.linalg.solve(gram,y).T@x)
        up=branch.up.weight
        pred=xv.to(device)@down.T
        truth=yv.to(device)
        delta=(pred-truth)@up.T
        target=truth@up.T
        error=float(delta.square().sum()); norm=float(target.square().sum())
        reports.append(dict(target=adapter.names[i],train_tokens=len(x),dev_tokens=len(xv),
            ridge=float(ridge),squared_error=error,teacher_squared_norm=norm,
            relative_mse=error/max(norm,1e-20)))
        downs.append(down.cpu())
        if i%28==0: print('fit',i,len(adapter.names),flush=True)
    total=sum(r['squared_error'] for r in reports)/max(sum(r['teacher_squared_norm'] for r in reports),1e-20)
    return downs,dict(heldout_projection_relative_mse=total,projections=reports)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True,help='Prepared model, targets and runs')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=Path,required=True,help='JSON release sample requests')
    p.add_argument('--particle-samples-output',type=Path)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.25)
    runtime=TurboRuntime(args.root/'model','cuda:0')
    samples=json.loads(args.samples.read_text())
    for variation in ('candlelit','moonlit'):
        teacher=args.root/'runs'/variation/'ema-001600.safetensors'
        adapter=ParticleAdapter(runtime.transformer).to(runtime.device)
        adapter.load_export(teacher,model_identity=runtime.identity)
        adapter.requires_grad_(False).eval()
        runtime.mixer.add(variation,adapter)
        report_path=args.output/f'{variation}-evaluation.json'
        complete=report_path.exists() and all((args.output/fmt/f'{variation}.safetensors').exists() for fmt in ('native','comfyui'))
        if complete:
            report=json.loads(report_path.read_text())
            if report['teacher_sha256']!=file_hash(teacher):raise ValueError('Resume teacher changed')
        else:
            with runtime.mixer.scales({variation:1.}):
                train=collect(runtime,adapter,TargetCache(args.root/'targets'/variation/'train',verify=False),training=True)
                dev=collect(runtime,adapter,TargetCache(args.root/'targets'/variation/'dev',verify=False),training=False)
        if args.particle_samples_output:
            for sample in samples:
                if sample['variation'] != variation: continue
                dest=args.particle_samples_output/variation/f"case{sample['case']}"/f"str{sample['strength']}"
                if dest.with_suffix('.png').exists(): continue
                payload=sample['payload']
                with runtime.mixer.scales({variation:sample['strength']}):
                    image=runtime.render(payload['prompt'],payload['seed'],payload['width'],payload['height'],payload['steps'])
                dest.parent.mkdir(parents=True,exist_ok=True)
                image.save(dest.with_suffix('.png'))
                dest.with_suffix('.json').write_text(json.dumps(dict(**payload,format='particles',
                    strength=sample['strength'],split='dev',model_identity=runtime.identity,
                    adapter_sha256=file_hash(teacher)),indent=2)+'\n')
                print('particle',dest.name,flush=True)
        if not complete:
            downs,report=fit(adapter,train,dev)
            save_loras(adapter,downs,args.output,variation,teacher)
            report.update(variation=variation,teacher_sha256=file_hash(teacher),method='rank8-ridge-full-input',ridge_fraction=.01)
            report_path.write_text(json.dumps(report,indent=2)+'\n')
            del train,dev,downs
        runtime.mixer.close()
        for sample in samples:
            if sample['variation'] != variation or sample['strength'] not in (1,3,5): continue
            payload=sample['payload']; strength=sample['strength']
            name=f"{variation}-case{sample['case']}-str{strength}"
            dest=args.output/'samples'; dest.mkdir(exist_ok=True)
            if (dest/(name+'.png')).exists() and (dest/(name+'.json')).exists():continue
            path=args.output/'native'/f'{variation}.safetensors'
            handles=attach_lora(runtime.transformer,path,strength)
            try:
                image=runtime.render(payload['prompt'],payload['seed'],payload['width'],payload['height'],payload['steps'])
            finally:
                for h in handles:h.remove()
            name=f"{variation}-case{sample['case']}-str{strength}"
            dest=args.output/'samples'; dest.mkdir(exist_ok=True)
            image.save(dest/(name+'.png'))
            (dest/(name+'.json')).write_text(json.dumps(dict(**payload,format='distilled',adapter_sha256=file_hash(path)),indent=2)+'\n')
            print('rendered',name,flush=True)
    runtime.close()

if __name__=='__main__': main()
