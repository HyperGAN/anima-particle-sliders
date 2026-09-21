"""Clone-local particle execution for ComfyUI Anima models."""
import math
import threading
import weakref
import torch
try:
    from .particle_io import ParticleNetwork, comfy_name
except ImportError:
    from particle_io import ParticleNetwork, comfy_name

_LOCKS=weakref.WeakKeyDictionary()
_LOCK_GUARD=threading.Lock()


def model_lock(model):
    with _LOCK_GUARD:
        return _LOCKS.setdefault(model,threading.RLock())


class ParticleWrapper:
    def __init__(self,transformer,network,strength,previous=None):
        self.transformer=transformer
        self.network=network
        self.strength=strength
        self.previous=previous
        modules=dict(transformer.named_modules())
        self.modules=[modules[comfy_name(name)] for name in network.names]
        for module,branch in zip(self.modules,network.branches):
            if tuple(module.weight.shape)!=(branch.up.out_features,branch.down.in_features):
                raise ValueError('ComfyUI projection shape does not match the adapter')

    def to(self,device):
        if not isinstance(device,torch.dtype):
            self.network.to(device=device,dtype=torch.float32)
        if hasattr(self.previous,'to'):
            self.previous.to(device)
        return self

    def __call__(self,model_function,arguments):
        def call():
            if self.previous is not None:
                return self.previous(model_function,arguments)
            return model_function(arguments['input'],arguments['timestep'],**arguments['c'])
        if self.strength==0:
            return call()
        with model_lock(self.transformer):
            self.network.to(device=arguments['input'].device,dtype=torch.float32)
            handles=[]
            try:
                for module,branch in zip(self.modules,self.network.branches):
                    def hook(module,args,output,branch=branch):
                        delta=branch(args[0],self.network.particles)*self.strength
                        return output+delta.to(output.dtype)
                    handles.append(module.register_forward_hook(hook))
                return call()
            finally:
                for handle in handles:
                    handle.remove()


class AnimaParticleSlider:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {'required':{'model':('MODEL',),'adapter_name':(folder_paths.get_filename_list('loras'),),
                'strength':('FLOAT',{'default':1.,'min':0.,'max':5.,'step':.05})}}

    RETURN_TYPES=('MODEL',)
    FUNCTION='load'
    CATEGORY='NTC/Anima'

    def load(self,model,adapter_name,strength):
        import folder_paths
        if not math.isfinite(strength) or not 0<=strength<=5:
            raise ValueError('Strength must be between 0 and 5')
        clone=model.clone()
        if strength==0:
            return (clone,)
        transformer=clone.model.diffusion_model
        if transformer.__class__.__name__!='Anima':
            raise ValueError('Use the Anima Turbo v1.1 diffusion model')
        path=folder_paths.get_full_path_or_raise('loras',adapter_name)
        network,_=ParticleNetwork.load(path)
        previous=clone.model_options.get('model_function_wrapper')
        clone.set_model_unet_function_wrapper(ParticleWrapper(transformer,network,strength,previous))
        return (clone,)

NODE_CLASS_MAPPINGS={'NTCAnimaParticleSlider':AnimaParticleSlider}
NODE_DISPLAY_NAME_MAPPINGS={'NTCAnimaParticleSlider':'Anima Particle Slider (ntc-ai)'}
