"""Anima native response binding adapted from Particle Sliders core 0.15.1 (MIT)."""
import torch

from particle_sliders.backends import AdapterParameters, FrozenResponseAdapter, bind_native_gradient_checkpointing


class AnimaRuntimeResponseAdapter(FrozenResponseAdapter):
    """Reuse lumen_studio's TurboRuntime, actual ParticleAdapter and Mixer.

    No copied router/particle prior. The supplied adapter must already be attached
    to this runtime under ``name``. The same actual bank is exposed to core.
    """
    family = "anima_turbo_runtime"
    time_coordinates = "official scheduler timestep; runtime.predict divides by training timestep count"

    def __init__(self, runtime, adapter, name):
        if runtime.mixer.adapters.get(name) is not adapter:
            raise ValueError("attach this actual particle adapter to runtime.mixer first")
        if any(p.requires_grad for p in runtime.transformer.parameters()):
            raise ValueError("Anima runtime base must be frozen")
        self.runtime, self.adapter, self.name = runtime, adapter, name
        network = AdapterParameters(adapter.named_parameters(),
                                    lambda scale: runtime.mixer.scales({name: scale}))
        self.particles = adapter.particles  # the actual routed bank, never a replacement
        super().__init__(network, decoder=runtime.decode, model_identity=runtime.identity)
        bind_native_gradient_checkpointing(runtime.transformer, network)

    def _velocity(self, condition, latent, timestep):
        embedding = self.runtime.encode(condition) if isinstance(condition, str) else condition
        return self.runtime.predict(latent, timestep, embedding)

    def _advance(self, latent, velocity, delta):
        # Match TurboRuntime.step exactly: FP32 tensor scalar times BF16
        # model output, FP32 sample addition, then BF16 output cast.
        velocity = velocity.to(self.runtime.dtype)
        dt = torch.as_tensor(delta, device=latent.device, dtype=torch.float32)
        return (latent.float() + dt * velocity).to(self.runtime.dtype)

    def metadata(self):
        return dict(super().metadata(), model_identity=self.runtime.identity,
                    particle_bank="existing lumen_studio ParticleAdapter.particles",
                    trajectory_rounding=str(self.runtime.dtype))
