"""Paired-error game. The schedule and losses come from winning_formulation()."""
import math

import torch

from particle_sliders import winning_formulation

STAMP = winning_formulation()
# Generator step size is an Anima model surface. Every other knob stays on the stamp.
TRAINING_CONFIG = {**STAMP.as_dict(), "g_lr": 2e-5}
STAMP.require(TRAINING_CONFIG)
HORIZON = int(STAMP.spec["noise_decay_steps"])


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


RECIPE = _jsonable(TRAINING_CONFIG)


class Sampler:
    def __init__(self, count, seed, batch=8):
        self.count, self.batch = count, batch
        self.generators = {name: torch.Generator().manual_seed(seed + i * 1009)
                           for i, name in enumerate(("d_rows", "g_rows", "d_noise", "g_noise", "vic"))}

    def draw(self, phase, dimension):
        indices = torch.randint(self.count, (self.batch,), generator=self.generators[f"{phase}_rows"]).tolist()
        noise = torch.randn(self.batch, dimension, generator=self.generators[f"{phase}_noise"])
        return indices, noise

    def state_dict(self):
        return dict(count=self.count, batch=self.batch, generators={k: g.get_state() for k, g in self.generators.items()})

    def load_state_dict(self, state):
        if (state["count"], state["batch"]) != (self.count, self.batch):
            raise ValueError("Sampler dimensions changed")
        for name, value in state["generators"].items():
            self.generators[name].set_state(value.cpu())


def grad_norm(parameters):
    norms = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(norms).sum().sqrt()) if norms else 0.


class Game:
    def __init__(self, adapter, normalization, count, seed=7, microbatch=1):
        if microbatch not in (1, 2, 4, 8):
            raise ValueError("Microbatch must divide effective batch eight")
        self.stamp = STAMP
        self.adapter, self.microbatch = adapter, microbatch
        self.device = adapter.particles.device
        self.scale = normalization["scale"].to(self.device)
        self.edit_rms = normalization["edit_rms"].to(self.device)
        dimension = self.scale.shape[-1]
        # Residuals arriving here are already whitened by the Anima target cache.
        # Zeros and ones keep the shared critic builder from rescaling them again.
        targets = torch.stack((torch.zeros(dimension), torch.ones(dimension)))
        self.critic = self.stamp.critic(targets).to(self.device)
        betas = tuple(TRAINING_CONFIG["betas"])
        self.g = torch.optim.Adam([
            dict(params=[p for n, p in adapter.named_parameters() if n != "particles"],
                 lr=float(TRAINING_CONFIG["g_lr"])),
            dict(params=[adapter.particles], lr=float(TRAINING_CONFIG["particle_lr"]))],
            betas=betas, weight_decay=0.)
        self.d = torch.optim.Adam(self.critic.parameters(), lr=float(TRAINING_CONFIG["d_lr"]),
                                   betas=betas, weight_decay=0.)
        self.sampler = Sampler(count, seed, batch=int(TRAINING_CONFIG["adv_batch"]))
        self.capper = self.stamp.regularizer()
        self.d_loss, self.g_loss, self.vic_loss = self.stamp.losses()

    def noise_level(self, step, position):
        """Shared stamp curve. Update k uses schedule index k-1."""
        if step < 1:
            raise ValueError("Updates are numbered starting at one")
        return self.stamp.noise_std_at(step - 1, float(self.edit_rms[position]))

    def update(self, predict_residual, positions, step, prefetch=None):
        """predict_residual(indices) returns ΔS−ΔT = v_adapted(neutral)−v_frozen(positive).

        Caller holds adapter scales active through this entire method, including
        checkpoint recomputation. Noise and row streams are independent for D/G.
        """
        if step < 1:
            raise ValueError("Updates are numbered starting at one")
        dimension = self.scale.shape[-1]
        self.critic.requires_grad_(True)
        self.d.zero_grad(set_to_none=True)
        di, dn = self.sampler.draw("d", dimension)
        gi, gn = self.sampler.draw("g", dimension)
        if prefetch is not None:
            prefetch(di + gi)
        sigma_by_t = [self.noise_level(step, t) for t in range(self.edit_rms.shape[0])]

        def noise_and_scale(indices, noise):
            ts = [positions[i] for i in indices]
            sigma = torch.tensor([sigma_by_t[t] for t in ts], device=self.device)
            return noise.to(self.device) * sigma[:, None], self.scale[ts]

        da = cap = 0.
        for start in range(0, len(di), self.microbatch):
            ids = di[start:start + self.microbatch]
            noise, scale = noise_and_scale(ids, dn[start:start + len(ids)])
            with torch.no_grad():
                fake = noise + predict_residual(ids) / scale
            adv = self.d_loss(self.critic(noise), self.critic(fake))
            penalty, _ = self.capper.penalty(self.critic, noise, fake, step=step, collect_stats=False)
            loss = (adv + penalty) * len(ids) / len(di)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite critic loss")
            loss.backward()
            da += float(adv.detach()) * len(ids) / len(di)
            cap += float(penalty.detach()) * len(ids) / len(di)
        d_norm = grad_norm(self.critic.parameters())
        if not math.isfinite(d_norm):
            raise FloatingPointError("Non-finite critic gradient")
        self.d.step()
        self.d.zero_grad(set_to_none=True)
        self.critic.requires_grad_(False)
        self.g.zero_grad(set_to_none=True)
        ga = 0.
        for start in range(0, len(gi), self.microbatch):
            ids = gi[start:start + self.microbatch]
            noise, scale = noise_and_scale(ids, gn[start:start + len(ids)])
            with torch.no_grad():
                real_score = self.critic(noise)
            fake = noise + predict_residual(ids) / scale
            loss = self.g_loss(real_score, self.critic(fake)) * len(ids) / len(gi)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite generator loss")
            loss.backward()
            ga += float(loss.detach())
        particle_gan_norm = grad_norm([self.adapter.particles])
        parts = int(self.stamp.spec["parts"])
        vic_batch = int(self.stamp.spec["particle_vic_batch"])
        selection = torch.randperm(parts, generator=self.sampler.generators["vic"])[:vic_batch].to(self.device)
        vic = self.vic_loss(self.adapter.particles[selection])
        vic.backward()
        g_norm = grad_norm(self.adapter.parameters())
        if not math.isfinite(g_norm):
            raise FloatingPointError("Non-finite generator gradient")
        self.g.step()
        return dict(step=step, d_adv=da, d_penalty=cap, g_adv=ga, vic=float(vic.detach()),
                    d_grad_norm=d_norm, g_grad_norm=g_norm, particle_gan_grad_norm=particle_gan_norm,
                    noise_min=min(sigma_by_t), noise_max=max(sigma_by_t), d_rows=di, g_rows=gi)

    def state_dict(self):
        return dict(critic=self.critic.state_dict(), g=self.g.state_dict(), d=self.d.state_dict(),
                    sampler=self.sampler.state_dict())

    def load_state_dict(self, state):
        self.critic.load_state_dict(state["critic"])
        self.g.load_state_dict(state["g"])
        self.d.load_state_dict(state["d"])
        self.sampler.load_state_dict(state["sampler"])
