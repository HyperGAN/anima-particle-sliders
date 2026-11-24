"""Render and replay frozen teacher target recipes for Anima --vibe."""
import json
from pathlib import Path

from .contracts import atomic_json, digest, file_hash

STEPS = 10
DIRECT = (1.0,) * STEPS
EARLY = (1.55, 1.55, 1.45, 1.35, 1.15, 0.95, 0.75, 0.55, 0.4, 0.3)
LATE = (0.15, 0.25, 0.5, 0.75, 1.0, 1.2, 1.4, 1.45, 1.45, 1.2)

# Every recipe is evaluated on the same neutral prompts and initial noise.
RECIPES = {
    "off": ("Neutral reference", (), DIRECT, None),
    "literal": ("Literal vibe suffix", ("literal",), DIRECT, None),
    "append": ("Concrete cues appended", ("append",), DIRECT, None),
    "replace": ("Concrete costume replacement", ("replace",), DIRECT, None),
    "front": ("Costume first", ("front",), DIRECT, None),
    "early": ("Early-weighted costume guidance", ("replace",), EARLY, None),
    "late": ("Late-weighted costume guidance", ("replace",), LATE, None),
    "blend": ("Two-costume velocity blend", ("replace", "alternate"), DIRECT, None),
    "local": ("Center-weighted costume guidance", ("replace",), EARLY, 0.35),
}


def parse_case(row, concept):
    fold, name, seed, person, ordinary, scene, lighting, costume, alternate = row
    framing = f"Anime illustration, waist-up eye-level portrait of one {person}."
    context = f"In {scene}, {lighting}, centered figure with the setting visible around them."
    neutral = f"{framing} {ordinary}. {context}"
    return {
        "id": name,
        "fold": fold,
        "seed": seed,
        "neutral": neutral,
        "prompts": {
            "literal": f"{neutral} The character is a formidable fantasy RPG {concept}.",
            "append": f"{neutral} Give the character {costume}.",
            "replace": f"{framing} Dressed in {costume}. {context}",
            "front": f"Dressed in {costume}. {framing} {context}",
            "alternate": f"{framing} Dressed in {alternate}. {context}",
        },
    }


def make_recipe(case, method):
    label, prompt_names, gains, edge_floor = RECIPES[method]
    return dict(method=method, label=label, neutral=case["neutral"],
                targets=[case["prompts"][p] for p in prompt_names],
                gains=list(gains), edge_floor=edge_floor)


class TargetField:
    """Native velocity field used on both the teacher and neutral trajectories."""

    def __init__(self, runtime, recipe):
        self.runtime = runtime
        self.recipe = recipe
        self.neutral = runtime.encode(recipe["neutral"])
        self.targets = [runtime.encode(prompt) for prompt in recipe["targets"]]
        self.mask = None

    def velocity(self, latent, timestep, index, neutral=None):
        import torch

        predict = self.runtime.predict
        gain = self.recipe["gains"][index]
        floor = self.recipe["edge_floor"]
        if not self.targets or not gain:
            return predict(latent, timestep, self.neutral) if neutral is None else neutral
        target = predict(latent, timestep, self.targets[0])
        if len(self.targets) == 2:
            target = 0.5 * (target + predict(latent, timestep, self.targets[1]))
        if gain == 1 and floor is None:
            return target
        if neutral is None:
            neutral = predict(latent, timestep, self.neutral)
        if floor is not None:
            if self.mask is None:
                height, width = latent.shape[-2:]
                y = torch.linspace(-1, 1, height, device=latent.device)[:, None]
                x = torch.linspace(-1, 1, width, device=latent.device)[None, :]
                self.mask = floor + (1 - floor) * torch.exp(-((x / 0.62)**4 + (y / 0.85)**4))
            gain = gain * self.mask
        return neutral + gain * (target - neutral)

    def render(self, seed, width, height):
        scheduler = self.runtime.scheduler(STEPS)
        latent = self.runtime.noise(seed, width, height)
        for index, timestep in enumerate(scheduler.timesteps):
            velocity = self.velocity(latent, timestep, index)
            latent = self.runtime.step(scheduler, velocity, timestep, latent)
        return self.runtime.decode(latent)


def read(path):
    return json.loads(Path(path).read_text())


def render_case(product, case, method, directory, *, run_identity):
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    png = directory / "image.png"
    receipt = directory / "render.json"
    recipe = make_recipe(case, method)
    binding = dict(case=case["id"], fold=case["fold"], seed=case["seed"],
                   recipe=recipe, run_identity=run_identity, model=product.identity["model"],
                   resolution=product.resolution, steps=STEPS, cfg=1, adapters=[], ema_decay=None)
    if receipt.is_file():
        saved = read(receipt)
        if saved != dict(binding, image_sha256=file_hash(png)):
            raise ValueError(f"Saved render differs from its binding: {receipt}")
        return saved
    if png.exists():
        raise ValueError(f"Image has no completed render receipt: {png}")
    if method == "off" or (len(recipe["targets"]) == 1 and recipe["gains"] == list(DIRECT)):
        prompt = recipe["neutral"] if method == "off" else recipe["targets"][0]
        # Exercise the ordinary --vibe path for the control and pure prompt arms.
        temp = directory / ".native"
        product.render(prompt, case["seed"], temp, request_id=digest(binding))
        (temp / "image.png").replace(png)
        temp.rmdir()
    else:
        runtime = product.get_runtime()
        with torch.no_grad(), runtime.mixer.scales({}):
            image = TargetField(runtime, recipe).render(
                case["seed"], product.resolution, product.resolution)
        temp = png.with_suffix(".partial")
        image.save(temp, format="PNG")
        temp.replace(png)
    saved = dict(binding, image_sha256=file_hash(png))
    atomic_json(receipt, saved)
    return saved
