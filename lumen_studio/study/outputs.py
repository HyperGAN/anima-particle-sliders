"""Render matched native images and measure a single signed concept change."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
from statistics import fmean

import torch
from particle_sliders.alpha_tune_jev import JevJudge, _Image
from particle_sliders.candidate_failure import FileRecord
from particle_sliders.strength_schedule import interval_strengths

from ..contracts import atomic_json, digest
from ..vibe_integrity import FrameIntegrity


def read(path):
    return json.loads(Path(path).read_text())


class Unmeasured(RuntimeError):
    pass


@torch.no_grad()
def sample_adapter(rt, variation, gains, prompt, seed):
    emb = rt.encode(prompt)
    latent = rt.noise(seed, 512, 512)
    scheduler = rt.scheduler(10)
    for timestep, strength in zip(scheduler.timesteps, gains, strict=True):
        with rt.mixer.scales({variation: strength}):
            velocity = rt.predict(latent, timestep, emb)
        latent = rt.step(scheduler, velocity, timestep, latent)
    return rt.decode(latent)


class Outputs:
    def __init__(self, root, runtime, concept, rubric):
        self.root = root
        self.runtime = runtime
        self.rubric = rubric
        self.judge = JevJudge(root / "judgments", concept)
        self.integrity = FrameIntegrity(root / "judgments")

    @torch.no_grad()
    def image(self, prompt, seed, *, raw=None, variation=None, gains=None):
        rt = self.runtime
        identity = dict(
            prompt=prompt,
            seed=seed,
            runtime=rt.identity,
            size=512,
            sampler="Euler10-CFG1",
            raw=raw,
            variation=variation,
            gains=gains,
        )
        dest = self.root / "images" / digest(identity)
        dest.mkdir(parents=True, exist_ok=True)
        receipt = dest / "render.json"
        path = dest / "image.png"
        if receipt.exists():
            recorded = read(receipt)
            actual = FileRecord.from_path(path)
            if recorded != dict(identity=identity, image=asdict(actual)):
                raise ValueError(f"Changed render at {receipt}")
            return actual
        if raw is None:
            with rt.mixer.scales({}):
                img = rt.render(prompt, seed, width=512, height=512, steps=10)
        else:
            img = sample_adapter(rt, variation, gains, prompt, seed)
        tmp = dest / "image.pending.png"
        img.convert("RGB").save(tmp)
        tmp.replace(path)
        record = FileRecord.from_path(path)
        atomic_json(receipt, dict(identity=identity, image=asdict(record)))
        return record

    def score(self, record):
        result = self.judge._decide(
            (_Image.read(record),),
            "Assess the visible image alone, without making assumptions from prompts.",
            {"effect": self.rubric},
        )
        if result is None:
            raise Unmeasured(f"Unmeasured effect for {record.sha256}")
        return dict(
            output=asdict(record),
            judgment={"score": result["effect"]},
            integrity=self.integrity.check(record),
        )

    def compare(self, cases, variation, raw, alpha, profile, *, owner=None):
        sigmas = self.runtime.scheduler(10).sigmas.tolist()
        gains = list(
            interval_strengths(
                alpha, [b - a for a, b in zip(sigmas, sigmas[1:])], profile=profile
            )
        )
        observation = owner.scheduled(gains) if owner else nullcontext()
        with observation, ThreadPoolExecutor(4) as pool:
            work = []
            for case in cases:
                record = self.image(
                    case["prompt"],
                    case["seed"],
                    raw=raw.sha256,
                    variation=variation,
                    gains=gains,
                )
                work.append((case, pool.submit(self.score, record)))
            pairs = []
            for case, job in work:
                on = job.result()
                delta = on["judgment"]["score"] - case["off"]["judgment"]["score"]
                pairs.append(dict(case=case, on=on, gain=delta))
        statuses = {p["on"]["integrity"]["status"] for p in pairs}
        status = (
            "fail"
            if "fail" in statuses
            else "unmeasured"
            if "unmeasured" in statuses
            else "pass"
        )
        return dict(
            raw=asdict(raw),
            profile=profile,
            alpha=alpha,
            gains=gains,
            sigmas=sigmas,
            pairs=pairs,
            status=status,
            mean_gain=fmean(p["gain"] for p in pairs),
        )
