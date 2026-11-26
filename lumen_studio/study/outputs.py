"""Render matched native images and measure a single signed concept change."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
from statistics import fmean

import torch
from particle_sliders.alpha_tune_jev import _Image
from particle_sliders.candidate_failure import FileRecord
from particle_sliders.strength_schedule import interval_strengths

from ..contracts import atomic_json, digest
from .objectives import (BOUND, DISTINCT_BOSS, MATCHED_GAIN, default_objective, measure_design,
                         measure_working, protocol_id)
from .working_score import DrawingReader
from .judge import StudyJudge
from .pair_checks import EFFECT_QUESTIONS, PAIR_PROTOCOLS, pair_status
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
    def __init__(self, root, runtime, concept, rubric, *, objective=None):
        self.root = root
        self.runtime = runtime
        self.concept = concept
        self.rubric = rubric
        self.objective = objective or default_objective(concept)
        self.judge = StudyJudge(root / "judgments", concept)
        self.integrity = FrameIntegrity(root / "judgments")
        self.drawing = DrawingReader(root / "judgments" / "drawing")

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

    def boss_teacher_pair(self, neutral, positive):
        images = tuple(
            _Image.read(FileRecord(**side["output"])) for side in (neutral, positive)
        )
        answer = self.judge._decide(
            images,
            "Image 1 is the source. Image 2 is a proposed transformation of its central character.",
            {
                "separate_boss": {
                    "type": "noul",
                    "instructions": (
                        "In image 2, does the original subject remain present as an ordinary person "
                        "while a separate figure supplies the boss design? Look for two distinct figures. "
                        "A single transformed person in armor, a mask or a new pose is a valid edit."
                    ),
                }
            },
        )
        probability = None if answer is None else answer["separate_boss"]
        status = pair_status("separate-boss-v1", probability)
        return dict(
            status=status,
            separate_boss=probability,
            neutral=neutral["output"]["sha256"],
            positive=positive["output"]["sha256"],
        )

    def teacher_pair(self, neutral, positive):
        concept = self.concept.casefold()
        if concept == "final boss":
            return self.boss_teacher_pair(neutral, positive)
        images = tuple(_Image.read(FileRecord(**side["output"])) for side in (neutral, positive))
        result = self.judge._decide(
            images,
            "Image 1 is the neutral source. Image 2 is the proposed target.",
            {"paired_effect": {"type": "noul", "instructions": EFFECT_QUESTIONS[concept]}},
        )
        probability = result["paired_effect"] if result is not None else None
        return dict(status=pair_status(PAIR_PROTOCOLS[concept], probability), paired_effect=probability,
                    neutral=neutral["output"]["sha256"], positive=positive["output"]["sha256"])

    def score(self, record):
        result = self.judge._decide(
            (_Image.read(record),),
            "Assess the visible image alone, without making assumptions from prompts.",
            {"effect": self.rubric},
        )
        return dict(
            output=asdict(record),
            judgment={"score": result["effect"] if result is not None else None},
            integrity=self.integrity.check(record),
        )

    def with_objective(self, row):
        """Old observations can be scored from their saved images without replaying training."""
        if self.objective not in BOUND:
            return row
        identity = dict(protocol=protocol_id(self.objective), raw=row["raw"]["sha256"], gains=row["gains"],
                        cases=[[p["case"]["prompt"], p["case"]["seed"], p["case"]["off"]["output"]["sha256"],
                                p["on"]["output"]["sha256"]] for p in row["pairs"]])
        path = self.root / "design-observations" / (digest(identity) + ".json")
        if path.exists():
            proof = read(path)
            if proof["identity"] != identity or digest(proof["measurement"]) != proof["measurement_sha256"]:
                raise ValueError("Retained Final Boss design measurements differ from their binding")
            measured = proof["measurement"]
        else:
            measured = (measure_design(row["pairs"], self.judge) if self.objective == DISTINCT_BOSS
                        else measure_working(row["pairs"], self.judge, self.drawing))
            atomic_json(path, dict(identity=identity, measurement=measured, measurement_sha256=digest(measured)))
        return dict(row, objective=self.objective, objective_score=measured["score"],
                    design_protocol=identity["protocol"], design_observation=str(path))

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
                before, after = (
                    case["off"]["judgment"]["score"],
                    on["judgment"]["score"],
                )
                delta = (
                    after - before if after is not None and before is not None else None
                )
                pairs.append(dict(case=case, on=on, gain=delta))
        gains_by_case = [p["gain"] for p in pairs]
        measured = all(gain is not None for gain in gains_by_case)
        statuses = {p["on"]["integrity"]["status"] for p in pairs}
        status = (
            "fail"
            if "fail" in statuses
            else "unmeasured"
            if "unmeasured" in statuses or (not measured and getattr(self, "objective", MATCHED_GAIN) not in BOUND)
            else "pass"
        )
        row = dict(
            raw=asdict(raw),
            profile=profile,
            alpha=alpha,
            gains=gains,
            sigmas=sigmas,
            pairs=pairs,
            status=status,
            mean_gain=fmean(gains_by_case) if measured else None,
        )
        return self.with_objective(row) if getattr(self, "objective", MATCHED_GAIN) in BOUND else row
