"""Apply the pinned campaign rubric to matched search and ranking panels."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from statistics import fmean
import json
from hashlib import sha256

from particle_sliders.alpha_tune_jev import (
    JEV_MODEL,
    JEV_URL,
    _Image,
    _digest,
    _response_values,
)
from particle_sliders.candidate_failure import FileRecord
from lumen_studio.contracts import atomic_json, file_hash
from lumen_studio.study.judge import StudyJudge
from lumen_studio.study.outputs import read
from release_tools.constant_alpha_retest import search_constant
from release_tools.final_boss_design_scores import summarize
from release_tools.final_boss_visible_damage import (
    STATE,
    QUESTIONS,
    VERSION,
    damage,
    score_case,
)

from locations import RESULTS as here

hist = here.parent.parent / "anima-final-boss-vibe12-20261124"
cfg = read(here / "panels.json")
from lumen_studio.study.objectives import protocol

if any(cfg[key] != value for key, value in protocol().items()):
    raise ValueError(
        "The research leaderboard and native Final Boss trainer use different judge protocols"
    )
cases = {x["id"]: x for x in cfg["cases"]}
assert cfg["model"] == JEV_MODEL
assert (
    sha256(
        json.dumps(
            dict(state=STATE, questions=QUESTIONS, version=VERSION, model=JEV_MODEL),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    == cfg["binding"]
)
design_dir = hist / "metric-calibration/design-range"
judge = StudyJudge(here / "judgments", "Final Boss")
D = "distinct_boss_score"
G = "matched_boss_gain"


@lru_cache(maxsize=256)
def image(path, sha):
    r = FileRecord.from_path(Path(path))
    assert r.sha256 == sha
    return _Image.read(r)


def ask(refs, state, questions, stores=()):
    imgs = tuple(image(ref["path"], ref["sha256"]) for ref in refs)
    binding = dict(
        endpoint=JEV_URL,
        model=JEV_MODEL,
        state=state,
        questions=questions,
        images=[i.identity() for i in imgs],
    )
    key = _digest(binding)
    for store in stores:
        p = store / (key + ".json")
        if p.is_file():
            r = read(p)
            if r["status"] == "ok":
                assert r["request"] == binding and r["response_sha256"] == _digest(
                    r["response"]
                )
                return _response_values(r["response"], questions), key
    return judge._decide(imgs, state, questions), key


def distinction(off, on):
    p = hist / "boss-character-review" / (off["sha256"] + "-" + on["sha256"] + ".json")
    if p.is_file():
        v = read(p)
        assert v["rubric"] == cfg["boss"] and (v["off_sha"], v["on_sha"]) == (
            off["sha256"],
            on["sha256"],
        )
        if v["value"] is not None:
            return v["value"]
    v, _ = ask([off, on], **cfg["boss"])
    return v["boss_distinction"] if v else None


def broken(on):
    p = hist / "metric-calibration/ratings" / cfg["binding"] / (on["sha256"] + ".json")
    if p.is_file():
        v = read(p)
        assert (
            v["binding"] == cfg["binding"]
            and v["image"] == on["sha256"]
            and v["severity"] == damage(v["answers"])
        )
        return v["severity"]
    v, _ = ask([on], STATE, QUESTIONS)
    return damage(v) if v else None


def pair(a, b, rows):
    x, y = rows[a]["on"], rows[b]["on"]
    measures = []
    for refs in ([x, y], [y, x]):
        answer, key = ask(
            refs,
            **cfg["repeat"],
            stores=[
                design_dir / n
                for n in ("receipts", "receipts-retry-1", "receipts-retry-2")
            ],
        )
        if answer is None:
            return None
        measures.append(dict(value=answer["repeat"], receipt=key))
    return dict(
        cases=[a, b],
        images=[x["sha256"], y["sha256"]],
        forward=measures[0],
        reverse=measures[1],
        repeat=fmean(v["value"] for v in measures),
    )


def rate(case, on, distinct):
    off = cases[case]["off"]
    assert file_hash(off["path"]) == off["sha256"]
    before = distinction(off, off)
    after = distinction(off, on)
    row = dict(off=off, on=on, original_boss_distinction=before, boss_distinction=after)
    if distinct:
        row["image_breakage"] = broken(on)
    return row


def retest(name, outputs, raw, objective, *, archive=True):
    assert objective in (D, G)
    root = outputs.root / objective
    distinct = objective == D
    sources = [
        here.parent.parent
        / "anima-final-boss-open-alpha-20261125/leaderboard-retune-open"
        / name
    ]
    sources += [
        hist / "strength-search" / n / name
        for n in (
            "leaderboard-retune-design",
            "leaderboard-retune-visible",
            "leaderboard-retune-damage",
            "leaderboard-retune-adaptive",
            "leaderboard-retune-centered",
            "leaderboard-retune",
        )
    ]
    sources.append(hist / "tuning" / name)
    if archive:
        distinct_profile = read(hist / "strength-search" / (name + ".json"))
        assert distinct_profile["checkpoint"]["sha256"] == raw["sha256"]
    else:
        sources = []
    for src in sources[:-1]:
        if (src / "request.json").is_file():
            req = read(src / "request.json")
            definition = req.get("definition")
            if definition:
                assert file_hash(definition["path"]) == definition["sha256"]
                assert read(definition["path"])["checkpoint"]["sha256"] == raw["sha256"]
    baseline_model = (
        read(outputs.root / "pg012-parity.json")["training_model"]
        if (outputs.root / "pg012-parity.json").exists()
        else None
    )

    def render(case, alpha):
        suffix = Path("renders/constant") / alpha.hex() / case
        for src in sources:
            rec = src / suffix / "render.json"
            if not rec.is_file():
                continue
            r = read(rec)
            assert r["model"] == baseline_model and r["gains"] == [alpha] * 10
            assert (
                all(r[k] == v for k, v in cases[case]["request"].items())
                and file_hash(r["image"]["path"]) == r["image"]["sha256"]
            )
            return r["image"]
        return asdict(
            outputs.image(
                **cases[case]["request"],
                raw=raw["sha256"],
                variation=name,
                gains=[alpha] * 10,
            )
        )

    with ThreadPoolExecutor(6) as pool:

        def evaluate(alpha, phase):
            path = root / phase / (alpha.hex() + ".json")
            ident = dict(
                method=name,
                raw=raw["sha256"],
                alpha=alpha,
                gains=[alpha] * 10,
                objective=objective,
                phase=phase,
                cases=cfg["panels"][phase],
            )
            if path.is_file():
                doc = read(path)
                assert doc["identity"] == ident
                return doc
            names = ident["cases"]
            # Reuse the exact PNG and pinned source/target judgment from the original campaign.
            rows = {}
            for c in names:
                on = render(c, alpha)
                rows[c] = pool.submit(rate, c, on, distinct)
            rows = {c: r.result() for c, r in rows.items()}
            is_measured = all(
                r["boss_distinction"] is not None
                and r["original_boss_distinction"] is not None
                for r in rows.values()
            )
            gain = (
                fmean(
                    r["boss_distinction"] - r["original_boss_distinction"]
                    for r in rows.values()
                )
                if is_measured
                else None
            )
            pairs = []
            design = None
            if (
                distinct
                and is_measured
                and all(r["image_breakage"] is not None for r in rows.values())
            ):
                jobs = [
                    pool.submit(pair, a, b, rows) for a, b in combinations(names, 2)
                ]
                pairs = [j.result() for j in jobs]
                if all(p is not None for p in pairs):
                    design = summarize(
                        {c: score_case(r) for c, r in rows.items()}, pairs
                    )
            score = (design["score"] if design else None) if distinct else gain
            doc = dict(
                identity=ident,
                score=score,
                mean_gain=gain,
                design=design,
                pairs=pairs,
                rows=rows,
            )
            atomic_json(path, doc)
            print(name, phase, alpha, score, flush=True)
            return doc

        settings = search_constant(lambda a: evaluate(a, "search")["score"])
        usable = [a for a in settings if settings[a] is not None]
        if not usable:
            atomic_json(
                root / "result.json",
                dict(
                    method=name,
                    raw=raw["sha256"],
                    objective=objective,
                    decision="unmeasured",
                    trials=list(settings.items()),
                ),
            )
            return
        alpha = max(usable, key=lambda a: (settings[a], -a))
        pin = dict(
            method=name,
            raw=raw["sha256"],
            objective=objective,
            alpha=alpha,
            trials=list(map(list, settings.items())),
            search_score=settings[alpha],
        )
        reservation = root / "selected.json"
        if reservation.is_file():
            assert read(reservation) == pin
        else:
            atomic_json(reservation, pin)
        result = evaluate(alpha, "ranking")
        integrity = outputs.integrity
        fixture = integrity.calibrate()
        assert fixture["passed"]

        def assess(row):
            r = row["on"]
            return integrity.check(FileRecord.from_path(Path(r["path"])))

        checks = dict(
            zip(result["rows"], pool.map(assess, result["rows"].values()), strict=True)
        )
        states = {r["status"] for r in checks.values()}
        status = (
            "fail"
            if "fail" in states
            else "unmeasured"
            if "unmeasured" in states
            else "pass"
        )
        atomic_json(
            root / "result.json",
            dict(
                pin,
                score=result["score"],
                mean_gain=result["mean_gain"],
                checks=checks,
                integrity=status,
                decision="measured" if result["score"] is not None else "unmeasured",
            ),
        )


def independent_effect_outputs(base):
    """Read any verified common image and isolate simultaneous new effect-only renders."""
    from lumen_studio.contracts import digest
    from lumen_studio.study.outputs import Outputs

    fresh = Outputs(base.root / "signed-native", base.runtime, "Final Boss", {})

    class EffectReplay(Outputs):
        def image(self, prompt, seed, *, raw=None, variation=None, gains=None):
            ident = dict(
                prompt=prompt,
                seed=seed,
                runtime=self.runtime.identity,
                size=512,
                sampler="Euler10-CFG1",
                raw=raw,
                variation=variation,
                gains=gains,
            )
            receipt = self.root / "images" / digest(ident) / "render.json"
            if receipt.is_file():
                recorded = read(receipt)
                assert recorded["identity"] == ident
                asset = FileRecord(**recorded["image"])
                asset.verify()
                return asset
            return fresh.image(prompt, seed, raw=raw, variation=variation, gains=gains)

    return EffectReplay(base.root, base.runtime, "Final Boss", {})
