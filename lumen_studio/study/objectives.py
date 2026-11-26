"""Register the product objective and retain the measurements that determine its value."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from statistics import fmean

from particle_sliders.alpha_tune_jev import JEV_MODEL, JEV_URL, _Image, _digest
from particle_sliders.candidate_failure import FileRecord

from ..contracts import digest
from . import working_score
from .design_score import STATE, QUESTIONS, VERSION, damage, score_case, summarize

DISTINCT_BOSS = "distinct_boss_score"
WORKING_BOSS = "working_boss_score"
MATCHED_GAIN = "matched_intended_effect"
# Objectives measured by a registered judge protocol on the finished images.
BOUND = (DISTINCT_BOSS, WORKING_BOSS)


def default_objective(concept):
    """Final Boss is selected and ranked by whether ON gives a boss that is not broken.

    The distinct Boss score remains readable for the results recorded with it.
    """
    return WORKING_BOSS if concept.casefold() == "final boss" else MATCHED_GAIN


def selection_score(row, objective=None):
    if objective is not None and row.get("objective", MATCHED_GAIN) != objective:
        raise ValueError("Measurement and selection objective differ")
    metric = row.get("objective", MATCHED_GAIN)
    if metric in BOUND:
        return row["objective_score"]
    if metric == MATCHED_GAIN:
        return row["mean_gain"]
    raise ValueError(f"Unknown study objective {metric}")


@lru_cache(maxsize=1)
def protocol():
    path = Path(__file__).resolve().parents[2] / "configs/anima/final-boss-design.json"
    cfg = json.loads(path.read_bytes())
    binding = dict(state=STATE, questions=QUESTIONS, version=VERSION, model=JEV_MODEL)
    if (
        cfg["model"] != JEV_MODEL
        or sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        != cfg["binding"]
    ):
        raise ValueError("The registered Final Boss rubric or judge has changed")
    return cfg


def protocol_id(objective=DISTINCT_BOSS):
    if objective == WORKING_BOSS:
        return digest(
            dict(
                objective=WORKING_BOSS,
                formula=working_score.VERSION,
                rubric=protocol(),
                drawing=dict(model=working_score.MODEL, state=working_score.STATE,
                             questions=working_score.QUESTIONS, tile=working_score.TILE,
                             span=working_score.SPAN),
                figure=dict(state=working_score.FIGURE_STATE, questions=working_score.FIGURE_QUESTIONS),
                pair=dict(state=working_score.PAIR_STATE, questions=working_score.PAIR_QUESTIONS),
                recolor=dict(state=working_score.RECOLOR_STATE, questions=working_score.RECOLOR_QUESTIONS),
                effect=[working_score.BOSS_FLOOR, working_score.BOSS_ENOUGH],
            )
        )
    if objective != DISTINCT_BOSS:
        raise ValueError(f"{objective} has no registered judge protocol")
    return digest(
        dict(
            objective=DISTINCT_BOSS,
            formula="vibe12-distinct-design-v1",
            rubric=protocol(),
        )
    )


def bound_protocol(objective):
    """The protocol a reserved alpha belongs to, or None for the plain matched gain."""
    return protocol_id(objective) if objective in BOUND else None


def measure_working(pairs, judge, reader):
    """Measure whether each ON image is a final boss, one whole finished figure, in its own scene."""
    cfg = protocol()
    inputs = {}
    for pair in pairs:
        case = pair["case"]
        key = digest([case["prompt"], case["seed"]])
        if key in inputs:
            raise ValueError("Repeated Final Boss case")
        inputs[key] = []
        for record in (case["off"]["output"], pair["on"]["output"]):
            asset = FileRecord(**record)
            asset.verify()
            inputs[key].append((asset, _Image.read(asset)))
    if len(inputs) < 2:
        raise ValueError("The working Boss score requires at least two different cases")

    def rate_case(key):
        (off_file, off), (on_file, on) = inputs[key]
        baseline = judge._decide((off, off), **cfg["boss"])
        edited = judge._decide((off, on), **cfg["boss"])
        paired = judge._decide((off, on), working_score.PAIR_STATE, working_score.PAIR_QUESTIONS)
        whole = judge._decide((on,), working_score.FIGURE_STATE, working_score.FIGURE_QUESTIONS)
        tint = judge._decide((off, on), working_score.RECOLOR_STATE, working_score.RECOLOR_QUESTIONS)
        row = dict(
            off_sha256=off.identity()["sha256"],
            on_sha256=on.identity()["sha256"],
            original_boss_distinction=baseline and baseline["boss_distinction"],
            boss_distinction=edited and edited["boss_distinction"],
            broken=reader.broken(on_file.path),
            off_broken=reader.broken(off_file.path),
            one_figure=paired and paired["one_figure"],
            same_person=paired and paired["same_person"],
            figure=whole and whole["figure"],
            recolored=tint and tint["recolored"],
        )
        row["quality"] = working_score.quality(row["broken"], row["off_broken"], row["one_figure"], row["figure"])
        row["case"] = working_score.score_case(
            row["boss_distinction"], row["original_boss_distinction"], row["quality"], row["recolored"])
        return row

    def rate_pair(keys):
        images = [inputs[k][1][1] for k in keys]
        values = []
        for order in (images, images[::-1]):
            response = judge._decide(tuple(order), **cfg["repeat"])
            values.append(response and response["repeat"])
        return dict(cases=list(keys), images=[im.identity()["sha256"] for im in images],
                    forward=values[0], reverse=values[1],
                    repeat=None if None in values else fmean(values))

    with ThreadPoolExecutor(4) as pool:
        rows = dict(zip(inputs, pool.map(rate_case, inputs), strict=True))
        comparisons = list(pool.map(rate_pair, combinations(inputs, 2)))
    score = working_score.working_score({key: row["case"] for key, row in rows.items()}, comparisons)
    return dict(protocol=protocol_id(WORKING_BOSS), rows=rows, pairs=comparisons, score=score)


def measure_design(pairs, judge):
    """Measure fixed OFF/ONs, visible damage and every pair of ONs in both orders."""
    cfg = protocol()
    inputs = {}
    for pair in pairs:
        case = pair["case"]
        key = digest([case["prompt"], case["seed"]])
        if key in inputs:
            raise ValueError("Repeated Final Boss case")
        inputs[key] = []
        for record in (case["off"]["output"], pair["on"]["output"]):
            asset = FileRecord(**record)
            asset.verify()
            inputs[key].append(_Image.read(asset))
    if len(inputs) < 2:
        raise ValueError("Distinct Boss requires at least two different cases")

    def ask(images, rubric):
        binding = dict(
            endpoint=JEV_URL,
            model=JEV_MODEL,
            **rubric,
            images=[im.identity() for im in images],
        )
        return judge._decide(tuple(images), **rubric), _digest(binding)

    def rate_case(key):
        off, on = inputs[key]
        baseline, baseline_key = ask([off, off], cfg["boss"])
        edited, edited_key = ask([off, on], cfg["boss"])
        accident, damage_key = ask([on], dict(state=STATE, questions=QUESTIONS))
        return dict(
            off_sha256=off.identity()["sha256"],
            on_sha256=on.identity()["sha256"],
            original_boss_distinction=baseline and baseline["boss_distinction"],
            boss_distinction=edited and edited["boss_distinction"],
            image_breakage=accident and damage(accident),
            receipts=dict(off=baseline_key, on=edited_key, damage=damage_key),
        )

    def rate_pair(keys):
        images = [inputs[k][1] for k in keys]
        votes = []
        for order in (images, images[::-1]):
            response, key = ask(order, cfg["repeat"])
            votes.append(dict(value=response and response["repeat"], receipt=key))
        values = [vote["value"] for vote in votes]
        return dict(
            cases=list(keys),
            images=[im.identity()["sha256"] for im in images],
            forward=votes[0],
            reverse=votes[1],
            repeat=None if None in values else fmean(values),
        )

    with ThreadPoolExecutor(4) as pool:
        rows = dict(zip(inputs, pool.map(rate_case, inputs), strict=True))
        comparisons = list(pool.map(rate_pair, combinations(inputs, 2)))
    per_case = {case: score_case(row) for case, row in rows.items()}
    measured = None not in per_case.values() and all(
        p["repeat"] is not None for p in comparisons
    )
    design = summarize(per_case, comparisons) if measured else None
    return dict(
        protocol=protocol_id(),
        rows=rows,
        pairs=comparisons,
        design=design,
        score=design["score"] if measured else None,
    )
