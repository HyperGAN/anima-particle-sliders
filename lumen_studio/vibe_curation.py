"""Collect, judge, and select native teacher portfolios for named vibes."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path
from threading import Lock

from .contracts import atomic_json, digest, file_hash
from .vibe import training_manifest
from .vibe_observer import IMAGE_STATE, PAIR_STATE, decision_values, image_questions, observe
from .vibe_portfolio import (
    FAMILY_LIMIT, distinct_finish, exclude_duplicates, finish_score, observed_family, pair_audit, select,
)
from .vibe_profiles import FINAL_BOSS, yes_no
from .vibe_target_search import read


def prompts(profile, proposal):
    neutral = proposal["neutral"].rstrip(". ")
    lead = ("In-world anime scene of a formidable final RPG boss." if profile is FINAL_BOSS
            else "Full-bleed frame from 1990s Japanese hand-drawn cel animation.")
    candidates = {"proposed": proposal["positive"]}
    if profile.family_choice:
        candidates["literal"] = f"{neutral}. Hand-painted 1990s Japanese cel animation."
    return candidates | {
        family: f"{neutral}. {lead} {cue.capitalize()}. A continuous shot with visible surroundings."
        for family, cue in profile.families.items()
    }


def curate_round(product, profile, proposals, index, args, root, call):
    rendered = []
    for proposal in proposals:
        case = digest(proposal["neutral"])
        candidates = prompts(profile, proposal)
        for take in range(args.best_of):
            seed = args.seed + index * args.best_of + take
            outputs = {}
            for method, prompt in {"off": proposal["neutral"], **candidates}.items():
                outputs[method] = call(
                    "render", dict(prompt=prompt, seed=seed),
                    lambda key: product.render(prompt, seed, root / "outputs" / key, request_id=key),
                )
            rendered.append((case, seed, outputs))

    request_locks = {}
    off_questions = image_questions(profile, candidate=False)
    on_questions = image_questions(profile, candidate=True)

    def measure(paths, state, questions):
        binding = dict(provider=args.judge_provider, model=args.judge_model, state=state, questions=questions,
                       images=[file_hash(path) for path in paths])
        key = digest(dict(kind="judge", inputs=binding))
        with request_locks.setdefault(key, Lock()):
            receipt = call("judge", binding,
                           lambda _: observe(args.judge_provider, args.judge_model, paths, state, questions))
        values = decision_values(receipt["raw"], questions)
        if values != receipt["values"]:
            raise ValueError("Stored Jev readings differ from the provider response")
        return values, key

    def measure_case(task):
        case, seed, outputs = task
        off = outputs["off"]
        neutral, off_receipt = measure((off["path"],), IMAGE_STATE, off_questions)
        before = finish_score(neutral["effect"], neutral["overbake"], neutral["layout"])
        rows = []
        for method, output in outputs.items():
            if method == "off":
                continue
            values, decision = measure((output["path"],), IMAGE_STATE, on_questions)
            on = finish_score(values["effect"], values["overbake"], values["layout"])
            severity = max(neutral["catastrophic"], values["catastrophic"])
            integrity = "pass" if severity <= .20 else "fail" if severity >= .80 else "unmeasured"
            family_scores = values["family"] if profile.family_choice else {
                name: values[f"family_{name}"] for name in profile.families
            }
            family = observed_family(family_scores, minimum=profile.family_minimum)
            if family not in profile.families:
                family = None
            rows.append(dict(case=case, method=f"{method}:{seed}", seed=seed, observed_family=family,
                             integrity=integrity, usable=values["usable"], finish_gain=on - before,
                             neutral=dict(output=off, judgment={"score": neutral["effect"]},
                                          readings=neutral, decision=off_receipt),
                             positive=dict(output=output, judgment={"score": values["effect"]},
                                           readings=values, decision=decision)))
        return rows

    def compare(left, right):
        images = tuple(row["positive"]["output"]["path"] for row in (left, right))
        values, decision = measure(images, PAIR_STATE, {"near": yes_no(profile.lookalike)})
        return dict(near=values["near"], decision=decision)

    by_case = {digest(proposal["neutral"]): [] for proposal in proposals}
    with ThreadPoolExecutor(max_workers=6) as workers:
        for rows in workers.map(measure_case, rendered):
            by_case[rows[0]["case"]].extend(rows)
        groups = list(by_case.values())
        chosen = select(groups, FAMILY_LIMIT, allow_partial=True)
        chosen, proof, repairs = exclude_duplicates(
            groups, chosen, lambda rows: pair_audit(rows, compare, workers),
            limit=FAMILY_LIMIT, allow_partial=True,
        )
    metric = distinct_finish(chosen, proof, panel_size=len(proposals))
    credited = set(metric["credited"])
    selected = {row["case"]: row for row in chosen}
    ranking = []
    for proposal in proposals:
        case = digest(proposal["neutral"])
        row = selected.get(case)
        fitness = row["finish_gain"] if row and (case in credited or row["finish_gain"] < 0) else 0
        ranking.append(dict(prompt=proposal, case=case, fitness=fitness))
    ranking.sort(key=lambda row: -row["fitness"])
    return dict(index=index, ranking=ranking, selected=chosen, alternatives=by_case, pair_audit=proof,
                repairs=repairs, distinct_finish=metric)


def collect_portfolio(product, profile, args):
    from particle_sliders_vibe import SCHEMA, _call, collection_lock, file_digest, prompt_pair

    if not 1 <= args.pool_size <= len(profile.families) * FAMILY_LIMIT:
        raise ValueError("Pool size exceeds the capacity of the visible design families")
    if (not 1 <= args.replace <= args.pool_size or args.best_of < 1 or args.rounds < 1
            or not 0 <= args.seed < 2**63 - args.rounds * args.best_of):
        raise ValueError("Invalid replacement, rounds, best-of or seed")
    root = Path(args.out).resolve()
    identity = dict(schema=SCHEMA, vibe=profile.concept, product=product.identity, profile=asdict(profile),
                    metric="distinct-finish-gain", selection="raw", pool_size=args.pool_size, replace=args.replace,
                    best_of=args.best_of, seed=args.seed, judge=args.judge_model, provider=args.judge_provider,
                    engine={name: file_digest(Path(__file__).with_name(name)) for name in
                            ("vibe_curation.py", "vibe_portfolio.py", "vibe_profiles.py", "vibe_observer.py")})
    path = root / "portfolio.json"
    with collection_lock(root):
        if path.exists():
            state = read(path)
            if digest(state["identity"]) != digest(identity):
                raise ValueError("Portfolio settings or source changed. Use a new output directory.")
        else:
            if any(p.name != ".lock" for p in root.iterdir()):
                raise ValueError("A new portfolio requires an empty output directory")
            state = dict(identity=identity, rounds=[])
            atomic_json(path, state)
        call = partial(_call, root, retry_pending=args.retry_pending)
        for index in range(len(state["rounds"]), args.rounds):
            history = [{"index": row["index"], "ranking": row["ranking"]} for row in state["rounds"]]
            retained = history[-1]["ranking"][:args.pool_size - args.replace] if history else []
            scenes = [row["prompt"] for row in retained]
            slots = dict(profile.subject_slots) or {"scene": ""}
            kinds = list(slots)
            remaining = Counter(kinds[i % len(kinds)] for i in range(args.pool_size))
            remaining.subtract(scene["kind"] for scene in scenes)
            for kind, instruction in slots.items():
                count = remaining[kind]
                if not count:
                    continue
                scene_request = f"{profile.concept}. {profile.scene_instruction} {instruction}".strip()
                proposals = call(
                    "propose", dict(round=index, kind=kind, count=count, history=history, criterion=scene_request),
                    lambda key: product.propose(scene_request, count, history, request_id=key),
                )
                if not isinstance(proposals, list) or len(proposals) != count:
                    raise ValueError("Proposer did not return the requested number of scenes")
                scenes.extend(prompt_pair(pair) | {"kind": kind} for pair in proposals)
            if len({pair["neutral"] for pair in scenes}) != len(scenes):
                raise ValueError("The proposer returned duplicate neutral scenes")
            result = curate_round(product, profile, scenes, index, args, root, call)
            state["rounds"].append(result)
            atomic_json(path, state)
    return state


def portfolio_manifest(state, min_score):
    """Pass the latest audited selection to the existing native target format."""
    latest = state["rounds"][-1]
    credited = set(latest["distinct_finish"]["credited"])
    ranking = []
    for row in latest["selected"]:
        if row["case"] not in credited:
            continue
        neutral, positive = row["neutral"], row["positive"]
        case = dict(seed=row["seed"], neutral=neutral, positive=positive,
                    gain=positive["judgment"]["score"] - neutral["judgment"]["score"])
        ranking.append(dict(prompt=dict(neutral=neutral["output"]["prompt"], positive=positive["output"]["prompt"]),
                            fitness=row["finish_gain"], cases=[case]))
    evidence = digest(latest)
    manifest = training_manifest(
        dict(identity=state["identity"], rounds=[dict(index=latest["index"], ranking=ranking)],
             portfolio_round_sha256=evidence),
        min_score=min_score,
    )
    manifest.pop("sha256")
    manifest["portfolio_round_sha256"] = evidence
    manifest["sha256"] = digest(manifest)
    return manifest
