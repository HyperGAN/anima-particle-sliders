"""Bounded teacher replacement and validation-panel construction."""

from concurrent.futures import ThreadPoolExecutor

from particle_sliders.candidate_failure import FileRecord

from ..contracts import atomic_json, digest
from ..vibe import SCHEMA, validate_manifest
from ..vibe_profiles import profile_for
from .pair_checks import PAIR_PROTOCOLS
from .outputs import Unmeasured, read


def replacement_scene(scene, concept, index, trial):
    """After five unsuccessful seeds, change the framing on both sides of the pair."""
    if trial < 5:
        return scene
    framing = "Single full-bleed cinematic shot from one camera. The adult subject occupies the foreground in a continuous setting."
    neutral = scene["neutral"].replace(
        "head and shoulders filled by the subject",
        "medium shot with surroundings visible",
    )
    neutral = f"{framing} {neutral}"
    profile = profile_for(concept)
    if profile is not None:
        cues = list(profile.families.values())
        cue = cues[(index + trial) % len(cues)]
        named = (
            f"{concept} screencap"
            if profile.family_choice
            else f"{concept} character reveal"
        )
        treatments = [f"{named}. {neutral} {cue}.", f"{concept}. {cue}. {neutral}"]
    else:
        treatments = [f"{framing} {prompt}" for prompt in scene["treatments"]]
    return dict(neutral=neutral, treatments=treatments, seed=scene["seed"])


def collect(outputs, config, concept, minimum, attempts):
    root = outputs.root
    identity = dict(
        vibe=concept,
        model=outputs.runtime.identity,
        teacher_config=digest(config),
        effect=outputs.rubric,
        max_neutral_score=config["neutral_ceiling"],
        selection="raw",
        adult_subjects="prompt-only",
        min_score=0.75,
        required=minimum,
    )
    protocol = PAIR_PROTOCOLS.get(concept.casefold())
    if protocol:
        identity["pair_check"] = protocol
    variation = "vibe-" + digest(identity)[:16]
    rows = []
    used = set()
    trials = []
    for trial in range(attempts):
        for index, scene in enumerate(config["teachers"]):
            if index in used:
                continue
            scene = replacement_scene(scene, concept, index, trial)
            seed = scene["seed"] + 104729 * trial
            neutral = outputs.score(outputs.image(scene["neutral"], seed))
            evidence = dict(
                scene=index,
                attempt=trial + 1,
                seed=seed,
                neutral=neutral,
                treatments=[],
            )
            trials.append(evidence)
            if (
                neutral["integrity"]["status"] == "pass"
                and neutral["judgment"]["score"] is not None
                and neutral["judgment"]["score"] <= config["neutral_ceiling"]
            ):
                with ThreadPoolExecutor(3) as pool:
                    candidates = []
                    for prompt in scene["treatments"]:
                        image = outputs.image(prompt, seed)
                        candidates.append((prompt, pool.submit(outputs.score, image)))
                    for prompt, job in candidates:
                        positive = job.result()
                        score = positive["judgment"]["score"]
                        gain = (
                            score - neutral["judgment"]["score"]
                            if score is not None
                            else None
                        )
                        eligible = (
                            positive["integrity"]["status"] == "pass"
                            and score is not None
                            and score >= 0.75
                            and gain > 0
                        )
                        pair = {}
                        if eligible and protocol:
                            check = outputs.teacher_pair(neutral, positive)
                            pair["pair_check"] = check
                            eligible = check["status"] == "pass"
                        evidence["treatments"].append(
                            dict(
                                prompt=prompt,
                                positive=positive,
                                gain=gain,
                                eligible=eligible,
                                **pair,
                            )
                        )
                qualified = [t for t in evidence["treatments"] if t["eligible"]]
                if qualified:
                    best = max(qualified, key=lambda t: t["gain"])
                    prompt = dict(neutral=scene["neutral"], positive=best["prompt"])
                    key = digest(dict(round=trial, prompt=prompt))
                    selection = dict(
                        id=key,
                        round=trial,
                        prompt=prompt,
                        seed=seed,
                        gain=best["gain"],
                        neutral=neutral,
                        positive=best["positive"],
                    )
                    if protocol:
                        selection["pair_check"] = best["pair_check"]
                    rows.append(
                        dict(
                            id="vibe-" + key,
                            variation=variation,
                            split="train",
                            **prompt,
                            seeds=[seed],
                            selection=selection,
                        )
                    )
                    used.add(index)
            atomic_json(root / "teacher-decisions.json", trials)
        if len(rows) >= minimum:
            break
    if len(rows) < minimum:
        raise Unmeasured(
            f"{len(rows)} of {minimum} required distinct scenes qualified in {attempts} attempts"
        )
    manifest = dict(
        schema=1,
        extension=SCHEMA,
        split="train",
        variation=variation,
        collection_identity=identity,
        collection_sha256=digest(trials),
        criterion=concept,
        min_score=0.75,
        rows=rows,
    )
    manifest["sha256"] = digest(manifest)
    validate_manifest(manifest)
    atomic_json(root / "manifest.json", manifest)
    return manifest


def validation_panels(outputs, groups, attempts):
    path = outputs.root / "validation.json"
    if path.exists():
        panels = read(path)
        if set(panels) != set(groups):
            raise ValueError("The frozen validation groups differ from this study")
        for name, cases in panels.items():
            proposals = {scene["prompt"]: scene["seed"] for scene in groups[name]}
            seen = set()
            if len(cases) < 2:
                raise ValueError(f"Incomplete frozen {name} panel")
            for case in cases:
                prompt, seed = case["prompt"], case["seed"]
                if prompt not in proposals or prompt in seen:
                    raise ValueError(
                        f"Changed or duplicate prompt in frozen {name} panel"
                    )
                offset = seed - proposals[prompt]
                if (
                    offset < 0
                    or offset % 104729
                    or case["off"]["integrity"]["status"] != "pass"
                ):
                    raise ValueError(f"Invalid matched case in frozen {name} panel")
                seen.add(prompt)
                record = FileRecord(**case["off"]["output"])
                if FileRecord.from_path(record.path) != record:
                    raise ValueError(f"Changed OFF image in frozen {name} panel")
        return panels
    panels = {}
    for group, proposed in groups.items():
        accepted = []
        for scene in proposed:
            for trial in range(attempts):
                seed = scene["seed"] + trial * 104729
                record = outputs.image(scene["prompt"], seed)
                off = outputs.score(record)
                if (
                    off["integrity"]["status"] == "pass"
                    and off["judgment"]["score"] is not None
                ):
                    accepted.append(dict(prompt=scene["prompt"], seed=seed, off=off))
                    break
        if len(accepted) < 2:
            raise Unmeasured(f"{group} has fewer than two coherent validation scenes")
        panels[group] = accepted
    atomic_json(path, panels)
    return panels
