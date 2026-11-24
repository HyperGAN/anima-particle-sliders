"""Construct a bounded native rewrite pass for the strongest measured round."""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from . import vibe_curation
from .contracts import digest, file_hash
from .vibe import workflow_graphs
from .vibe_observer import IMAGE_STATE, decision_values, image_questions
from .vibe_portfolio import finish_score, observed_family

INSTRUCTION = (
    "You rewrite scene prompts to make a finished Anima image stronger and clearer. "
    "Read the JSON Context. Context.vibe is a JSON document with a target and a list of scenes. "
    "For each scene return its exact id and a positive field containing one self-contained English prompt, "
    "in a single JSON array. Preserve the named subject, viewpoint, environment and every concrete identifying "
    "prop. Each positive describes one single continuous shot. Add the specified visual treatment using "
    "concrete shapes, edges, shadows and materials. A whole-frame style must work on objects or landscapes "
    "with no human additions. For Final Boss, the imposing silhouette comes from visible costume or construction. "
    "Keep each prompt under 100 words. Avoid reference boards, insets, montage, title text, artist names, "
    "cinematic adjectives without visual consequences and unsupported facts. Do not add commentary."
)


class RewriteUnavailable(ValueError):
    """The text-model reply did not supply a usable scene rewrite."""


def rewrite_batch(profile, scenes, writer_model, judge_model, provider, call):
    from particle_sliders_vibe.nanoodle import NanoodleWorkflows

    prompts, score = workflow_graphs(writer_model, judge_model, profile.concept)
    prompts["nodes"][1]["fields"]["system"] = INSTRUCTION
    writer = NanoodleWorkflows(prompts, score, judge_provider=provider)
    request = dict(target=profile.concept, scenes=scenes)
    binding = dict(input=request, workflow=writer.identity, instructions=INSTRUCTION)
    proposed = call("rewrite", binding,
                    lambda key: writer.propose(json.dumps(request), len(scenes), [], request_id=key))
    if not isinstance(proposed, list) or len(proposed) != len(scenes):
        raise RewriteUnavailable("The rewriting model did not return every specified teacher scene")
    rewritten = {row["id"]: row["positive"].strip() for row in proposed
                 if isinstance(row, dict) and set(row) == {"id", "positive"} and isinstance(row["positive"], str)
                 and 15 <= len(row["positive"].split()) <= 120}
    if rewritten.keys() != {row["id"] for row in scenes}:
        raise RewriteUnavailable("Teacher rewrite scene names or prompt lengths changed")
    return rewritten


def augment_round(round_, picked, profile, product, root, call, *, writer_model, judge_model, provider):
    families = tuple(profile.families)
    selected = {row["case"]: row for row in picked["selected"]}
    references = {}
    scenes = []
    for i, (case, group) in enumerate(round_["alternatives"].items()):
        reference = selected.get(case) or min(group, key=lambda row: row["seed"])
        label = f"scene-{i+1}"
        cue = profile.families[families[i % len(families)]]
        references[label] = reference
        scenes.append(dict(id=label, neutral=reference["neutral"]["output"]["prompt"], treatment=cue))
    rewrites = rewrite_batch(profile, scenes, writer_model, judge_model, provider, call)
    pending = []
    for scene in scenes:
        label, reference = scene["id"], references[scene["id"]]
        variants = {
            "cue-first": f"{profile.concept}. {scene['treatment'].capitalize()}. {scene['neutral']}",
            "rewrite": rewrites[label],
            "rewrite-named": f"{profile.concept}. {rewrites[label]}",
        }
        for mode, prompt in variants.items():
            seed = reference["seed"]
            output = call("render", dict(prompt=prompt, seed=seed),
                          lambda key: product.render(prompt, seed, root / "outputs" / key, request_id=key))
            pending.append((f"{mode}:{seed}", output, reference))

    questions = image_questions(profile, candidate=True)
    locks = {}

    def measure(request):
        method, output, reference = request
        path = output["path"]
        if file_hash(path) != output["sha256"]:
            raise ValueError("Rewritten teacher image changed")
        binding = dict(provider=provider, model=judge_model, state=IMAGE_STATE, questions=questions, images=[output["sha256"]])
        with locks.setdefault(output["sha256"], Lock()):
            receipt = call("judge", binding, lambda _: vibe_curation.observe(
                provider, judge_model, (path,), IMAGE_STATE, questions,
            ))
        values = decision_values(receipt["raw"], questions)
        if values != receipt["values"]:
            raise ValueError("Saved rewrite readings differ from the typed response")
        neutral = reference["neutral"]
        before = neutral["readings"]
        severity = max(before["catastrophic"], values["catastrophic"])
        integrity = "pass" if severity <= .20 else "fail" if severity >= .80 else "unmeasured"
        scores = values["family"] if profile.family_choice else {
            name: values[f"family_{name}"] for name in profile.families
        }
        family = observed_family(scores, minimum=profile.family_minimum)
        gain = finish_score(values["effect"], values["overbake"], values["layout"]) - finish_score(
            before["effect"], before["overbake"], before["layout"],
        )
        positive = dict(output=output, judgment={"score": values["effect"]}, readings=values,
                        decision=digest(dict(kind="judge", inputs=binding)))
        return dict(case=reference["case"], method=method, seed=reference["seed"],
                    observed_family=family if family in profile.families else None,
                    integrity=integrity, usable=values["usable"], finish_gain=gain, neutral=neutral, positive=positive)

    groups = {case: list(rows) for case, rows in round_["alternatives"].items()}
    with ThreadPoolExecutor(5) as pool:
        for row in pool.map(measure, pending):
            groups[row["case"]].append(row)
    return round_ | {"alternatives": groups}
