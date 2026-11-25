"""Typed image decisions and finish rubrics for the named vibe portfolios."""
import base64
import json
import math
import os
from pathlib import Path

from .vibe_calls import InvalidReply, NotDispatched, ProviderUnavailable
from .vibe_profiles import score_question, yes_no

FINISH = {
    "overbake": score_question(
        "How much does this image look visibly overprocessed? Judge clipped featureless highlights, crushed tonal "
        "detail, excessive sharpening halos or crunchy edges, and harsh coalesced textures or artificial "
        "microcontrast. Limited cel palettes, strong intentional shadows, and ink outlines can all be clean.",
        (
            "Clean tonal and material separation. Edges and texture look resolved within the chosen style.",
            "Slight crunchy edges, tonal clipping or texture harshness in localized areas.",
            "Clear overprocessing across major areas. Harsh edges or loss of useful tonal or material separation.",
            "Pervasive clipping, halos or compressed textures make most of the image visibly overprocessed.",
        ),
    ),
    "layout": score_question(
        "Rate unintended layout defects in this finished frame, including bordered insets, disconnected duplicate "
        "views, reference sheets, or split panels. An environment, object, animal or figure can be the subject of "
        "one coherent shot. Natural scenery, grouped subjects and plausible reflections are part of the same shot.",
        (
            "One continuous, coherent shot.",
            "One recognizable shot with a minor localized continuity ambiguity.",
            "A visible disconnected view, inset or layout break disrupts an otherwise readable scene.",
            "A montage, disconnected views or divided layout dominates the frame.",
        ),
    ),
    "catastrophic": yes_no(
        "Is the finished image visually unusable because almost all of it is blank, digitally corrupted, "
        "severely smeared, or incoherent noise? Intentional stylization, blur or grain in a readable shot can pass."
    ),
}
SUITABILITY = (
    "Little recognizable effect or major visible defects.",
    "Weak effect or significantly compromised finish.",
    "Clear effect with a coherent finish.",
    "Unmistakable effect in a clean, integrated shot.",
)
IMAGE_STATE = "Assess the finished image from the visible pixels."
PAIR_STATE = "Compare two unlabeled images for near-duplicate treatments of the requested vibe."


def image_questions(profile, *, candidate):
    questions = {"effect": profile.effect, **FINISH}
    if candidate:
        questions["usable"] = score_question(profile.suitability, SUITABILITY)
        if profile.family_choice:
            questions["family"] = dict(
                type="choice",
                instructions="Which treatment dominates the graphic style of this entire image? Compare the defining "
                             "line, shadow and painted background features among the listed choices. Use unclassified "
                             "when the treatment is mixed or too generic to distinguish. Judge by style across the frame.",
                criteria=profile.families | {"unclassified": "No clear match to a single listed treatment."},
            )
        else:
            for key, cue in profile.families.items():
                questions[f"family_{key}"] = yes_no(f"Does this treatment visibly use {cue} as a defining feature?")
    return questions


def checked_number(value, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError("Invalid Jev number")
    return value


def decision_values(response, questions):
    answers = response["answers"]
    if set(answers) != set(questions):
        raise ValueError("Jev answer names differ from the requested questions")
    result = {}
    for name, question in questions.items():
        answer = answers[name]
        if question["type"] != answer["type"]:
            raise ValueError("Jev answer type differs from the requested question")
        if question["type"] == "noul":
            result[name] = checked_number(answer["noul"], 1)
            continue
        if question["type"] == "choice":
            probabilities = {key: checked_number(p, 1) for key, p in answer["probabilities"].items()}
            if (probabilities.keys() != question["criteria"].keys()
                    or answer["choice"] not in probabilities
                    or abs(sum(probabilities.values()) - 1) > len(probabilities) * .005 + 1e-9):
                raise ValueError("Jev classification differs from the requested choices")
            result[name] = probabilities
            continue
        score = checked_number(answer["score"], 3)
        distribution = answer["probabilities"]
        if set(distribution) != {"0", "1", "2", "3"}:
            raise ValueError("Jev score distribution differs from the four levels")
        weights = [checked_number(distribution[str(i)], 1) for i in range(4)]
        if abs(sum(weights) - 1) > .020000001 or abs(sum(i*p for i, p in enumerate(weights)) - score) > .035000001:
            raise ValueError("Jev score distribution exceeds NanoGPT rounding bounds")
        result[name] = score / 3
    return result


ENDPOINTS = {
    "nanogpt": ("https://api.nano-gpt.com/api/v1/decisions", "NANOGPT_API_KEY"),
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY"),
}


def judge_endpoint(provider):
    """Resolve the decision endpoint and credential before anything is sent."""
    url, variable = ENDPOINTS[provider]
    key = os.environ.get(variable)
    if not key:
        raise NotDispatched(f"{variable} is required for the {provider} judge")
    return url, {"Content-Type": "application/json", "Authorization": "Bearer " + key}


def observe(provider, model, images, state, questions):
    from nanoodle.transport import default_http
    from particle_sliders_vibe import ProviderRejected

    url, headers = judge_endpoint(provider)
    parts = [state]
    for i, image in enumerate(images, 1):
        if len(images) > 1:
            parts.append(f"image_{i}:")
        encoded = base64.b64encode(Path(image).read_bytes()).decode("ascii")
        parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
    payload = dict(model=model, state=parts, questions=questions)
    if provider == "typesafe":
        from particle_sliders_vibe.nanoodle import typesafe_payload

        headers["TypeSafe-Danger-Zone-Images"] = "true"
        payload = typesafe_payload(payload)
    response = default_http("POST", url, headers=headers, body=json.dumps(payload).encode(), timeout=75)
    if 400 <= response.status < 500:
        raise ProviderRejected("TypeSafe" if provider == "typesafe" else "NanoGPT", response.status)
    if response.status != 200:
        raise ProviderUnavailable("Jev", response.status)
    text = response.text()
    try:
        raw = json.loads(text)
        return dict(values=decision_values(raw, questions), raw=raw)
    except (ValueError, KeyError, TypeError) as error:
        reason = str(error) if isinstance(error, ValueError) else f"Malformed Jev reply: {error!r}"
        raise InvalidReply(reason, reply=text) from error
