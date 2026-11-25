"""Bounded corpus proposal and Jev evidence receipts for character-only Final Boss."""

import base64
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
ALIAs = "google/gemini-flash-latest"
JEV = "typesafe/jev-1.14"
BASE_URL = "https://api.nano-gpt.com/api/v1"
SCENES = [
    ("greenhouse", "glass greenhouse, rows of potted ferns and a pale stone path", "soft midday sunlight through the glass", "adult woman with short auburn hair"),
    ("harbor", "seaside walkway beside a blue moored boat and stone bollards", "clear cool daylight", "adult man with shaved head and brown skin"),
    ("cafe", "small cafe with a green tiled counter and two wooden stools", "warm diffuse window light", "adult woman with shoulder-length black hair"),
    ("bridge", "low stone bridge beside yellow reed beds and a slow river", "flat bright overcast light", "adult man with wavy silver hair"),
    ("library", "quiet wood-paneled library with tall bookshelves and a green reading lamp", "warm steady indoor lamp light", "adult woman with black curls and brown skin"),
    ("train", "empty railway platform with a blue enamel station clock and metal bench", "cool overhead fluorescent lighting at night", "adult man with short black hair"),
    ("workshop", "tidy ceramics workshop with shelves of unpainted bowls and a pottery wheel", "neutral diffuse studio lighting", "adult woman with a long dark braid"),
    ("garden", "walled garden with a round stone fountain and white flowers", "clear blue moonlight", "adult man with shoulder-length brown hair"),
    ("courtyard", "red-brick courtyard with a cream arched doorway and two terracotta pots", "soft late-afternoon sunlight", "adult woman with close-cropped silver hair"),
    ("mall", "modern atrium with a glass balustrade and pale marble floor", "even skylight on a cloudy day", "adult man with tightly curled dark hair"),
    ("aquarium", "aquarium corridor with a large blue tank and a short gray bench", "muted blue aquarium light", "adult woman with a short dark bob"),
    ("museum", "museum gallery with a rectangular limestone sculpture and dark blue wall", "neutral white museum track lighting", "adult man with a black ponytail"),
]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as writer:
        json.dump(value, writer, indent=2, allow_nan=False)
        writer.write("\n")
        writer.flush()
        os.fsync(writer.fileno())
    temporary.replace(path)


def api(tag, endpoint, payload, inputs):
    receipt = ROOT / "calls" / f"{tag}.json"
    pending = receipt.with_suffix(".pending")
    if receipt.exists():
        recorded = json.loads(receipt.read_text())
        if recorded["inputs"] != inputs:
            raise RuntimeError(f"Saved request differs for {tag}")
        if recorded["status"] != 200:
            raise RuntimeError(f"Prior {tag} HTTP {recorded['status']}")
        return recorded["response"]
    if pending.exists():
        raise RuntimeError(f"Unresolved paid request for {tag}")
    key = os.environ.get("NANOGPT_API_KEY")
    if not key:
        raise RuntimeError("NanoGPT key unavailable")
    save(pending, inputs)
    request = urllib.request.Request(
        BASE_URL + endpoint,
        json.dumps(payload).encode(),
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=100) as response:
            status, body = response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = {"error": "provider returned HTTP error"}
    save(receipt, {"inputs": inputs, "status": status, "response": body})
    pending.unlink()
    if status != 200:
        raise RuntimeError(f"NanoGPT {tag}: HTTP {status}")
    return body


def propose():
    contacts = {name: person for name, _, _, person in SCENES}
    instruction = (
        "Write exactly 12 pairs of SHORT anime character costume descriptions in a JSON object "
        "with a rows array. Each row has id, ordinary, boss. Preserve the same person's exact hair, "
        "face, identity and body type. The ordinary and boss values describe only outfit, worn "
        "accessories, and expression. Ordinary is everyday practical clothing. Boss must make the "
        "same solitary person read unmistakably as a formidable final video-game antagonist "
        "through commanding tailored armor, an iconic exaggerated silhouette, regal details and "
        "a stern expression. Vary costume designs across subjects. No scenery, camera, lighting, "
        "time of day, magic effects, glow, smoke, environmental effects, crowds, battles, "
        "or pose in either value. Avoid references to existing characters. "
        "Keep each value under 32 words. Use these ids and identities: " + json.dumps(contacts)
    )
    payload = dict(model=ALIAs, temperature=0.45, max_tokens=5000,
                   response_format={"type": "json_object"},
                   messages=[{"role": "user", "content": instruction}])
    body = api("gemini-proposal-v1", "/chat/completions", payload, payload)
    result = json.loads(body["choices"][0]["message"]["content"])
    tags = {item["id"]: item for item in result["rows"]}
    if set(tags) != set(contacts) or len(result["rows"]) != len(contacts):
        raise ValueError("Gemini did not return each exact scene ID once")
    rows = []
    for index, (name, background, lighting, person) in enumerate(SCENES):
        ordinary, boss = (tags[name][k].strip() for k in ("ordinary", "boss"))
        if ordinary == boss or not (2 <= len(ordinary.split()) <= 40 and 2 <= len(boss.split()) <= 40):
            raise ValueError(f"Invalid costume clauses for {name}")
        prefix = f"Anime illustration. Waist-up eye-level portrait of one {person}, standing still facing the viewer, arms relaxed."
        suffix = f" Background: {background}. Lighting: {lighting}. Clear view of the setting around the figure."
        rows.append(dict(id=name, seed=552010 + index, background=background, lighting=lighting,
                         person=person, ordinary=ordinary, boss=boss,
                         neutral=f"{prefix} {ordinary}{suffix}", positive=f"{prefix} {boss}{suffix}"))
    save(ROOT / "design.json", dict(rows=rows, requested_model=ALIAs, served_model=body.get("model"),
                                    system_fingerprint=body.get("system_fingerprint"),
                                    judge=JEV, resolution=512, sampler_steps=10))
    print(f"Proposed {len(rows)} paired costumes. Model receipt: {body.get('model')}")


def measure(tag, image_paths, state, questions):
    images = []
    inputs = {"model": JEV, "state": state, "questions": questions, "images": []}
    for path in image_paths:
        raw = Path(path).read_bytes()
        inputs["images"].append(hashlib.sha256(raw).hexdigest())
        images.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(raw).decode("ascii")}})
    parts = [state]
    for index, image in enumerate(images, 1):
        parts.extend((f"image_{index}:", image))
    body = api(tag, "/decisions", dict(model=JEV, state=parts, questions=questions), inputs)
    answers = body["answers"]
    if set(answers) != set(questions) or any(
        not isinstance(value, dict) or value.get("type") != questions[key]["type"]
        for key, value in answers.items()
    ):
        raise RuntimeError("Jev answer keys or types differ from the frozen questions")
    return answers


def canary():
    folder = Path.home() / ".local/share/particle-sliders-runs/anima-final-boss-RAW192-native-schedule-heldout-v3/outputs/case-3"
    facts = {
        "reference": "Is a narrow bright shaft of daylight clearly visible across the tiled floor in image_1?",
        "candidate": "Is a narrow bright shaft of daylight clearly visible across the tiled floor in image_2?",
        "comparison": "Did the narrow daylight stripe on the floor disappear from image_1 to image_2?",
    }
    answers = measure("jev-temple-canary-v1", [folder / s / "image.png" for s in ("off", "early")],
                      "Answer only from the two supplied anime images in labeled order.",
                      {key: {"type": "noul", "instructions": value} for key, value in facts.items()})
    print(json.dumps(answers, indent=2))


if __name__ == "__main__":
    import sys
    {"propose": propose, "canary": canary}[sys.argv[1]]()
