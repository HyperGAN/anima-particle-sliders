"""Write and freeze Anima's teaching prompts before the first render."""
import fcntl
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lumen_studio.contracts import atomic_json, digest
import library

DEFAULT_MODEL = "google/gemini-flash-latest"
TARGET = "Anima Turbo v1.1"
BATCH_SIZE = 3
BRIEF = """You are the prompt writer for Anima Turbo v1.1, an illustration model running at 512x512,
10 steps, CFG 1. Translate the user's visual idea into original, adventurous image prompts.
Anima understands lowercase booru tags with spaces, descriptive natural language, or a mix.
Choose the language that best conveys each image. Put the important visual evidence early.
Give the idea a concrete physical or stylistic presence. Honor its scope, count, placement and
exclusions. Depict the actual thing requested. A costume, print, logo or metaphor only makes
sense when the user asks for one. Push the effect far enough to read at thumbnail size.

The user message contains the idea and scene anchors as JSON data. For each anchor invent a
neutral starting illustration with an identifiable subject, without the requested change.
Then art-direct three genuinely different ways the idea could manifest in that same shot.
Explore scale, materials, behavior, composition or a surprising interaction with the setting.
Keep enough of the baseline subject, scene and framing to recognize the pair. Vary the visual
idea itself across candidates; mere synonyms make bad training data. Write every prompt in
full as a standalone image caption. Be concise enough for the native text encoder, roughly
25–65 words per prompt. Each candidate also needs a standalone explicit cue for training:
intensify its defining visual details while retaining that same interpretation and scene.

Return a JSON object with a scenes array. Each scene has id, neutral, and treatments.
Echo each supplied id once. Each treatments array has exactly three objects with positive
(the complete render prompt) and explicit (the complete stronger training cue) strings."""


def request_batch(writer, inputs):
    key = os.environ.get("NANOGPT_API_KEY")
    if not key:
        raise RuntimeError("NANOGPT_API_KEY is required to write new Anima training prompts")
    body = dict(model=writer["model"], temperature=1.0, max_tokens=8192,
                response_format={"type": "json_object"}, messages=[
                    {"role": "system", "content": writer["instructions"]},
                    {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)}])
    req = Request("https://api.nano-gpt.com/api/v1/chat/completions",
                  data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    req.add_unredirected_header("Authorization", "Bearer " + key)
    with urlopen(req, timeout=120) as response:
        result = json.load(response)
    choices = result.get("choices") if isinstance(result, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("The writer returned no completion")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("The writer returned no text")
    model = result.get("model")
    return model if isinstance(model, str) and model else writer["model"], json.loads(content)


def validate_rows(reply, anchors):
    rows = reply.get("scenes") if isinstance(reply, dict) else None
    if not isinstance(rows, list) or len(rows) != len(anchors):
        raise ValueError("The writer must return one scene for every supplied ID")
    by_id = {r["id"]: r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)}
    if set(by_id) != {a["id"] for a in anchors}:
        raise ValueError("The writer returned a missing, duplicate or unknown scene ID")

    def prompt(row, field):
        value = row.get(field) if isinstance(row, dict) else None
        if not isinstance(value, str) or not value.strip() or len(value) > 2400:
            raise ValueError(f"Invalid {field} prompt")
        return value.strip()

    ordered = []
    for anchor in anchors:
        row = by_id[anchor["id"]]
        neutral = prompt(row, "neutral")
        treatments = row.get("treatments")
        if not isinstance(treatments, list) or len(treatments) != 3:
            raise ValueError("Each scene needs three treatments")
        options = [dict(positive=prompt(t, "positive"), explicit=prompt(t, "explicit")) for t in treatments]
        captions = [neutral, *(t["positive"] for t in options)]
        if len({" ".join(s.casefold().split()) for s in captions}) != len(captions):
            raise ValueError("A scene repeated a render prompt")
        ordered.append(dict(id=anchor["id"], neutral=neutral, treatments=options))
    return ordered


def generate_batch(writer, inputs, ask):
    for attempt in range(3):
        try:
            model, reply = ask(writer, inputs)
            return dict(model=model, scenes=validate_rows(reply, inputs["scenes"]))
        except HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"Anima prompt writer returned HTTP {error.code}") from error
        except (URLError, TimeoutError, ValueError) as error:
            if attempt == 2:
                raise RuntimeError(f"Anima prompt writer could not complete the batch: {error}") from error
        time.sleep(2 ** attempt)


def load_spec(root, vibe, *, ask=request_batch):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "spec.json"
    with (root / "prompt-setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            saved = json.loads(path.read_bytes())
            if saved.get("vibe") != vibe:
                raise ValueError("The submitted phrase changed under this run")
            version = saved.get("phrase_version", library.WARDROBE_VERSION)
            grid = library.spec(vibe, version=version)
            if version == library.WRITER_VERSION:
                # Reuse the exact authored text, including when Studio copies a run to a new job.
                validate_rows({"scenes": saved["teaching"]}, grid["teaching"])
                for base, row in zip(grid["teaching"], saved["teaching"], strict=True):
                    base.update(neutral=row["neutral"], treatments=row["treatments"])
                grid["prompt_writer"] = saved["prompt_writer"]
                grid["teaching_sha256"] = digest(grid["teaching"])
            if saved != grid:
                raise ValueError("The saved teaching grid differs from the run specification")
            return saved

        grid = library.spec(vibe, version=library.WRITER_VERSION)
        folder = root / "prompt-rewrites"
        plan_path = folder / "writer.json"
        if plan_path.is_file():
            plan = json.loads(plan_path.read_bytes())
            if plan["phrase"] != vibe:
                raise ValueError("Saved prompt rewrites belong to another phrase")
            writer = plan["writer"]
        else:
            writer = dict(provider="NanoGPT", model=os.environ.get("ANIMA_PROMPT_MODEL") or DEFAULT_MODEL,
                          target=TARGET, instructions=BRIEF)
            atomic_json(plan_path, dict(phrase=vibe, writer=writer))
        progress = folder / "progress.json"
        if not progress.is_file():
            atomic_json(progress, dict(vibe=vibe, phrase_version=library.WRITER_VERSION,
                                       teaching=[], prompt_writer=writer))
        served = []
        for start in range(0, len(grid["teaching"]), BATCH_SIZE):
            group = grid["teaching"][start:start + BATCH_SIZE]
            anchors = [{k: scene[k] for k in ("id", "framing")} for scene in group]
            inputs = dict(idea=vibe, scenes=anchors)
            identity = digest(dict(version=library.WRITER_VERSION, writer=writer, inputs=inputs))
            receipt = folder / (identity + ".json")
            cached = receipt.is_file()
            if cached:
                batch = json.loads(receipt.read_bytes())
            else:
                batch = generate_batch(writer, inputs, ask)
                atomic_json(receipt, batch)
            for scene, written in zip(group, validate_rows(batch, anchors), strict=True):
                scene.update(written)
            served.append(batch["model"])
            if not cached:
                atomic_json(progress, dict(vibe=vibe, phrase_version=library.WRITER_VERSION,
                                           teaching=grid["teaching"][:start + len(group)],
                                           prompt_writer=dict(writer, served_models=list(dict.fromkeys(served)))))

        grid["prompt_writer"] = dict(writer, served_models=list(dict.fromkeys(served)))
        grid["teaching_sha256"] = digest(grid["teaching"])
        atomic_json(path, grid)
        return grid
