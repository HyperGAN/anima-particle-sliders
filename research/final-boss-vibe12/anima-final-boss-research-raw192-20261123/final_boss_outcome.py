"""Qualify a Jev outcome question and select strength on the saved Prompt-only RAW192."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
import json
import sys
from time import perf_counter, time_ns

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "final-boss-outcome-jev"
ALPHA = ROOT / "alpha-calibration-v1"
COMMAND = ROOT / "prompt-only-alpha-tune"
BASELINE = ("constant", .4)
SCREEN = ("ranger", "smith", "watchmaker", "bookkeeper")
STRENGTHS = {
    "early": (.5, 2 / 3, .8, 1., 1.2),
    "constant": (.4, .6083333333333334, .85, 1.1),
}
PROMPTS = {
    "direct-final-boss": (
        "How well does the candidate convey ‘Final Boss’ without breaking the image?",
        (
            "No convincing Final Boss effect, or the image is visibly broken.",
            "A weak suggestion of Final Boss in a coherent image.",
            "Clearly conveys Final Boss in a coherent, intact image.",
            "Unmistakably conveys Final Boss in a coherent, intact image.",
        ),
    ),
    "original-final-boss": (
        "How well does the original character now read as ‘Final Boss’ without breaking the image?",
        (
            "Ordinary portrait, unrelated character, or visibly broken image.",
            "A hint of Final Boss. A change in mood alone belongs here.",
            "The original character clearly reads as Final Boss in an intact image.",
            "Unmistakable Final Boss transformation of the original character in a coherent, intact image.",
        ),
    ),
    "final-boss-transformation": (
        "How convincingly does the candidate turn the original character into ‘Final Boss’ in one intact image?",
        (
            "No convincing transformation, a visibly broken render, or a different character.",
            "Subtle styling that leaves the character looking ordinary.",
            "Convincing Final Boss transformation with a recognizable character and intact illustration.",
            "An unmistakable Final Boss reveal of the original character in a coherent, intact image.",
        ),
    ),
    "direct-character-scale": (
        "How well does the candidate convey ‘Final Boss’ without breaking the image?",
        (
            "No Final Boss character design. A plain portrait, a replaced subject or a broken frame.",
            "Weak Final Boss character design. The changes are mainly mood, face, hair or everyday clothing.",
            "Clear Final Boss character design that visibly transforms the original character in an intact image.",
            "An unmistakable and distinctive Final Boss reveal of the original character in a coherent image.",
        ),
    ),
    "direct-context": (
        "How well does the candidate convey ‘Final Boss’ without breaking the image?",
        (
            "Ordinary portrait, broken image, or an unrelated replacement.",
            "Some characterization, but the figure mainly looks like an ordinary person.",
            "The recognizable original character clearly reads as Final Boss in an intact image.",
            "Unmistakable Final Boss reveal of the original character in a coherent, intact image.",
        ),
    ),
    "clear-character-edit": (
        "How clearly does the edited character convey ‘Final Boss’ in an intact image? A mood change "
        "alone or a replacement character is a weak edit.",
        (
            "Ordinary or unchanged character, broken illustration, or unrelated replacement.",
            "Changes mainly expression, styling or lighting. Still reads as an ordinary person.",
            "Clearly reads as Final Boss through a visible character change in an intact image.",
            "An unmistakable, distinctive Final Boss reveal of the original character in one coherent illustration.",
        ),
    ),
    "boss-read": (
        "How well does the candidate convey ‘Final Boss’ without breaking the image? "
        "An everyday portrait with harsher hair, expression or shadows is weak. A strong result "
        "visibly redesigns the recognizable original character.",
        (
            "Ordinary clothing without a distinctive design change, broken image, or replaced subject.",
            "A small change hints at a boss, but the figure still looks ordinary or generic.",
            "The original character clearly reads as Final Boss through a visible design change.",
            "Unmistakable Final Boss reveal of the original character in a coherent, intact image.",
        ),
    ),
    "transformation": (
        "How convincingly did image_2 turn the character in image_1 into a distinctive final boss "
        "in one coherent illustration? Look for readable costume, silhouette and a focal point on "
        "the character. A stern face, darker lighting or more ornament alone is weak evidence. "
        "A broken picture or a generic replacement for the character scores zero.",
        (
            "No credible final-boss transformation, a broken picture, or an unrelated replacement.",
            "Some boss-like styling, but still ordinary, generic, or hard to read.",
            "Clear, coherent final-boss design with readable character-specific changes.",
            "Distinctive endgame boss with a strong, unified, readable character design.",
        ),
    ),
    "art-direction": (
        "Would image_2 work as the final-boss reveal of image_1's character? Judge the legible, "
        "intentional design of the figure in the finished scene. Reward a signature costume or "
        "silhouette that carries the transformation. Count darkness, glare and featureless armor "
        "only if they help the character read. A visibly broken image scores zero.",
        (
            "The character is ordinary, the image is broken, or the design has lost the subject.",
            "An accessory, mood or generic costume suggests a boss without a convincing reveal.",
            "A readable, clearly transformed endgame boss recognizable as the original character.",
            "An unmistakable, memorable boss reveal with a cohesive design and intact illustration.",
        ),
    ),
    "boss-reveal": (
        "Does image_2 make the original character read as the game's final boss in one intact "
        "illustration? Reward a distinctive, readable costume or silhouette that fits this character. "
        "A darker or sterner portrait is weak. A muddled figure, duplicated scene or unrelated "
        "replacement scores zero.",
        (
            "No successful transformation. Ordinary portrait, visibly broken output, or a lost subject.",
            "Minor styling or a generic costume. Still lacks a readable, distinctive boss design.",
            "Clear endgame-boss transformation with a legible design and recognizable original subject.",
            "Unmistakable final boss with a signature design and a coherent, finished illustration.",
        ),
    ),
    "signature-design": (
        "Would image_2 work as the final-boss reveal of the original character? Look for a "
        "distinctive, readable costume, armor, emblem or prop. Ordinary clothing with changed hair, "
        "a stern face or darker lighting is weak. Require one intact illustration with a recognizable subject.",
        (
            "No boss design, a broken illustration, or an unrelated subject.",
            "Ordinary clothes, mood, changed hair or minor styling. No distinctive boss costume or prop.",
            "Clear endgame-boss design with distinctive costume, armor, emblem or prop and a readable figure.",
            "Unmistakable boss reveal with a signature, cohesive design and a recognizable original character.",
        ),
    ),
    "character-reveal": (
        "Would image_2 work as the final-boss reveal of image_1's character? Reward a recognizable "
        "figure with a deliberate signature outfit or prop. Changing only everyday clothes, hair, "
        "expression or lighting is weak. A broken frame or a replacement character scores zero.",
        (
            "No meaningful change from the reference, broken render, or a different character.",
            "Some change, but the figure still reads as an ordinary portrait or generic costume.",
            "Clear, legible final-boss design on the original character, with distinctive costume or gear.",
            "A convincing boss reveal with a unified, signature look in a single intact illustration.",
        ),
    ),
    "design-threshold": (
        "How convincing is image_2 as a final-boss transformation of image_1's character? Judge "
        "visible design. A strong answer needs a clear signature costume, armor, insignia or prop. "
        "A stern face in ordinary clothes is weak even with dramatic lighting. Require one coherent "
        "illustration and a recognizable subject.",
        (
            "Unchanged or ordinary character, obvious image breakage, or the original subject is lost.",
            "Visible mood, hair or wardrobe changes with no distinctive final-boss design.",
            "Recognizable character transformed through distinctive, clearly illustrated costume or gear.",
            "Unmistakable final boss with a readable signature design and intact image.",
        ),
    ),
    "visible-gear": (
        "How convincing is the final-boss redesign of this character in image_2? Strong designs "
        "have visible signature armor, regalia, an emblem or a prop. A regular shirt or jacket "
        "with harsher hair, expression or shadows is still ordinary. Require an intact illustration "
        "and a recognizable original character.",
        (
            "Ordinary clothing without any visible boss-specific costume or gear, a broken image, or a replaced subject.",
            "A small costume detail hints at a boss, but the main figure still looks ordinary or generic.",
            "A recognizable character with clear and readable final-boss costume, regalia or gear.",
            "An unmistakable final boss with focused, cohesive character design in a finished illustration.",
        ),
    ),
}
PROMPTS["direct-validated-scale"] = (PROMPTS["direct-final-boss"][0], PROMPTS["visible-gear"][1])
PROMPTS["boss-read-validated-scale"] = (PROMPTS["boss-read"][0], PROMPTS["visible-gear"][1])
PAIR_STATE = "Image_1 is the original character. Image_2 is the candidate. Score only image_2."
DIRECT_CONTEXT = (
    "Compare the same character across the two images. The candidate is image_2. "
    "Judge the character transformation in the candidate only. A more intimidating expression, "
    "ordinary clothing with a darker palette, or a different person does not by itself convey Final Boss."
)
DEFECTS = {
    "no_readable_image": (
        "Has the render collapsed largely into texture, color, glare or noise, so a recognizable "
        "figure and coherent illustration are missing? A readable fantasy figure or a dim scene is normal."
    ),
    "split_or_duplication": (
        "Does the candidate visibly break into separate panels or repeat the main character "
        "because of an obvious render error?"
    ),
    "broken_structure": (
        "Is there a clear structural rendering failure, such as a duplicated head, fused or "
        "detached body parts, incoherent face or torso, torn geometry, or smeared shapes that "
        "erase structure? Require a specific visible defect. Stylized costumes, sharp armor, "
        "occlusion and cropping are normal."
    ),
}
STRUCTURE_DETAILS = {
    "accidental_parts": (
        "Is a body part obviously duplicated, attached in the wrong place or visibly separated "
        "from the figure by a drawing error? Count only visible evidence."
    ),
    "accidental_face": (
        "Is the visible face plainly malformed by extra eyes, misplaced features or disintegrating "
        "shapes? An intentional mask, stylized shading and expression are valid."
    ),
    "accidental_geometry": (
        "Does an obvious drawing error tear apart or smear the figure's contours so it ceases "
        "to read as a single formed figure? Coherent hair, armor and props can overlap."
    ),
}


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def image_records():
    panel = {row["label"]: row for row in read(ROOT / "panel.json")}
    milestone = read(ROOT / "cohorts/prompt-only/milestones/RAW000192.json")
    off = {}
    rendered = {}

    def insert(profile, alpha, case, asset):
        key = (profile, alpha, case)
        previous = rendered.setdefault(key, asset)
        if previous["sha256"] != asset["sha256"]:
            raise ValueError(f"Mismatched render receipts for {key}")

    for profile, arm in milestone["arms"].items():
        for row in arm["cases"]:
            case = row["row"]["label"]
            off[case] = row["off"]
            insert("constant" if profile == "constant_dose" else profile, 2 / 3, case, row["on"])
    for path in sorted(COMMAND.glob("*/renders/*/*/*/render-pin.json")):
        case = path.parent.name
        if case not in panel:
            continue
        insert(path.parents[2].name, float.fromhex(path.parents[1].name), case, read(path))
    for path in sorted((ALPHA / "prompt-only").glob("*/*.json")):
        if path.name not in ("screen.json", "reserved.json"):
            continue
        result = read(path)
        for row in result["cases"]:
            if row["off"]["sha256"] != off[row["row"]["label"]]["sha256"]:
                raise ValueError("A saved strength used a different reference")
            profile = result["profile"].replace("constant_dose", "constant")
            insert(profile, result["alpha"], row["row"]["label"], row["on"])
    for result in read(ALPHA / "shared-api/prompt-only.json")["reserved"]:
        for row in result["cases"]:
            insert(result["profile"], result["alpha"], row["row"]["label"], row["on"])
    for path in sorted((OUT / "renders").glob("*/*/*/render.json")):
        row = read(path)
        case = path.parent.name
        if (row["prompt"], row["seed"]) != (panel[case]["prompt"], panel[case]["seed"]):
            raise ValueError(f"Prompt or seed mismatch {path}")
        insert(path.parents[2].name, float.fromhex(path.parents[1].name), case, row["image"])
    return panel, off, rendered


def observer():
    from particle_sliders.alpha_tune_jev import JevJudge, _Image
    from particle_sliders.candidate_failure import FileRecord

    judge = JevJudge(OUT / "jev", "Coherent Final Boss transformation")

    def load(asset):
        path = Path(asset["path"])
        return _Image.read(FileRecord(str(path), asset.get("bytes", path.stat().st_size), asset["sha256"]))

    def concept(name, off, on):
        prompt, criteria = PROMPTS[name]
        values = judge._decide(
            (load(off), load(on)), DIRECT_CONTEXT if name == "direct-context" else PAIR_STATE,
            {"boss": {"type": "score", "instructions": prompt, "criteria": list(criteria)}},
        )
        return values["boss"] if values else None

    def integrity(on):
        image = load(on)
        checks = judge._decide(
            (image,), "Inspect the single finished image for visible rendering failures.",
            {name: {"type": "noul", "instructions": prompt} for name, prompt in DEFECTS.items()},
        )
        if checks is None:
            return {"status": "unmeasured", "checks": None}
        details = None
        for name, probability in list(checks.items()):
            if .5 <= probability < .8:
                if name == "broken_structure":
                    details = judge._decide(
                        (image,), "Inspect the candidate for specific visible drawing errors.",
                        {key: {"type": "noul", "instructions": text}
                         for key, text in STRUCTURE_DETAILS.items()},
                    )
                    checks[name] = max(details.values()) if details else None
                else:
                    independently = judge._decide(
                        (image,), "Look at the actual visible structures before deciding.",
                        {name: {"type": "noul", "instructions": DEFECTS[name]}},
                    )
                    checks[name] = independently[name] if independently else None
        known = [p for p in checks.values() if p is not None]
        status = "fail" if any(p >= .8 for p in known) else (
            "pass" if len(known) == len(checks) and max(known) < .5 else "unmeasured"
        )
        outcome = {"status": status, "checks": checks}
        if details:
            outcome["structural_details"] = details
        return outcome

    return concept, integrity


def audit(stage, names):
    _, off, candidates = image_records()
    fixtures = read(ROOT / "jev-rubric-audit-v2/fixtures.json")
    if stage.endswith("screen"):
        positives = ("ranger", "smith", "archivist")
        weak = ("guide",)
        replacements = ("ranger", "smith")
    else:
        positives = ("watchmaker", "bookkeeper", "botanist")
        weak = ("ferryman",)
        replacements = ("guard", "guide")
    assets = {}
    for case in (*positives, *weak, *replacements):
        assets[(case, "off")] = (off[case], off[case])
    for case in positives:
        assets[(case, "previous")] = (off[case], candidates[(*BASELINE, case)])
        assets[(case, "coherent")] = (off[case], candidates[("early", 2 / 3, case)])
    for case in weak:
        assets[(case, "ordinary")] = (off[case], candidates[("early", 2 / 3, case)])
    for case in replacements:
        assets[(case, "replacement")] = (off[case], candidates[("early", 2.0625, case)])
    fixture_names = ("broken", "split")
    for case in fixture_names:
        assets[(case, "fixture")] = (fixtures[case]["off"], fixtures[case]["on"])

    concept, integrity = observer()
    with ThreadPoolExecutor(8) as pool:
        judges = {(name, *key): pool.submit(concept, name, *pair)
                  for name in names for key, pair in assets.items()}
        guards = {key: pool.submit(integrity, pair[1])
                  for key, pair in assets.items() if key[1] in ("fixture", "coherent", "ordinary", "off")}
        guard_results = {key: job.result() for key, job in guards.items()}
        guard_qualified = all(
            entry["status"] == ("fail" if key[1] == "fixture" else "pass")
            for key, entry in guard_results.items()
        )
        summary = {}
        for name in names:
            scores = {key: judges[(name, *key)].result() for key in assets}
            clean = fmean(scores[case, "coherent"] or 0 for case in positives)
            prior = fmean(scores[case, "previous"] or 0 for case in positives)
            replace = fmean(scores[case, "replacement"] or 0 for case in replacements)
            checks = {
                "measured": all(value is not None for value in scores.values()),
                "reference": all((scores[case, "off"] or 0) <= .10
                                 for case, condition in assets if condition == "off"),
                "ordinary": all((scores[case, "ordinary"] or 0) <= .40 for case in weak),
                "recognizable_boss": clean >= .58 and all((scores[case, "coherent"] or 0) >= .48
                                                         for case in positives),
                "compared_to_previous": clean - prior >= .15,
                "compared_to_replacement": clean - replace >= .25,
                "image_integrity": guard_qualified,
            }
            summary[name] = {
                "qualified": all(checks.values()), "checks": checks,
                "coherent": clean, "previous": prior, "replacements": replace,
                "scores": {"/".join(key): score for key, score in scores.items()},
            }
    result = {"positives": positives, "assets": {"/".join(key): pair for key, pair in assets.items()},
              "rubrics": summary, "integrity": {"/".join(key): value for key, value in guard_results.items()}}
    write(OUT / f"{stage}.json", result)
    for name, candidate in summary.items():
        print(stage, name, f"{candidate['coherent']:.3f}", candidate["checks"], flush=True)
    print(stage, "integrity", {key: value["status"] for key, value in result["integrity"].items()}, flush=True)
    return result


def search(stage, name, selected=None):
    started = perf_counter()
    started_at = time_ns()
    panel, off, candidates = image_records()
    cases = SCREEN if stage.endswith("-screen") else tuple(case for case in panel if case not in SCREEN)
    settings = list(selected) if selected else [(p, a) for p, alphas in STRENGTHS.items() for a in alphas]
    concept, integrity = observer()
    results = []
    with ThreadPoolExecutor(8) as pool:
        references = {case: pool.submit(concept, name, off[case], off[case]) for case in cases}
        pending = {}
        for profile, alpha in settings:
            pending[profile, alpha] = {}
            for case in cases:
                on = candidates.get((profile, alpha, case))
                if on is None:
                    raise ValueError(f"Missing native output {profile} {alpha} {case}")
                pending[profile, alpha][case] = (on, pool.submit(concept, name, off[case], on),
                                                 pool.submit(integrity, on))
        for (profile, alpha), jobs in pending.items():
            rows = {}
            for case, (asset, effect, guard) in jobs.items():
                reference, on, status = references[case].result(), effect.result(), guard.result()
                rows[case] = {"off": off[case], "on": asset, "reference": reference, "score": on,
                              "gain": None if reference is None or on is None else on - reference,
                              "integrity": status}
            complete = all(row["gain"] is not None for row in rows.values())
            all_checks = [row["integrity"]["status"] for row in rows.values()]
            outcome = "fail" if "fail" in all_checks else (
                "pass" if complete and all(check == "pass" for check in all_checks) else "unmeasured"
            )
            mean = fmean(row["gain"] for row in rows.values()) if complete else None
            results.append(dict(profile=profile, alpha=alpha, status=outcome, gain=mean, cases=rows))
            print(stage, profile, alpha, outcome, mean, flush=True)
    write(OUT / f"{stage}.json", {"rubric": name, "settings": results,
          "started_unix_ns": started_at, "assessment_elapsed_s": perf_counter() - started})
    return results


def gpu(profile, replay_alpha=None, strengths=None):
    started = perf_counter()
    import torch
    from lumen_studio.backends.anima import TurboRuntime
    from lumen_studio.core_source import verify_core
    from alpha_sweep import resolved
    from prepare import guarded, render
    from replay import loaded

    guarded()
    verify_core()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(19 * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    panel, _, existing = image_records()
    checkpoint = read(ROOT / "cohorts/prompt-only/result.json")["equal_best"][0]["raw"]
    samp = read(COMMAND / "prompt-only-readable-jev.alpha-tune/tuning.json")["sampler"]
    sigmas = samp["sigmas"]
    model = Path(read(ROOT / "nvme-execution.json")["mirror_model"])
    if replay_alpha is None:
        missing = [(alpha, case) for alpha in strengths or STRENGTHS[profile] for case in panel
                   if (profile, alpha, case) not in existing]
        destination = OUT / "renders"
    else:
        missing = [(replay_alpha, case) for case in ("ranger", "botanist", "ferryman", "smith")]
        destination = OUT / "timed-replay"
        if destination.exists():
            raise ValueError("A cold replay requires an unused output path")
    if not missing:
        return
    configured = perf_counter()
    runtime = TurboRuntime(model, device="cuda:0", verify=True)
    torch.cuda.synchronize()
    model_loaded = perf_counter()
    timed = {}
    try:
        with torch.no_grad():
            variation = loaded(runtime, checkpoint, 192)
            torch.cuda.synchronize()
            adapter_loaded = perf_counter()
            for alpha, case in missing:
                row = panel[case]
                gains = resolved(alpha, profile, sigmas)
                if gains is None:
                    raise ValueError(f"Unsupported strength {alpha}")
                render_start = perf_counter()
                asset = render(runtime, row["prompt"], row["seed"], destination / profile / alpha.hex() / case,
                               gains=gains, variation=variation)
                if replay_alpha is not None:
                    torch.cuda.synchronize()
                    if asset["sha256"] != existing[profile, alpha, case]["sha256"]:
                        raise ValueError(f"Fresh selected-strength replay differs for {case}")
                    timed[case] = {"seconds": perf_counter() - render_start, "image": asset}
                print("rendered", profile, alpha, case, flush=True)
    finally:
        cleanup_started = perf_counter()
        runtime.close()
    if replay_alpha is not None:
        write(OUT / "replay-timing.json", {
            "profile": profile, "alpha": replay_alpha, "checkpoint": checkpoint,
            "effective_strengths": resolved(replay_alpha, profile, sigmas),
            "configure_s": configured - started, "model_load_s": model_loaded - configured,
            "adapter_load_s": adapter_loaded - model_loaded, "cleanup_s": perf_counter() - cleanup_started,
            "elapsed_s": perf_counter() - started, "cases": timed,
        })


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "final-boss-outcome":
        if len(args) > 2 and args[2] == "grid":
            gpu(args[1], strengths=[float(a) for a in args[3:]])
        else:
            gpu(args[1], float(args[2]) if len(args) == 3 else None)
    elif command in ("screen", "holdout", "direct-controls-screen", "direct-controls-holdout"):
        audit(command, args or list(PROMPTS))
    elif command == "strength-screen":
        search(command, args[0])
    elif command in ("strength-reserved", "refinement-screen", "refinement-reserved",
                     "direct-screen", "direct-reserved", "bare-screen", "bare-reserved"):
        name, *settings = args
        search(command, name, [(p, float(a)) for p, a in (item.split("=") for item in settings)])
    else:
        raise ValueError(command)
