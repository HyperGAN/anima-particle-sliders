"""Evaluate a second blind preference question on the same frozen panel."""
from PIL import Image
from preference_round import OUT, QUESTIONS, STATE, PROTOCOL, ROOT, freeze, harness, judge, run

HERE = OUT.parent / "preference-finish"
FINISH = {"preference": dict(QUESTIONS["preference"], instructions=(
    "Which finished edit would you keep for 'Final Boss'? First look at the character: an ordinary person or "
    "generic guard falls short, even if impeccably drawn. When both read as formidable final opponents, judge "
    "how convincingly each design is drawn. Prefer forms, surfaces, and focal features that read clearly together. "
    "Pushing the effect until details fuse, flatten, or become muddy can make a powerful design worse. "
    "Give full credit to intentional stylization, nonhuman anatomy, masks, and dramatic light. "
    "Choose the middle answer if the finished results are equally successful."))}
RULE = (PROTOCOL + " This second wording was written after inspecting the first experiment. To qualify as a "
        "finalist view it must additionally retain both damage-panel preferences, favor the intact illustration "
        "over the two controlled damage variants in both image orders on all three transfer subjects, and preserve "
        "a strict majority of the first panel's six preferences. Historical cases are exposed calibration labels.")


def panels():
    data = freeze()
    top, opening = next(iter(data["settings"])), "shared-best-opening"
    current = data["groups"]["current"]
    ref = next(r for r in harness.read(ROOT / "storage/alpha-standings.json")["rows"]
               if (r["method"], r["profile"], r["alpha"]) == (opening, "early", 1.15625))
    data["groups"]["reference"] = {"opening-reference":ref["cases"], opening:current[opening], top:current[top]}
    # Audit repair sensitivity at fixed concept strength by corrupting a copy of the same finished illustration.
    for defect in ("channel-offset", "displacement"):
        clean, damaged = {}, {}
        for case in ("watchmaker", "guide", "guard"):
            source = current[top][case]
            path = HERE / "controls" / f"{case}-{defect}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source["on"]["path"]) as image:
                img = image.convert("RGB")
            w, h = img.size
            if defect == "channel-offset":
                red, green, blue = img.split()
                red.paste(red.crop((0,0,3*w//4,h)), (w//8,0))
                img = Image.merge("RGB",(red,green,blue))
            else:
                original = img.copy()
                for y in range(h//5, 4*h//5, max(1,h//16)):
                    patch = original.crop((0,y,w-w//7,y+max(1,h//26)))
                    img.paste(patch, (w//7,y))
            img.save(path)
            clean[case] = source
            damaged[case] = dict(source, on=harness.pin(path), final_boss_score=None)
        data["groups"][defect] = {"intact":clean, "damaged":damaged}
    harness.write(HERE / "panels.json", data)
    return data


if __name__ == "__main__":
    judge()
    run(panels(), HERE, STATE, FINISH, RULE)
