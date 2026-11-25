"""Write a portable Markdown comparison from recorded finished outputs."""

import os
from pathlib import Path
from urllib.parse import quote
from .outputs import read
from .objectives import DISTINCT_BOSS, selection_score


def gallery(root, outcome, *, reference=None):
    best = outcome["strength"]
    design = best.get("objective") == DISTINCT_BOSS
    label = "Distinct Boss score" if design else "Mean matched gain"
    scalar_run = (root / "training-selection.json").exists()
    lines = [
        "# " + outcome["vibe"],
        "",
        f"RAW update {outcome['selected_step']}. "
        f"The selected strength is {best['profile']} at α {best['alpha']:g}.",
        "",
        ("The [training selection](training-selection.json) pins the rates and RAW before calibration. "
         "Each alpha below applies equally to all ten denoising steps." if scalar_run
         else "This archived run uses its recorded checkpoint and strength selection."),
        "",
    ]

    def sample(record):
        link = quote(os.path.relpath(record["path"], root), safe="/")
        return f"[![sample]({link})]({link})"

    def display_gain(value):
        return "unmeasured" if value is None else f"{value:+.4f}"

    if scalar_run:
        probes = []
        for path in sorted(root.glob("rates/*/fit/[0-9]*.json")):
            arm = path.parent.parent
            row = read(path)
            if row["profile"] != "constant" or row["alpha"] != 1 or row["gains"] != [1] * 10:
                raise ValueError("Training comparison requires a shared constant alpha of one")
            rates = read(arm / "cursor.json")["rates"]
            check = arm / "fit-screen" / path.name
            status = read(check)["status"] if check.exists() and row["status"] == "pass" else row["status"]
            if (arm / "failure.json").exists():
                status = "candidate stopped"
            gain = None if design and row.get("objective") != DISTINCT_BOSS else selection_score(row)
            selected = " · selected" if row["raw"] == best["raw"] else ""
            text = (f"| [{path.stem}]({path.relative_to(root).as_posix()}){selected} "
                    f"| {rates['g']:g} | {rates['d']:g} | {rates['particle']:g} "
                    f"| {display_gain(gain)} | {status} |")
            probes.append((gain, text))
        if probes:
            lines.extend(["## Training checkpoint comparison", "",
                          "Every checkpoint used the same selection cases at constant α 1.", "",
                          f"| Update | G | D | Particle | {label} | Integrity |",
                          "| ---: | ---: | ---: | ---: | ---: | --- |"])
            for _, text in sorted(probes, key=lambda item: (item[0] is None, -(item[0] or 0))):
                lines.append(text)
            lines.append("")
    lines.extend([
        "## Constant alpha calibration" if scalar_run else "## Historical strength comparison",
        "",
        f"| RAW | Schedule | Alpha | {label} | Status | OFF | ON |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ])
    rows = read(root / "strengths.json")
    for row in sorted(
        rows,
        key=lambda r: (
            selection_score(r) is None,
            -(selection_score(r) or 0),
            r["alpha"],
        ),
    ):
        case = row["pairs"][0]
        lines.append(
            f"| {Path(row['raw']['path']).stem} | {row['profile']} | {row['alpha']:g} | {display_gain(selection_score(row))} | {row['status']} | "
            f"{sample(case['case']['off']['output'])} | {sample(case['on']['output'])} |"
        )
    result = outcome["confirmation"]
    lines.extend(
        [
            "",
            "## Held-out cases",
            "",
            f"{label} {display_gain(selection_score(result))}.",
            "",
            "| Seed | OFF | ON | Gain |",
            "| ---: | --- | --- | ---: |",
        ]
    )
    for pair in result["pairs"]:
        case = pair["case"]
        lines.append(
            f"| {case['seed']} | {sample(case['off']['output'])} | {sample(pair['on']['output'])} | {display_gain(pair['gain'])} |"
        )
    if reference is not None:
        lines.extend(["", f"## Published {outcome['vibe']} comparison", "", "This run is a study candidate."])
        if reference["status"] == "matched" and reference.get("objective") == DISTINCT_BOSS:
            lines.append(f"On the same cases the reference scored {reference['reference_score']:.4f} and this candidate scored {reference['trial_score']:.4f} on distinct Boss design.")
        elif reference["status"] == "matched":
            lines.append(f"On the same prompts, seeds and OFF images, {reference['reference']} gained "
                         f"{reference['reference_mean_gain']:+.4f} and this run gained "
                         f"{reference['trial_mean_gain']:+.4f}.")
        else:
            lines.append("The saved reference measurements do not cover this exact panel and rubric.")
        lines.extend(["", "[Per-case effect and integrity evidence](reference-comparison.json)"])
    pending = root / "leaderboard.pending.md"
    pending.write_text("\n".join(lines) + "\n")
    pending.replace(root / "leaderboard.md")
