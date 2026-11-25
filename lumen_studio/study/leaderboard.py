"""Write a portable Markdown comparison from recorded finished outputs."""

import os
from urllib.parse import quote
from .outputs import read


def gallery(root, outcome):
    best = outcome["strength"]
    lines = [
        "# " + outcome["vibe"],
        "",
        f"RAW update {outcome['selected_step']}. "
        f"The selected strength is {best['profile']} at α {best['alpha']:g}.",
        "",
        "## Strength comparison",
        "",
        "| Schedule | Alpha | Mean matched gain | Status | OFF | ON |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]

    def sample(record):
        link = quote(os.path.relpath(record["path"], root), safe="/")
        return f"[![sample]({link})]({link})"

    def display_gain(value):
        return "unmeasured" if value is None else f"{value:+.4f}"

    rows = read(root / "strengths.json")
    for row in sorted(
        rows,
        key=lambda r: (
            r["status"] != "pass",
            r["mean_gain"] is None,
            -(r["mean_gain"] or 0),
        ),
    ):
        case = row["pairs"][0]
        lines.append(
            f"| {row['profile']} | {row['alpha']:g} | {display_gain(row['mean_gain'])} | {row['status']} | "
            f"{sample(case['case']['off']['output'])} | {sample(case['on']['output'])} |"
        )
    result = outcome["confirmation"]
    lines.extend(
        [
            "",
            "## Held-out cases",
            "",
            f"Mean intended-effect gain {display_gain(result['mean_gain'])}.",
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
    pending = root / "leaderboard.pending.md"
    pending.write_text("\n".join(lines) + "\n")
    pending.replace(root / "leaderboard.md")
