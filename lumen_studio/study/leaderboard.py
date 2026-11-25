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
        "| Schedule | Alpha | Mean matched gain | Integrity | OFF | ON |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]

    def sample(record):
        link = quote(os.path.relpath(record["path"], root), safe="/")
        return f"[![sample]({link})]({link})"

    rows = read(root / "strengths.json")
    for row in sorted(rows, key=lambda r: (r["status"] != "pass", -r["mean_gain"])):
        case = row["pairs"][0]
        lines.append(
            f"| {row['profile']} | {row['alpha']:g} | {row['mean_gain']:+.4f} | {row['status']} | "
            f"{sample(case['case']['off']['output'])} | {sample(case['on']['output'])} |"
        )
    result = outcome["confirmation"]
    lines.extend(
        [
            "",
            "## Held-out cases",
            "",
            f"Mean intended-effect gain {result['mean_gain']:+.4f}.",
            "",
            "| Seed | OFF | ON | Gain |",
            "| ---: | --- | --- | ---: |",
        ]
    )
    for pair in result["pairs"]:
        case = pair["case"]
        lines.append(
            f"| {case['seed']} | {sample(case['off']['output'])} | {sample(pair['on']['output'])} | {pair['gain']:+.4f} |"
        )
    pending = root / "leaderboard.pending.md"
    pending.write_text("\n".join(lines) + "\n")
    pending.replace(root / "leaderboard.md")
