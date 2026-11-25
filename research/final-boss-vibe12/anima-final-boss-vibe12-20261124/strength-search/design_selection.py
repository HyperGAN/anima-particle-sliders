"""Expose only completed, receipt-bound distinct-design selections on the public board."""

from hashlib import sha256
from html import escape
import json
from pathlib import Path

RUN = "leaderboard-retune-design"
PAGE = "final-boss-vibe12-design-alpha.html"


def read(path):
    return json.loads(path.read_bytes())


def selections(root, cases):
    selected = {}
    for terminal in sorted(root.glob("*/result.json")):
        folder = terminal.parent
        result = read(terminal)
        saved = result["selected"]
        if (
            saved != read(folder / "tuning.json")
            or saved["method"] != folder.name
            or saved["objective"] != "distinct_boss_score"
        ):
            raise ValueError("Mismatched distinct-design tuning")
        path = (
            folder
            / "renders"
            / saved["profile"]
            / saved["alpha"].hex()
            / "validation.json"
        )
        trial = read(path)
        if (
            sha256(path.read_bytes()).hexdigest() != saved["measurement"]["sha256"]
            or trial["design"] != saved["design"]
            or trial["setting"]["cases"] != cases
            or trial["setting"]["effective_strengths"] != saved["effective_strengths"]
        ):
            raise ValueError("Selected measurement differs from its native tuning")
        selected[folder.name] = (saved, trial)
    return selected


def publish_paths(root, public, labels, selections):
    sections = []
    for method, (tuning, trial) in selections.items():
        folder = root / method
        result = read(folder / "result.json")
        profile, alpha = tuning["profile"], tuning["alpha"]
        portable = {
            k: tuning[k]
            for k in (
                "method",
                "objective",
                "profile",
                "alpha",
                "effective_strengths",
                "sigmas",
                "model",
                "design",
            )
        }
        portable.update(
            checkpoint_sha256=tuning["checkpoint"]["sha256"],
            definition_sha256=tuning["definition"]["sha256"],
        )
        payload = (json.dumps(portable, indent=2) + "\n").encode()
        link = f"final-boss-vibe12-tunings/{method}/{sha256(payload).hexdigest()}.json"
        artifact = public / link
        if not artifact.is_file():
            artifact.parent.mkdir(parents=True, exist_ok=True)
            pending = artifact.with_suffix(".pending")
            pending.write_bytes(payload)
            pending.replace(artifact)
        lines = []
        for name, path in result["search"].items():
            probes = " · ".join(
                f"{r['alpha']:g} → {100 * r['score']:.2f}" for r in path["trials"]
            )
            lines.append(f"<p><b>{name.title()}</b> {probes}</p>")
        confirmed = []
        for setting in result["measured"]:
            p, a = setting["profile"], setting["alpha"]
            measured = read(folder / "renders" / p / a.hex() / "validation.json")
            confirmed.append((measured["design"]["score"], p, a))
        table_rows = "".join(
            f"<tr><td>{p.title()}</td><td>{a:g}</td><td>{100 * s:.2f}</td></tr>"
            for s, p, a in sorted(confirmed, reverse=True)
        )
        sections.append(
            f'<section id="{escape(method)}"><h2>{escape(labels[method])}</h2>'
            f'<p>Selected <b>{profile.title()} α {alpha:g}</b> · distinct Boss score {100 * tuning["design"]["score"]:.2f} · <a href="{link}" download>Download tuning</a></p>'
            f"<details><summary>Four-character search path, α → score</summary>{''.join(lines)}</details>"
            f"<details><summary>Eight-character comparisons</summary><table><tr><th>Schedule</th><th>α</th><th>Score</th></tr>{table_rows}</table></details>"
            f"<small>{len(result['pruned'])} saved strengths could not beat the best measured design score even with zero repetition.</small></section>"
        )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Distinct Final Boss alpha search</title><style>:root{color-scheme:dark;font:16px/1.45 system-ui;background:#17191d;color:#eee}"
        "main{max-width:1000px;margin:22px auto;padding:0 12px}a{color:#ffc59f}section{border-top:1px solid #555}td,th{padding:3px 9px;text-align:left}"
        'summary{cursor:pointer}small{color:#c2c3ca}</style><main><a href="final-boss-vibe12-alpha-leaderboard.html?metric=design">Leaderboard and samples</a>'
        "<h1>Distinct Final Boss alpha search</h1><p>Jev scores each probe on four matched characters, then compares the finalists on eight. "
        "Both panels use the same effect, damage and character design repetition formula. Higher scores are better.</p>"
        + "".join(sections)
        + "</main></html>"
    )
    temp = public / (PAGE + ".pending")
    temp.write_text(page)
    temp.replace(public / PAGE)
