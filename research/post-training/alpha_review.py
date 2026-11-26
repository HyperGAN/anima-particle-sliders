"""Show one RAW at every searched strength with the readings that decide whether each edit works."""

from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from statistics import fmean
import contextlib
import io
import shutil
import sys

from PIL import Image
from particle_sliders.alpha_tune_jev import _Image
from particle_sliders.candidate_failure import FileRecord

from lumen_studio.contracts import atomic_json
from lumen_studio.vibe_integrity import FrameIntegrity

from campaign import PARENT_STUDY, POLICY, ROOT

with contextlib.redirect_stdout(io.StringIO()):
    from working_metric import D, ask, cases, judge, read, root

from native_tiles import ask as read_tile, tiles  # noqa: E402

DRAWING_MODEL = "inception/mercury-decide"
# Drawing quality is the worst of four overlapping 320-pixel tiles at native resolution, so
# nothing is shrunk. It counts from the OFF image's own reading up to this much above it.
BROKEN_SPAN = 0.40
integrity = FrameIntegrity(root / "judgments")
gallery = ROOT / "gallery"


def record(ref):
    return FileRecord.from_path(Path(ref["path"]))


def broken(ref):
    values = [read_tile(DRAWING_MODEL, tile) for tile in tiles(ref["path"], 2, 320)]
    return None if None in values else max(values)


def structure(ref):
    return max(integrity.check(record(ref))["checks"].values())


def thumb(ref, size=176):
    target = gallery / "review" / f'{ref["sha256"]}.webp'
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        with Image.open(ref["path"]) as picture:
            picture.convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(target, quality=85)
    full = gallery / "images" / f'{ref["sha256"]}.png'
    if not full.exists():
        shutil.copyfile(ref["path"], full)
    return f'review/{ref["sha256"]}.webp', f'images/{ref["sha256"]}.png'


def review(name):
    base = (PARENT_STUDY / POLICY["parent"]["calibration"] / D if name == "parent-0"
            else ROOT / "arms" / name.rsplit("-", 1)[0] / "calibration" / name.rsplit("-", 1)[1] / D)
    chosen = read(base / "result.json")["alpha"]
    docs = sorted((read(p) for p in (base / "search").glob("*.json")), key=lambda d: d["identity"]["alpha"])
    names = list(docs[0]["rows"])
    with ThreadPoolExecutor(8) as pool:
        off = {c: (pool.submit(broken, cases[c]["off"]), pool.submit(structure, cases[c]["off"])) for c in names}
        jobs = [(d, {c: (r, pool.submit(ask, cases[c]["off"], r["on"]), pool.submit(structure, r["on"]),
                         pool.submit(broken, r["on"])) for c, r in d["rows"].items()}) for d in docs]
        rows = []
        for d, row in jobs:
            cells = {}
            for c, (r, pair, broke, drawing) in row.items():
                a, b = pair.result(), drawing.result()
                parts = dict(
                    boss=r["boss_distinction"],
                    same_person=a and a["same_person"],
                    one_figure=a and a["one_figure"],
                    intact=1 - max(0.0, broke.result() - off[c][1].result()),
                    drawn=None if b is None else 1 - min(1.0, max(0.0, b - off[c][0].result()) / BROKEN_SPAN),
                )
                works = None if None in parts.values() else min(parts.values())
                cells[c] = dict(on=r["on"], works=works, **parts)
            values = [v["works"] or 0.0 for v in cells.values()]
            rows.append(dict(alpha=d["identity"]["alpha"], current=d["score"], gain=d["mean_gain"],
                             working=sum(v >= 0.5 for v in values), mean=fmean(values), cells=cells))
    best = max(rows, key=lambda r: (r["working"], r["mean"]))
    atomic_json(root / f"alpha-review-{name}.json", dict(chosen_current=chosen, chosen_working=best["alpha"], rows=rows))
    body = []
    head = "".join(f'<figure><a href="{thumb(cases[c]["off"])[1]}"><img width="176" height="176" src="{thumb(cases[c]["off"])[0]}"></a><figcaption>{escape(c)} OFF</figcaption></figure>' for c in names)
    body.append(f'<tr><th scope="row">OFF</th><td><div class="grid">{head}</div></td><td colspan="4"></td></tr>')
    for r in rows:
        figures = []
        for c, v in r["cells"].items():
            small, full = thumb(v["on"])
            works = v["works"] or 0.0
            mark = "ok" if works >= 0.5 else "bad" if works < 0.25 else "unsure"
            weakest = min((k for k in ("boss", "same_person", "one_figure", "intact", "drawn")), key=lambda k: v[k] or 0)
            figures.append(
                f'<figure class="{mark}"><a href="{full}"><img loading="lazy" width="176" height="176" src="{small}"></a>'
                f'<figcaption>works {100 * works:.0f} · weakest: {weakest.replace("_", " ")}<br>boss {100 * (v["boss"] or 0):.0f} · person {100 * (v["same_person"] or 0):.0f} · '
                f'intact {100 * (v["intact"] or 0):.0f} · drawn {100 * (v["drawn"] or 0):.0f}</figcaption></figure>')
        tags = (" · <b>current score picks this</b>" if r["alpha"] == chosen else "") + (" · <b>working metric picks this</b>" if r is best else "")
        body.append(f'<tr class="{"pick" if r is best else ""}"><th scope="row">α {r["alpha"]:g}{tags}</th><td><div class="grid">{"".join(figures)}</div></td>'
                    f'<td>{r["working"]} of {len(names)}</td><td>{100 * r["mean"]:.1f}</td><td>{100 * r["gain"]:+.1f}</td><td>{100 * (r["current"] or 0):.2f}</td></tr>')
    print(name, "current picks", chosen, "working picks", best["alpha"], f'({best["working"]} of {len(names)}, mean {100 * best["mean"]:.1f})')
    for r in rows:
        print(f'  α {r["alpha"]:<8g} working {r["working"]}/{len(names)} mean {100 * r["mean"]:5.1f} gain {100 * r["gain"]:+6.1f} current {100 * (r["current"] or 0):6.2f}')
    return f'<h2 id="{name}">{escape(name)}</h2><div class="scroll"><table><thead><tr><th>Strength</th><th>Four search characters</th><th>Working</th><th>Works mean</th><th>Boss gain</th><th>Current score</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>'


sections = [review(name) for name in (sys.argv[1:] or ["parent-0"])]
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Choosing alpha by working edits</title><link rel="stylesheet" href="gallery.css"><style>
main,header{{max-width:1300px}}.scroll{{overflow:auto;border:1px solid #41434a;border-radius:7px;margin-top:14px}}table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{padding:6px 8px;text-align:left;vertical-align:top}}thead{{background:#292b30}}tr.pick{{background:#1f2a22}}th[scope=row]{{min-width:150px;max-width:190px}}
.grid{{display:flex;gap:6px}}.grid figure{{width:176px;flex:none}}.grid img{{display:block;width:176px;height:176px;border-radius:3px}}
.grid figcaption{{font-size:.72rem;line-height:1.25}}figure.ok img{{outline:3px solid #6fd08c;outline-offset:-3px}}figure.bad img{{outline:3px solid #ff5d5d;outline-offset:-3px}}figure.unsure img{{outline:2px dashed #e0b84c;outline-offset:-2px}}
</style></head><body><header><a href="post-training-leaderboard.html">Post-training leaderboard</a></header><main>
<h1>Choosing alpha by working edits</h1>
<p>Each row is one constant strength of the same RAW on the four search characters. An edit works as far as its weakest reading: a final-boss design, the same recognizable person, exactly one figure, intact structure, and drawing quality. Structure and drawing are read against the same character's OFF image. Drawing is Mercury Decide's reading of the worst of four overlapping 320-pixel tiles at native resolution. It falls from 100 at the OFF reading to 0 at {100 * BROKEN_SPAN:.0f} points above it. Green is 50 or more, red is under 25. The highlighted row has the most working edits.</p>
{"".join(sections)}</main></body></html>"""
shutil.copyfile(PARENT_STUDY / "sample-gallery/gallery.css", gallery / "gallery.css")
(gallery / "alpha-review.html").write_text(page)
