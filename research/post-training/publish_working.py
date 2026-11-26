"""Publish the leaderboard re-searched and ranked by the working Boss score."""

from html import escape
from pathlib import Path
from statistics import fmean
import json
import shutil

from PIL import Image

from campaign import PARENT_STUDY, POLICY, ROOT

home = ROOT / "working-search"
gallery = ROOT / "gallery"
(gallery / "images").mkdir(parents=True, exist_ok=True)
(gallery / "thumbs").mkdir(exist_ok=True)
panels = json.loads((PARENT_STUDY / "cohort-retest/panels.json").read_bytes())
cases = {c["id"]: c for c in panels["cases"]}
ranking = panels["panels"]["ranking"]
LABELS = {
    "parent": "Parent · no further training",
    "continued": "Exact continuation · same targets",
    "control": "Fork control · fresh critic, same targets",
    "cleaner": "Fork · one cleaner teacher",
    "cleanest": "Fork · five least-damaged teachers",
    "control-seed2": "Fork control, second seed",
    "cleanest-seed2": "Fork, second seed · five least-damaged teachers",
    "control-warm": "Fork control with the parent's critic",
    "cleanest-warm": "Fork with the parent's critic · five least-damaged teachers",
    "control-low-g": "Fork control at the lower generator rate",
    "cleanest-low-g": "Fork at the lower generator rate · five least-damaged teachers",
}


def read(path):
    return json.loads(Path(path).read_bytes())


def asset(ref):
    name = ref["sha256"] + ".png"
    if not (gallery / "images" / name).exists():
        shutil.copyfile(ref["path"], gallery / "images" / name)
    thumb = gallery / "thumbs" / (ref["sha256"] + ".webp")
    if not thumb.exists():
        with Image.open(ref["path"]) as image:
            image.convert("RGB").resize((116, 116), Image.Resampling.LANCZOS).save(thumb, quality=82)
    return name


def cell(ref, title, mark=""):
    name = asset(ref)
    return (f'<a class="{mark}" href="images/{name}" title="{escape(title)}"><img loading="lazy" width="58" '
            f'height="58" alt="{escape(title)}" src="thumbs/{ref["sha256"]}.webp"></a>')


entries, pending = [], []
for directory in sorted(p for p in home.iterdir() if p.is_dir() and p.name != "drawing-receipts"):
    if not (directory / "result.json").exists():
        pending.append(directory.name)
        continue
    result = read(directory / "result.json")
    if result.get("score") is None:
        pending.append(directory.name + " (unmeasured)")
        continue
    doc = read(directory / "ranking" / (result["alpha"].hex() + ".json"))
    rows = doc["rows"]
    arm, extra = directory.name.rsplit("-", 1)
    working = sum(r["quality"] >= 0.5 and r["boss"] >= 0.5 for r in rows.values())
    entries.append(dict(
        name=directory.name, arm=arm, extra=int(extra), updates=result["step"] if arm in ("parent", "continued")
        else POLICY["parent"]["step"] + int(extra),
        alpha=result["alpha"], score=100 * result["score"], search=100 * result["search_score"],
        gain=100 * doc["mean_gain"], quality=100 * fmean(r["quality"] for r in rows.values()),
        drawn=100 * fmean(1 - min(1.0, max(0.0, r["broken"] - r["off_broken"]) / 0.30) for r in rows.values()),
        one_figure=100 * fmean(r["one_figure"] for r in rows.values()),
        same_person=100 * fmean(r["same_person"] for r in rows.values()),
        repeat=100 * fmean(p["repeat"] for p in doc["pairs"]), working=working,
        previous_alpha=result["previous_alpha"],
        previous_score=None if result["previous_score"] is None else 100 * result["previous_score"],
        trials=[[a, None if v is None else 100 * v] for a, v in result["trials"]], rows=rows))
entries.sort(key=lambda e: -e["score"])
old_order = sorted((e for e in entries if e["previous_score"] is not None), key=lambda e: -e["previous_score"])
names = [c.title() for c in ranking]
lines = ['<tr class="ref"><td>—</td><th scope="row">Matched baseline</th><td>—</td><td>OFF</td><td><div class="images">'
         + "".join(cell(cases[c]["off"], n) for c, n in zip(ranking, names)) + '</div></td><td colspan="12"></td></tr>']
for rank, e in enumerate(entries, 1):
    label = LABELS[e["arm"]] + (f" · +{e['extra']}" if e["extra"] else "")
    images = "".join(
        cell(e["rows"][c]["on"],
             f'{n} · case {100 * e["rows"][c]["case"]:.0f} · boss {100 * e["rows"][c]["boss"]:.0f} · quality '
             f'{100 * e["rows"][c]["quality"]:.0f} · one figure {100 * e["rows"][c]["one_figure"]:.0f} · same person '
             f'{100 * e["rows"][c]["same_person"]:.0f}',
             "ok" if e["rows"][c]["quality"] >= 0.5 and e["rows"][c]["boss"] >= 0.5
             else "bad" if e["rows"][c]["quality"] < 0.25 else "unsure")
        for c, n in zip(ranking, names))
    was = old_order.index(e) + 1 if e in old_order else "—"
    before = "—" if e["previous_score"] is None else f'{e["previous_score"]:.2f} at α {e["previous_alpha"]:g}'
    lines.append(
        f'<tr data-score="{e["score"]}" data-working="{e["working"] * 1000 + e["quality"]}" data-gain="{e["gain"]}" '
        f'data-quality="{e["quality"]}"><td>{rank}</td><th scope="row">{escape(label)}</th><td>{e["updates"]}</td>'
        f'<td>{e["alpha"]:g}</td><td><div class="images">{images}</div></td><td><b>{e["score"]:.2f}</b></td>'
        f'<td>{e["working"]} of 8</td><td>{e["gain"]:+.1f}</td><td>{e["quality"]:.0f}</td><td>{e["drawn"]:.0f}</td>'
        f'<td>{e["one_figure"]:.0f}</td><td>{e["repeat"]:.0f}</td><td>{e["same_person"]:.0f}</td>'
        f'<td>{e["search"]:.2f}</td><td>{len(e["trials"])}</td><td>{before}</td><td>{was}</td></tr>')
note = f"<p><b>Still running:</b> {escape(', '.join(pending))}.</p>" if pending else ""
shutil.copyfile(PARENT_STUDY / "sample-gallery/gallery.css", gallery / "gallery.css")
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Final Boss post-training · working Boss score</title><link rel="stylesheet" href="gallery.css"><style>
main,header{{max-width:1540px}}.scroll{{overflow:auto;border:1px solid #41434a;border-radius:7px;margin-top:14px}}
.scroll table{{margin:0;width:100%;font-size:14px;border-collapse:collapse}}.scroll td,.scroll th{{padding:6px 8px;text-align:left}}
.scroll thead{{background:#292b30}}.scroll th[scope=row]{{min-width:170px;max-width:300px}}.images{{display:flex;gap:4px;width:max-content}}
.images a,.images img{{display:block}}.images img{{width:58px;height:58px;object-fit:cover;border-radius:3px}}
.images a.ok img{{outline:3px solid #6fd08c;outline-offset:-3px}}.images a.bad img{{outline:3px solid #ff5d5d;outline-offset:-3px}}
.images a.unsure img{{outline:2px dashed #e0b84c;outline-offset:-2px}}.scroll tr:hover{{background:#242529}}.scroll .ref{{background:#202328}}
</style></head><body><header><a href="post-training-leaderboard.html">Earlier board, current distinct Boss score</a><nav><a href="alpha-review.html">Per-strength review</a><a href="working-results.json">Scores and provenance</a></nav></header><main>
<h1>Final Boss post-training · working Boss score</h1>
<p>Every RAW had its constant alpha chosen again by the working Boss score on the four search characters, and was then ranked on the eight held-out characters at that alpha. The score keeps the distinct Boss effect term, the Boss gain times Boss distinction cubed with the same discount for repeated designs. It replaces the damage term with quality: how cleanly the image is drawn, read by Mercury Decide on four 256-pixel tiles at native resolution against the same character's OFF image, and limited by whether there is exactly one figure. Looking like the original person is shown and not scored.</p>
<p>A green outline marks a working image, with quality and Boss distinction both 50 or more. Red is quality under 25. Hover an image for its readings.</p>{note}
<p><label>Order by <select id="metric"><option value="score">Working Boss score</option><option value="working">Working images</option><option value="gain">Boss gain</option><option value="quality">Quality</option></select></label></p>
<div class="scroll"><table><thead><tr><th>Rank</th><th>Approach</th><th>Updates</th><th>Constant α</th><th>Eight held-out characters</th><th>Working Boss / 100</th><th>Working</th><th>Boss gain</th><th>Quality</th><th>Drawn</th><th>One figure</th><th>Repeat</th><th>Same person, not scored</th><th>Search score</th><th>Trials</th><th>Distinct Boss before</th><th>Rank before</th></tr></thead><tbody>{"".join(lines)}</tbody></table></div>
</main><script>
(()=>{{const body=document.querySelector('tbody'),rows=[...body.querySelectorAll('[data-score]')],pick=document.querySelector('#metric');
function refresh(){{const key=pick.value;[...rows].sort((a,b)=>Number(b.dataset[key])-Number(a.dataset[key])).forEach((r,i)=>{{body.append(r);r.firstChild.textContent=i+1}})}}
const wanted=new URLSearchParams(location.search).get('metric');if(['score','working','gain','quality'].includes(wanted))pick.value=wanted;
pick.addEventListener('change',refresh);refresh()}})();</script></body></html>"""
(gallery / "working-leaderboard.html").write_text(page)
for e in entries:
    e["rows"] = {c: {k: (v["sha256"] if k == "on" else v) for k, v in r.items()} for c, r in e["rows"].items()}
(gallery / "working-results.json").write_text(json.dumps(dict(entries=entries, pending=pending), indent=1) + "\n")
print(f'{"rank":>4} {"was":>4}  {"row":<20}{"α":>9}{"was α":>9}{"score":>8}{"working":>9}{"gain":>7}{"quality":>9}{"repeat":>8}')
for rank, e in enumerate(entries, 1):
    was = old_order.index(e) + 1 if e in old_order else 0
    print(f'{rank:>4} {was:>4}  {e["name"]:<20}{e["alpha"]:>9.4g}{(e["previous_alpha"] or 0):>9.4g}{e["score"]:>8.2f}'
          f'{e["working"]:>7}/8{e["gain"]:>7.1f}{e["quality"]:>9.0f}{e["repeat"]:>8.0f}')
print("pending:", pending)
