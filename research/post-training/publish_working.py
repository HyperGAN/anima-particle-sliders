"""Publish the leaderboard re-searched and ranked by the working Boss score."""

from html import escape
from pathlib import Path
from statistics import fmean
import json
import shutil
import sys

from PIL import Image

from lumen_studio.study import working_score as ws
from lumen_studio.study.design_score import summarize

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
    "quiet-fresh": "Clean background target · from scratch",
    "quiet-15-fresh": "Background target at a 0.15 floor · from scratch",
    "quiet-warm": "Fork with the parent's critic · clean background target",
    "working-quiet-fresh": "Teachers chosen by the working Boss score, clean background · from scratch",
    "portraits-fresh": "Chosen teachers plus eight portrait scenes · from scratch",
    "portraits-warm": "Fork with the parent's critic · chosen teachers plus eight portrait scenes",
    "portraits-strong-fresh": "Chosen teachers plus eight portrait scenes, edit × 1.4 · from scratch",
    "portraits-strong-warm": "Fork with the parent's critic · chosen teachers plus eight portrait scenes, edit × 1.4",
    "portraits-original-fresh": "Parent's teachers plus eight portrait scenes · from scratch",
    "portraits-original-warm": "Fork with the parent's critic · parent's teachers plus eight portrait scenes",
    "portraits-original-strong-fresh": "Parent's teachers plus eight portrait scenes, edit × 1.4 · from scratch",
    "portraits-original-strong-warm": "Fork with the parent's critic · parent's teachers plus eight portrait scenes, edit × 1.4",
    "portraits-original-quiet-strong-fresh": "Parent's teachers plus portraits, clean background, character edit × 1.4 · from scratch",
    "portraits-original-quiet-strong-warm": "Fork with the parent's critic · parent's teachers plus portraits, clean background, character edit × 1.4",
    "portraits2-original-fresh": "Parent's teachers plus eight portrait scenes, stronger portrait teachers · from scratch",
    "portraits2-original-warm": "Fork with the parent's critic · parent's teachers plus portraits, stronger portrait teachers",
    "portraits2-original-quiet-strong-fresh": "Parent's teachers plus portraits (stronger teachers), clean background, character edit × 1.4 · from scratch",
    "portraits2-original-quiet-strong-warm": "Fork with the parent's critic · portraits (stronger teachers), clean background, character edit × 1.4",
    "portraits-original-low-g-warm": "Fork with the parent's critic at the lower generator rate · parent's teachers plus portraits",
    "portraits24-original-fresh": "Parent's teachers plus twenty-four portrait scenes · from scratch",
    "portraits24-original-warm": "Fork with the parent's critic · parent's teachers plus twenty-four portrait scenes",
    "working-warm": "Fork with the parent's critic · teachers chosen by the working Boss score",
    "working-fresh": "Teachers chosen by the working Boss score · from scratch",
    "working-quiet-strong-warm": "Fork with the parent's critic · chosen teachers, clean background, character edit × 1.4",
    "working-quiet-strong-fresh": "Chosen teachers, clean background, character edit × 1.4 · from scratch",
    "working2-quiet-fresh": "Second round of chosen teachers, clean background · from scratch",
    "working2-quiet-warm": "Fork with the parent's critic · second round of chosen teachers, clean background",
    "working-quiet-warm": "Fork with the parent's critic · teachers chosen by the working Boss score, clean background",
    "quiet-strong-fresh": "Clean background target, character edit × 1.4 · from scratch",
    "quiet-strong-warm": "Fork with the parent's critic · clean background target, character edit × 1.4",
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


def failed(e, case):
    return bool(e["structure"]) and e["structure"]["checks"][case]["status"] == "fail"


# "board" adds every arm of the constant-alpha retest to the post-training rows.
combined = sys.argv[1:] == ["board"]
inventory = read(PARENT_STUDY / "cohort-retest/inventory.json") if combined else {}
folders = [p for p in home.iterdir() if p.is_dir() and p.name != "drawing-receipts"]
if combined:
    folders += [p for p in (ROOT / "board-search").iterdir() if p.is_dir()]
entries, pending = [], []
for directory in sorted(folders):
    if not (directory / "result.json").exists():
        pending.append(directory.name)
        continue
    result = read(directory / "result.json")
    if result.get("score") is None:
        pending.append(directory.name + " (unmeasured)")
        continue
    doc = read(directory / "ranking" / (result["alpha"].hex() + ".json"))
    rows = doc["rows"]
    if directory.parent.name == "board-search":
        record = inventory[directory.name]
        arm, extra = None, 0
        label = record["label"]
        group = f'{record["run"]} · PG {record["pg"]} · {record["training"]}'
        updates = record["step"]
    else:
        arm, extra = directory.name.rsplit("-", 1)
        fresh = arm != "parent" and POLICY["arms"][arm]["entry"] == "fresh"
        label = LABELS[arm] + (f" · +{extra}" if int(extra) and not fresh else "")
        group = "Late cues from scratch · PG 0.12.0" if fresh else "RAW512 post-training · PG 0.12.0"
        updates = result["step"] if fresh or arm in ("parent", "continued") else POLICY["parent"]["step"] + int(extra)
    for r in rows.values():
        r["effect"] = ws.effect(r["boss"], r["original"])
    working = sum(r["quality"] >= 0.5 and r["effect"] >= 0.5 and r["recolored"] < 0.5 for r in rows.values())
    # How much further the strength can be raised before the search characters stop holding together.
    ladder = []
    for path in (directory / "search").glob("*.json"):
        probe = read(path)
        values = [ws.quality(r["broken"], r["off_broken"], r["one_figure"], r["figure"]) for r in probe["rows"].values()]
        ladder.append((probe["alpha"], fmean(values)))
    holds = result["alpha"]
    for alpha, value in sorted(ladder):
        if alpha > holds:
            if value < 0.5:
                break
            holds = alpha
    # The same readings without the recolor term, for comparison. It did not choose the alpha.
    kept = summarize({c: ws.effect(r["boss"], r["original"]) * r["quality"] for c, r in rows.items()},
                     doc["pairs"])["score"]
    structure = read(directory / "structure.json") if (directory / "structure.json").exists() else None
    if structure and structure["alpha"] != result["alpha"]:
        structure = None
    entries.append(dict(
        name=directory.name, arm=arm, extra=int(extra), updates=updates, label=label, group=group,
        alpha=result["alpha"], score=100 * result["score"], search=100 * result["search_score"],
        gain=100 * doc["mean_gain"], quality=100 * fmean(r["quality"] for r in rows.values()),
        effect=100 * fmean(r["effect"] for r in rows.values()), holds=holds,
        drawn=100 * fmean(1 - min(1.0, max(0.0, r["broken"] - r["off_broken"]) / 0.30) for r in rows.values()),
        one_figure=100 * fmean(r["one_figure"] for r in rows.values()),
        figure=100 * fmean(r["figure"] for r in rows.values()),
        recolored=100 * fmean(r["recolored"] for r in rows.values()), kept=100 * kept,
        same_person=100 * fmean(r["same_person"] for r in rows.values()),
        repeat=100 * fmean(p["repeat"] for p in doc["pairs"]), working=working,
        structure=structure and {k: structure[k] for k in ("passed", "failed", "unresolved", "checks")},
        previous_alpha=result["previous_alpha"],
        previous_score=None if result["previous_score"] is None else 100 * result["previous_score"],
        trials=[[a, None if v is None else 100 * v] for a, v in result["trials"]], rows=rows))
entries.sort(key=lambda e: -e["score"])
old_order = sorted((e for e in entries if e["previous_score"] is not None), key=lambda e: -e["previous_score"])
names = [c.title() for c in ranking]
lines = ['<tr class="ref"><td>—</td><th scope="row">Matched baseline</th><td>—</td><td>—</td><td>OFF</td><td><div class="images">'
         + "".join(cell(cases[c]["off"], n) for c, n in zip(ranking, names)) + '</div></td><td colspan="18"></td></tr>']
for rank, e in enumerate(entries, 1):
    label = e["label"]
    images = "".join(
        cell(e["rows"][c]["on"],
             f'{n} · case {100 * e["rows"][c]["case"]:.0f} · effect {100 * e["rows"][c]["effect"]:.0f} (boss '
             f'{100 * e["rows"][c]["boss"]:.0f}) · quality '
             f'{100 * e["rows"][c]["quality"]:.0f} · whole figure {100 * e["rows"][c]["figure"]:.0f} · one figure '
             f'{100 * e["rows"][c]["one_figure"]:.0f} · recolored {100 * e["rows"][c]["recolored"]:.0f} · same person '
             f'{100 * e["rows"][c]["same_person"]:.0f}',
             "bad" if e["rows"][c]["quality"] < 0.25 or failed(e, c)
             else "ok" if e["rows"][c]["quality"] >= 0.5 and e["rows"][c]["effect"] >= 0.5
             and e["rows"][c]["recolored"] < 0.5 else "unsure")
        for c, n in zip(ranking, names))
    was = old_order.index(e) + 1 if e in old_order else "—"
    st = e["structure"]
    gate = "unmeasured" if not st else ", ".join(
        f"{st[k]} {label}" for k, label in (("passed", "pass"), ("failed", "fail"), ("unresolved", "unresolved")) if st[k])
    before = "—" if e["previous_score"] is None else f'{e["previous_score"]:.2f} at α {e["previous_alpha"]:g}'
    lines.append(
        f'<tr data-score="{e["score"]}" data-working="{e["working"] * 1000 + e["quality"]}" data-gain="{e["gain"]}" '
        f'data-quality="{e["quality"]}" data-kept="{e["kept"]}"><td>{rank}</td><th scope="row">{escape(label)}</th><td>{escape(e["group"])}</td><td>{e["updates"]}</td>'
        f'<td>{e["alpha"]:g}</td><td><div class="images">{images}</div></td><td><b>{e["score"]:.2f}</b></td>'
        f'<td>{e["working"]} of 8</td><td>{gate}</td><td>{e["effect"]:.0f}</td><td>{e["holds"]:g}</td><td>{e["gain"]:+.1f}</td><td>{e["quality"]:.0f}</td><td>{e["drawn"]:.0f}</td>'
        f'<td>{e["figure"]:.0f}</td><td>{e["one_figure"]:.0f}</td><td>{e["repeat"]:.0f}</td><td>{e["recolored"]:.0f}</td><td>{e["kept"]:.2f}</td><td>{e["same_person"]:.0f}</td>'
        f'<td>{e["search"]:.2f}</td><td>{len(e["trials"])}</td><td>{before}</td><td>{was}</td></tr>')
note = f"<p><b>Still running:</b> {escape(', '.join(pending))}.</p>" if pending else ""
shutil.copyfile(PARENT_STUDY / "sample-gallery/gallery.css", gallery / "gallery.css")
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Final Boss post-training · a boss that is not broken</title><link rel="stylesheet" href="gallery.css"><style>
main,header{{max-width:1540px}}.scroll{{overflow:auto;border:1px solid #41434a;border-radius:7px;margin-top:14px}}
.scroll table{{margin:0;width:100%;font-size:14px;border-collapse:collapse}}.scroll td,.scroll th{{padding:6px 8px;text-align:left}}
.scroll thead{{background:#292b30}}.scroll th[scope=row]{{min-width:170px;max-width:300px}}.images{{display:flex;gap:4px;width:max-content}}
.images a,.images img{{display:block}}.images img{{width:58px;height:58px;object-fit:cover;border-radius:3px}}
.images a.ok img{{outline:3px solid #6fd08c;outline-offset:-3px}}.images a.bad img{{outline:3px solid #ff5d5d;outline-offset:-3px}}
.images a.unsure img{{outline:2px dashed #e0b84c;outline-offset:-2px}}.scroll tr:hover{{background:#242529}}.scroll .ref{{background:#202328}}
</style></head><body><header><a href="post-training-leaderboard.html">Earlier board, distinct Boss score</a><nav><a href="board-leaderboard.html">Every trained recipe</a><a href="working-leaderboard.html">Post-training rows</a><a href="working-leaderboard-v1.html">Before the whole-figure term</a><a href="alpha-review.html">Per-strength review</a><a href="working-results.json">Scores and provenance</a></nav></header><main>
<h1>{"Every trained Final Boss recipe" if combined else "Final Boss post-training"} · a boss that is not broken</h1>
<p>Switched on, the slider should give a final boss that is not broken. A wilder boss is the user's choice of a higher strength. Each image scores effect times quality times how much of the original scene's color it kept. Effect rises from 0 for an ordinary character to 1 once the character reads as a distinctive endgame antagonist, and does not grow past that. Quality is how cleanly the image is drawn, read by Mercury Decide on four 256-pixel tiles at native resolution against the same character's OFF image, limited by two whole-image readings: exactly one figure, and a head on a body. Repeated designs across characters are discounted as before. Every RAW had its constant alpha chosen by this score on the four search characters, with ties to the lower alpha, and was then ranked on the eight held-out characters. An edit that washes the surroundings in one new color is marked down by how far it did. Looking like the original person is shown and not scored.</p>
<p>A green outline marks a working image: effect and quality 50 or more, and the scene less than half recolored. Red is quality under 25 or a failed structural check. The structural check is the existing pass or fail gate, read at the new alpha and reported beside the score. A failure keeps a RAW out of release whatever it scores. Hover an image for its readings.</p>{note}
<p><label>Order by <select id="metric"><option value="score">Working Boss score</option><option value="kept">Score without the recolor term</option><option value="working">Working images</option><option value="gain">Boss gain</option><option value="quality">Quality</option></select></label></p>
<div class="scroll"><table><thead><tr><th>Rank</th><th>Approach</th><th>Run</th><th>Updates</th><th>Constant α</th><th>Eight held-out characters</th><th>Working Boss / 100</th><th>Working</th><th>Structural check</th><th>Is a boss</th><th>Quality holds to α</th><th>Boss gain</th><th>Quality</th><th>Drawn</th><th>Whole figure</th><th>One figure</th><th>Repeat</th><th>Scene recolored</th><th>Score without the recolor term</th><th>Same person, not scored</th><th>Search score</th><th>Trials</th><th>Distinct Boss before</th><th>Rank before</th></tr></thead><tbody>{"".join(lines)}</tbody></table></div>
</main><script>
(()=>{{const body=document.querySelector('tbody'),rows=[...body.querySelectorAll('[data-score]')],pick=document.querySelector('#metric');
function refresh(){{const key=pick.value;[...rows].sort((a,b)=>Number(b.dataset[key])-Number(a.dataset[key])).forEach((r,i)=>{{body.append(r);r.firstChild.textContent=i+1}})}}
const wanted=new URLSearchParams(location.search).get('metric');if(['score','kept','working','gain','quality'].includes(wanted))pick.value=wanted;
pick.addEventListener('change',refresh);refresh()}})();</script></body></html>"""
(gallery / ("board-leaderboard.html" if combined else "working-leaderboard.html")).write_text(page)
for e in entries:
    e["rows"] = {c: {k: (v["sha256"] if k == "on" else v) for k, v in r.items()} for c, r in e["rows"].items()}
(gallery / ("board-results.json" if combined else "working-results.json")).write_text(json.dumps(dict(entries=entries, pending=pending), indent=1) + "\n")
print(f'{"rank":>4} {"was":>4}  {"row":<20}{"α":>9}{"was α":>9}{"score":>8}{"working":>9}{"gain":>7}{"quality":>9}{"repeat":>8}  structure')
for rank, e in enumerate(entries, 1):
    was = old_order.index(e) + 1 if e in old_order else 0
    print(f'{rank:>4} {was:>4}  {e["name"][:30]:<31}{e["alpha"]:>9.4g}{(e["previous_alpha"] or 0):>9.4g}{e["score"]:>8.2f}'
          f'{e["working"]:>7}/8{e["gain"]:>7.1f}{e["quality"]:>9.0f}{e["repeat"]:>8.0f}  recolor {e["recolored"]:>3.0f} kept {e["kept"]:>6.2f}  '
          + ("—" if not e["structure"] else f'{e["structure"]["passed"]}p {e["structure"]["failed"]}f {e["structure"]["unresolved"]}u'))
print("pending:", pending)
