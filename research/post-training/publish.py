"""Publish the post-training leaderboard, its matched samples and the measured trade-off."""

from html import escape
from pathlib import Path
from statistics import fmean, stdev
import json
import shutil
import sys

from PIL import Image

from campaign import CODE, PARENT_STUDY, POLICY, ROOT

D = "distinct_boss_score"
budgets = [int(v) for v in sys.argv[1:]] or [POLICY["additional_updates"]]
parent = POLICY["parent"]
gallery = ROOT / "gallery"
(gallery / "images").mkdir(parents=True, exist_ok=True)
(gallery / "thumbs").mkdir(exist_ok=True)
panels = json.loads((PARENT_STUDY / "cohort-retest/panels.json").read_bytes())
cases = {c["id"]: c for c in panels["cases"]}
ranking = panels["panels"]["ranking"]
LABELS = {
    "parent": "Parent · no further training",
    "continued": "Exact continuation · parent critic and optimizer, same targets",
    "control": "Fork control · fresh critic and optimizer, same targets",
    "cleaner": "Fork · one cleaner teacher that keeps its Boss gain",
    "cleanest": "Fork · five least-damaged teachers, gain not protected",
    "control-seed2": "Fork control, second seed · same targets",
    "cleanest-seed2": "Fork, second seed · five least-damaged teachers",
    "control-warm": "Fork control with the parent's trained critic · same targets",
    "cleanest-warm": "Fork with the parent's trained critic · five least-damaged teachers",
    "control-low-g": "Fork control at the lower generator rate · same targets",
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


def strip(refs, labels):
    cells = []
    for ref, label in zip(refs, labels, strict=True):
        name = asset(ref)
        cells.append(
            f'<a href="images/{name}" title="{escape(label)}"><img loading="lazy" width="58" height="58" '
            f'alt="{escape(label)}" src="thumbs/{ref["sha256"]}.webp"></a>'
        )
    return '<div class="images">' + "".join(cells) + "</div>"


unmeasured = []


def measurement(root):
    result = read(root / "result.json")
    if result.get("alpha") is None:
        return None
    doc = read(root / "ranking" / (result["alpha"].hex() + ".json"))
    if doc["score"] is None:
        unmeasured.append(str(root.relative_to(ROOT)))
        return None
    rows = doc["rows"]
    trials = []
    for path in (root / "search").glob("*.json"):
        s = read(path)
        if s["score"] is None:
            continue
        trials.append(dict(alpha=s["identity"]["alpha"], score=100 * s["score"], gain=100 * s["mean_gain"],
                           damage=100 * fmean(r["image_breakage"] for r in s["rows"].values())))
    return dict(
        alpha=result["alpha"], score=100 * doc["score"], search=100 * result["search_score"],
        gain=100 * doc["mean_gain"], damage=100 * fmean(r["image_breakage"] for r in rows.values()),
        repeat=100 * doc["design"]["repeat"], integrity=result["integrity"], probes=len(result["trials"]),
        failed_checks=sorted(c for c, v in result["checks"].items() if v["status"] == "fail"),
        unresolved_checks=sorted(c for c, v in result["checks"].items() if v["status"] == "unmeasured"),
        cases={c: dict(gain=100 * (r["boss_distinction"] - r["original_boss_distinction"]),
                       damage=100 * r["image_breakage"], on=r["on"]) for c, r in rows.items()},
        trials=sorted(trials, key=lambda t: t["alpha"]),
    )


def trajectory(name, budget):
    arm = next((ROOT / "arms" / name / "rates").glob("*"))
    base = 0 if POLICY["arms"][name]["entry"] == "fork" else parent["step"]
    points = []
    for extra in sorted({budget * k // 4 for k in (1, 2, 3, 4)}):
        path = arm / "fit" / f"{base + extra}.json"
        if path.exists():
            row = read(path)
            score = row.get("objective_score")
            points.append(dict(extra=extra, score=None if score is None else 100 * score,
                               gain=row["mean_gain"], integrity=row["status"]))
    return points


def stopped(name):
    """A structural failure stops an arm before another update; its saved RAWs stay diagnostic."""
    path = next((ROOT / "arms" / name / "rates").glob("*")) / "failure.json"
    if not path.exists():
        return None
    failure = read(path)
    worst = max(p["on"]["integrity"]["checks"]["broken_structure"] for p in failure["result"]["pairs"])
    return dict(step=failure["step"], phase=failure["phase"], broken_structure=worst)


def checks(e):
    """Eight structural checks at the kept strength; any failure keeps a RAW out of release."""
    failed, unresolved = len(e["failed_checks"]), len(e["unresolved_checks"])
    parts = [f"{8 - failed - unresolved} pass"]
    if failed:
        parts.append(f"{failed} fail")
    if unresolved:
        parts.append(f"{unresolved} unresolved")
    return ", ".join(parts)


def status(e):
    if not e["extra"]:
        return "parent"
    if e["stopped"]:
        return f'stopped at +{e["stopped"]["step"]} · structural failure'
    return "completed"


entries = [dict(arm="parent", updates=parent["step"], extra=0, stopped=None,
                **measurement(PARENT_STUDY / parent["calibration"] / D))]
for budget in budgets:
    for name in POLICY["arms"]:
        root = ROOT / "arms" / name / "calibration" / str(budget) / D
        if (root / "result.json").exists() and (m := measurement(root)):
            entries.append(dict(arm=name, updates=parent["step"] + budget, extra=budget, stopped=stopped(name),
                                trajectory=trajectory(name, budget), **m))
base = entries[0]
for e in entries:
    diffs = {k: [e["cases"][c][k] - base["cases"][c][k] for c in ranking] for k in ("gain", "damage")}
    e["paired"] = {k: dict(mean=fmean(v), se=stdev(v) / len(v) ** 0.5) for k, v in diffs.items()}
    # Effect matched: least damaged searched strength within two gain points of the parent's choice.
    target = next(t for t in base["trials"] if t["alpha"] == base["alpha"])
    near = [t for t in e["trials"] if abs(t["gain"] - target["gain"]) <= 2]
    e["matched"] = min(near, key=lambda t: t["damage"]) if near else None
entries.sort(key=lambda e: -e["score"])
controls = [e for e in entries if e["arm"] in ("parent", "continued", "control", "control-low-g", "control-seed2", "control-warm")]
win = POLICY["practical_win"]
for e in entries:
    if e["arm"].startswith("clean"):
        rivals = [c for c in controls if c["extra"] in (0, e["extra"])]
        e["practical_win"] = not e["stopped"] and all(
            e["gain"] - c["gain"] >= win["gain_points"] and e["damage"] - c["damage"] <= win["damage_points"]
            and e["score"] > c["score"] for c in rivals)
        e["pareto"] = all(e["gain"] >= c["gain"] and e["damage"] < c["damage"] for c in rivals)

names = [c.title() for c in ranking]
rows = [
    '<tr class="ref"><td>—</td><th scope="row">Matched baseline</th><td>—</td><td>OFF</td><td>'
    + strip([cases[c]["off"] for c in ranking], names) + '</td><td colspan="10"></td></tr>'
]
for rank, e in enumerate(entries, 1):
    label = LABELS[e["arm"]] + (f" · +{e['extra']}" if e["extra"] else "")
    rows.append(
        f'<tr data-design="{e["score"]}" data-gain="{e["gain"]}" data-damage="{-e["damage"]}">'
        f'<td>{rank}</td><th scope="row">{escape(label)}</th><td>{e["updates"]}</td><td>{e["alpha"]:g}</td>'
        f'<td>{strip([e["cases"][c]["on"] for c in ranking], names)}</td>'
        f'<td><b>{e["score"]:.2f}</b></td><td>{e["gain"]:+.1f}</td><td>{e["damage"]:.1f}</td>'
        f'<td>{e["repeat"]:.1f}</td><td>{checks(e)}</td><td>{status(e)}</td><td>{e["search"]:.2f}</td><td>{e["probes"]}</td>'
        f'<td>{e["score"] - base["score"]:+.2f}</td>'
        f'<td>{e["paired"]["gain"]["mean"]:+.1f} ± {e["paired"]["gain"]["se"]:.1f} / '
        f'{e["paired"]["damage"]["mean"]:+.1f} ± {e["paired"]["damage"]["se"]:.1f}</td></tr>'
    )
colors = {"parent": "#f4f0e7", "continued": "#7fb7ff", "control": "#ffb454", "cleaner": "#8fe388", "cleanest": "#ff7fbf", "control-low-g": "#c9a36b", "control-warm": "#9ad0c2", "cleanest-warm": "#d6e36b", "control-seed2": "#ffd9a0", "cleanest-seed2": "#ffb8dc", "cleanest-low-g": "#c58fd6"}
points = [t for e in entries for t in e["trials"]]
x0, x1 = min(t["damage"] for t in points) - 1, max(t["damage"] for t in points) + 1
y0, y1 = min(t["gain"] for t in points) - 2, max(t["gain"] for t in points) + 2
sx = lambda v: 50 + 560 * (v - x0) / (x1 - x0)
sy = lambda v: 330 - 300 * (v - y0) / (y1 - y0)
marks = []
for e in entries:
    line = " ".join(f"{sx(t['damage']):.1f},{sy(t['gain']):.1f}" for t in e["trials"])
    dash = ' stroke-dasharray="4 3"' if e["extra"] > budgets[0] else ""
    marks.append(f'<polyline points="{line}" fill="none" stroke="{colors[e["arm"]]}" stroke-width="1.5" opacity=".75"{dash}/>')
    for t in e["trials"]:
        chosen = t["alpha"] == e["alpha"]
        marks.append(
            f'<circle cx="{sx(t["damage"]):.1f}" cy="{sy(t["gain"]):.1f}" r="{6 if chosen else 3}" '
            f'fill="{colors[e["arm"]]}"><title>{LABELS[e["arm"]]} +{e["extra"]} · α {t["alpha"]:g} · '
            f'gain {t["gain"]:+.1f} · damage {t["damage"]:.1f} · score {t["score"]:.2f}</title></circle>')
ticks = "".join(
    f'<text x="{sx(v):.0f}" y="348" text-anchor="middle">{v:.0f}</text>' for v in range(int(x0) + 1, int(x1) + 1, 2)
) + "".join(f'<text x="42" y="{sy(v) + 4:.0f}" text-anchor="end">{v:.0f}</text>' for v in range(int(y0 // 10 * 10 + 10), int(y1), 10))
legend = " ".join(f'<span style="color:{colors[k]}">● {escape(LABELS[k].split(" · ")[0])}</span>' for k in colors)
chart = (f'<svg viewBox="0 0 630 370" role="img" aria-label="Boss gain against visible damage across searched strengths" '
         f'style="max-width:760px;width:100%;font:11px system-ui;fill:#c9cac2"><rect x="50" y="30" width="560" height="300" '
         f'fill="#1e1f22" stroke="#41434a"/>{"".join(marks)}{ticks}<text x="330" y="366" text-anchor="middle">Visible damage / 100 on the four search characters</text>'
         f'<text x="14" y="180" transform="rotate(-90 14 180)" text-anchor="middle">Boss gain</text></svg>')
matched = "".join(
    f'<tr><th scope="row">{escape(LABELS[e["arm"]])}{" · +" + str(e["extra"]) if e["extra"] else ""}</th>'
    + (f'<td>{e["matched"]["alpha"]:g}</td><td>{e["matched"]["gain"]:+.1f}</td><td>{e["matched"]["damage"]:.1f}</td>'
       f'<td>{e["matched"]["score"]:.2f}</td>' if e["matched"] else '<td colspan="4">no searched strength in the band</td>')
    + "</tr>" for e in entries)
steps = sorted({p["extra"] for e in entries for p in e.get("trajectory", [])})
progress = "".join(
    f'<tr><th scope="row">{escape(LABELS[e["arm"]])} · +{e["extra"]}</th>'
    + "".join(next((f'<td>{p["score"]:.2f} · {p["integrity"]}</td>' if p["score"] is not None else f'<td>unmeasured · {p["integrity"]}</td>'
                    for p in e["trajectory"] if p["extra"] == s), "<td>—</td>") for s in steps)
    + "</tr>" for e in entries if e.get("trajectory"))
current = read(ROOT / "teachers/current.json")
selections = {arm: read(ROOT / f"teachers/selection-{arm}.json") for arm in POLICY["teachers"]["rules"]}
teacher_rows = []
for scene in current["scenes"]:
    c = scene["current"]
    picked = {arm: s["decisions"].get(scene["scene"], {}).get("selected") for arm, s in selections.items()}
    options = selections["cleaner"]["decisions"].get(scene["scene"], {}).get("proposals", [])
    cells = [f'<figure><a href="images/{asset(scene["neutral_image"])}"><img loading="lazy" width="116" height="116" src="thumbs/{scene["neutral_image"]["sha256"]}.webp" alt="neutral"></a><figcaption>neutral</figcaption></figure>',
             f'<figure><a href="images/{asset(c["image"])}"><img loading="lazy" width="116" height="116" src="thumbs/{c["image"]["sha256"]}.webp" alt="current teacher"></a><figcaption>current · gain {100 * c["gain"]:+.0f} · damage {100 * c["damage"]:.1f}{" · separate figure" if c["separate"] == "fail" else ""}</figcaption></figure>']
    for o in options:
        tags = [arm for arm, name in picked.items() if name == o["name"]]
        cells.append(f'<figure><a href="images/{asset(o["image"])}"><img loading="lazy" width="116" height="116" src="thumbs/{o["image"]["sha256"]}.webp" alt="{escape(o["name"])}"></a><figcaption>{escape(o["name"])} · gain {100 * o["gain"]:+.0f} · damage {100 * o["damage"]:.1f}{" · <b>" + " + ".join(tags) + "</b>" if tags else ""}</figcaption></figure>')
    teacher_rows.append(f'<div class="teachers">{"".join(cells)}</div>')
verdicts = "".join(
    f'<li>{escape(LABELS[e["arm"]])} · +{e["extra"]}: {"practical win" if e["practical_win"] else "no practical win"}'
    f'{" and a strict Pareto improvement" if e["pareto"] else ""} against the parent and both equal-update controls.</li>'
    for e in entries if "practical_win" in e)
parity = {name: read(ROOT / "arms" / name / "fork-parity.json") for name in POLICY["arms"]
          if (ROOT / "arms" / name / "fork-parity.json").exists()}
shutil.copyfile(PARENT_STUDY / "sample-gallery/gallery.css", gallery / "gallery.css")
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Final Boss post-training on the 512-update checkpoint</title><link rel="stylesheet" href="gallery.css"><style>
main,header{{max-width:1540px}} .scroll{{overflow:auto;border:1px solid #41434a;border-radius:7px;max-width:100%;margin-top:14px}}
.scroll table{{margin:0;width:100%;font-size:14px;border-collapse:collapse}}.scroll td,.scroll th{{padding:6px 8px;text-align:left}}
.scroll thead{{background:#292b30}}.scroll th[scope=row]{{min-width:170px;max-width:300px}}
.images{{display:flex;gap:4px;width:max-content}}.images a,.images img{{display:block}}
.images img{{width:58px;height:58px;object-fit:cover;border-radius:3px}}.scroll tr:hover{{background:#242529}}.scroll .ref{{background:#202328}}
.teachers{{display:flex;gap:8px;overflow:auto;margin:10px 0 18px}}.teachers figure{{width:116px;flex:none}}.teachers img{{width:116px;height:116px}}
.teachers figcaption{{font-size:.74rem;line-height:1.25}}.controls{{margin-top:14px}}
</style></head><body><header><a href="http://100.90.104.57:8796/constant-retest.html">Final Boss constant-alpha board</a><nav><a href="results.json">Scores and provenance</a></nav></header><main>
<h1>Final Boss post-training on the 512-update checkpoint</h1>
<p>Every row starts from the same PG 0.12 Final Boss RAW512 (<code>{parent["raw_sha256"][:12]}</code>). Each trained arm received {" and ".join("+" + str(b) for b in budgets)} stock EMPF updates at the parent's rates. Each RAW then had its own constant-alpha search on four characters. That alpha was frozen before the eight ranking characters shown here. Scores are distinct Boss points out of 100, the same measurement as the constant-alpha board.</p>
<ul>{verdicts}</ul>
<p class="controls"><label>Order by <select id="metric"><option value="design">Distinct Boss score</option><option value="gain">Boss gain</option><option value="damage">Least visible damage</option></select></label></p>
<div class="scroll"><table><thead><tr><th>Rank</th><th>Approach</th><th>Updates</th><th>Constant α</th><th>Matched samples</th><th>Distinct Boss / 100</th><th>Boss gain</th><th>Damage</th><th>Repeat</th><th>Integrity</th><th>Training</th><th>Search score</th><th>Trials</th><th>vs parent</th><th>Paired gain / damage vs parent</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p><small>Boss gain is signed ON minus OFF distinction. Damage is Jev's visible-damage severity of the ON image. Repeat is the mean repeated-design judgment over all 28 character pairs in both orders. Paired columns average each character's change from the parent, ± one standard error across the eight characters.</small></p>
<h2>Gain against damage at every searched strength</h2>
<p>Each line is one RAW across its searched alphas on the four search characters. The large dot is the alpha it kept. Post-training helps only where its line sits left of the parent's at the same height.</p>
{chart}<p>{legend}</p>
<div class="scroll"><table><thead><tr><th>Least damaged strength within ±2 gain points of the parent's choice</th><th>α</th><th>Boss gain</th><th>Damage</th><th>Search score</th></tr></thead><tbody>{matched}</tbody></table></div>
<h2>During training, at α 1</h2>
<p>Distinct Boss score and structural check on the four training-selection characters at each quarter of the added budget.</p>
<div class="scroll"><table><thead><tr><th>Arm</th>{"".join(f"<th>+{s}</th>" for s in steps)}</tr></thead><tbody>{progress}</tbody></table></div>
<h2>Teacher changes</h2>
<p>The six most damaged of the twelve Boss teachers received three proposals each, rendered from the same neutral prompt and seed. Gain and damage use the same Jev questions as the board. Bold names were selected for an arm.</p>
{"".join(teacher_rows)}
<h2>Fork checks</h2>
<p>{len(parity)} forked children reproduced the parent's eight ranking images at α {parent["alpha"]:g} and all eight OFF images byte for byte before their first update. Each child began with all 4,033 parent tensors, the seeded fresh critic and fresh optimizer state. The exact continuation resumed the parent's own FULL512.</p>
</main><script>
(()=>{{const body=document.querySelector('tbody'),rows=[...body.querySelectorAll('[data-design]')],pick=document.querySelector('#metric');
function refresh(){{const key=pick.value;[...rows].sort((a,b)=>Number(b.dataset[key])-Number(a.dataset[key])).forEach((r,i)=>{{body.append(r);r.firstChild.textContent=i+1}})}}
const wanted=new URLSearchParams(location.search).get('metric');if(['design','gain','damage'].includes(wanted))pick.value=wanted;
pick.addEventListener('change',refresh);refresh()}})();</script></body></html>"""
(gallery / "post-training-leaderboard.html").write_text(page)
# One strip of the eight ranking characters per row for the committed report.
samples = CODE.parents[1] / "docs/post-training/samples"
samples.mkdir(parents=True, exist_ok=True)


def sheet(refs, name):
    strip_ = Image.new("RGB", (8 * 160, 160))
    for index, ref in enumerate(refs):
        with Image.open(ref["path"]) as image:
            strip_.paste(image.convert("RGB").resize((160, 160), Image.Resampling.LANCZOS), (index * 160, 0))
    strip_.save(samples / name, quality=80)


sheet([cases[c]["off"] for c in ranking], "off.webp")
table = ["| Rank | Approach | Updates | Constant α | Eight ranking characters | Distinct Boss / 100 | Boss gain | Damage | Repeat | Integrity | Training |",
         "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
         "| — | Matched baseline | — | OFF | ![OFF](samples/off.webp) | — | — | — | — | — | — |"]
for rank, e in enumerate(entries, 1):
    name = f'{e["arm"]}-{e["extra"]}.webp'
    sheet([e["cases"][c]["on"] for c in ranking], name)
    label = LABELS[e["arm"]] + (f" · +{e['extra']}" if e["extra"] else "")
    table.append(f'| {rank} | {label} | {e["updates"]} | {e["alpha"]:g} | ![{label}](samples/{name}) | '
                 f'{e["score"]:.2f} | {e["gain"]:+.1f} | {e["damage"]:.1f} | {e["repeat"]:.1f} | {checks(e)} | {status(e)} |')
(samples.parent / "leaderboard-table.md").write_text("\n".join(table) + "\n")
for e in entries:
    for c in e["cases"].values():
        c["on"] = c["on"]["sha256"]
(gallery / "results.json").write_text(json.dumps(dict(policy=POLICY, entries=entries, fork_parity=parity,
    teachers=dict(current=current, selections=selections)), indent=1) + "\n")
print("Unmeasured rankings:", unmeasured)
print("Published", len(entries), "rows:", [(e["arm"], e["extra"], round(e["score"], 2), round(e["gain"], 1),
                                            round(e["damage"], 1), checks(e)) for e in entries])
