"""Publish the six highest point-score recipe winners after blinded round-robin comparison."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from statistics import fmean
from time import sleep

from case_sensitivity import rank_scores_by_omission
from preference_finish import HERE, FINISH
from preference_round import ROOT, STATE, harness, run

BOARD = HERE.parent / "finalists"
PAGE = "final-boss-vibe12-preference-review.html"
CALIBRATION = "https://github.com/HyperGAN/anima-particle-sliders/blob/main/docs/final-boss-raw192/vibe12/finalist-review.md"
PROTOCOL = ("Select the six highest-scoring distinct RAW192 recipes at their best measured strength. Compare every "
            "pair on all eight cases, both image orders. For each recipe, average the signed preference margin equally "
            "over cases and opponents. Zero is a tie. Omission ranks keep the same finalists and strengths.")


def standings(report, cases):
    margins = defaultdict(lambda: defaultdict(list))
    for pair in report["comparisons"]:
        for c,v in pair["cases"].items():
            margins[pair["a"]][c].append(v["margin"])
            margins[pair["b"]][c].append(-v["margin"])
    n = len(report["settings"])
    if len(margins) != n or any(set(rows) != set(cases) or any(len(v) != n-1 for v in rows.values())
                                 for rows in margins.values()):
        raise ValueError("Incomplete or unmatched finalist comparisons")
    values = {m:{c:fmean(v) for c,v in rows.items()} for m,rows in margins.items()}
    overall = {m:fmean(rows.values()) for m,rows in values.items()}
    return sorted(overall, key=lambda m:(-overall[m],m)), overall, values


def publish(panels, report, cases, signature):
    from publish_optimization import PUBLIC, sample
    order, overall, values = standings(report, cases)
    omissions = rank_scores_by_omission(values, cases)
    labels = harness.read(ROOT / "recipes.json")
    current = panels["groups"]["current"]
    def images(method):
        refs = [current[method or order[0]][c] for c in cases]
        assets = [r["on" if method else "off"] for r in refs]
        photos = []
        for c,img in zip(cases,assets):
            url = escape(sample(img, f'vibe12-alpha-{img["sha256"]}.png'),quote=True)
            photos.append(f'<a href="{url}"><img src="{url}" width="86" height="86" alt="{c}" title="{c.title()}"></a>')
        return '<div class="images">' + ''.join(photos) + '</div>'

    rows = []
    for place,m in enumerate(order,1):
        s = report["settings"][m]
        places = omissions[m]
        ranks = "".join(f"<small>Omit {c.title()} · #{place}</small>" for c,place in places.items())
        low,high = min(places.values()),max(places.values())
        span = str(low) if low == high else f"{low}–{high}"
        rows.append(f'<tr><th>{place}. {escape(labels[m])}<small>{s["profile"].title()} α {s["alpha"]:g}</small></th>'
                    f'<td>{images(m)}</td><td>{100*overall[m]:+.1f}<details><summary>7-case rank {span}</summary>{ranks}</details></td><td>{100*s["score"]:.2f}</td></tr>')
    pairs = []
    for p in report["comparisons"]:
        line = ''.join(f'<td>{100*p["cases"][c]["margin"]:+.1f}{"*" if not p["cases"][c]["both_orders_agree"] else ""}</td>' for c in cases)
        pairs.append(f'<tr><th>{escape(labels[p["a"]])}<small>vs. {escape(labels[p["b"]])}</small></th>{line}<td>{100*p["margin"]:+.1f}</td></tr>')
    columns = ''.join(f'<th>{c.title()}</th>' for c in cases)
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Final Boss finalist comparisons</title><style>:root{{color-scheme:dark;background:#17191d;color:#eee;font:15px/1.45 system-ui}}
main{{max-width:1440px;margin:24px auto;padding:0 14px}}a{{color:#ffc59f}}h1,h2{{margin-bottom:6px}}p{{max-width:960px}}
.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse}}td,th{{padding:7px;text-align:left;border-bottom:1px solid #414650}}
small{{display:block;font-weight:400;color:#bac0c8}}.images{{display:flex;gap:3px}}img{{display:block;border-radius:3px;object-fit:contain}}</style>
<main><a href="final-boss-vibe12-alpha-leaderboard.html">All recipes and strengths</a><h1>Which finished Final Boss edit would Jev keep?</h1>
<p>{now}. Each of the six finalists faced the other five on eight matched illustrations, with the images presented in both orders.
The preference margin ranges from −100 to +100. A positive value means Jev favored the edit on average.
The all-recipe score selects the six finalists and their strengths. <a href="{CALIBRATION}">Calibration and controls</a></p>
<div class="scroll"><table><thead><tr><th>Finalist</th><th>Matched samples</th><th>Preference margin</th><th>All-recipe score</th></tr></thead>
<tbody><tr><th>Slider off</th><td>{images(None)}</td><td></td><td></td></tr>{''.join(rows)}</tbody></table></div>
<h2>Head to head</h2><p>A positive margin favors the first recipe in each row. An asterisk means at least one presentation order gave a tie or favored the other edit.</p>
<div class="scroll"><table><thead><tr><th>Comparison</th>{columns}<th>Mean</th></tr></thead><tbody>{''.join(pairs)}</tbody></table></div></main></html>'''
    tmp = PUBLIC / (PAGE + '.publish')
    tmp.write_text(page)
    tmp.replace(PUBLIC / PAGE)
    harness.write(BOARD / "latest.json", dict(report, signature=signature, as_of=now, order=order,
                                              margins=overall, case_margins=values, omission_ranks=omissions))


def refresh():
    snapshot = (ROOT / "storage/alpha-standings.json").read_bytes()
    data = json.loads(snapshot)
    rows = [r for r in data["rows"] if r["best_for_recipe"]][:6]
    if len(rows) != 6:
        raise ValueError("Six complete recipe winners are required")
    settings = {r["method"]: dict(profile=r["profile"], alpha=r["alpha"], score=r["means"]["final_boss_score"]) for r in rows}
    identity = {"settings":settings, "original_and_edit":{r["method"]:[(c,r["cases"][c]["off"]["sha256"],r["cases"][c]["on"]["sha256"])
                        for c in data["cases"]] for r in rows}, "state":STATE,"questions":FINISH, "protocol":PROTOCOL}
    signature = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    latest = BOARD / "latest.json"
    if latest.is_file() and harness.read(latest).get("signature") == signature and "omission_ranks" in harness.read(latest):
        return
    panels = {"snapshot_sha256": sha256(snapshot).hexdigest(),"settings":settings,"groups":{"current":{r["method"]:r["cases"] for r in rows}}}
    target = BOARD / signature
    run(panels, target, STATE, FINISH, PROTOCOL, receipts=HERE / "receipts")
    publish(panels, harness.read(target / "report.json"), data["cases"], signature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        refresh()
        if not args.watch:
            break
        sleep(60)
