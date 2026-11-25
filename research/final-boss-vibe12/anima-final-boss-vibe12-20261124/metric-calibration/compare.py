"""Publish the nominated examples beside the current measured winner."""
from html import escape


def comparison(ranked, previous, labels, cases, public, sample):
    keys = (('shared-best-opening','early',.84375), ('shared-best-opening','early',1.15625),
            ('soft45-background-orthogonal','early',1.0))
    key = lambda r:(r['method'],r['profile'],r['alpha'])
    known = {key(r):r for r in ranked}
    if any(wanted not in known for wanted in keys):
        return
    best_opening = next(r for r in ranked if r['method'] == 'shared-best-opening')
    selected = list(dict.fromkeys((*keys[:2], key(best_opening), key(ranked[0]), keys[2])))
    older = {key(r):r for r in previous['rows']}

    def image(asset, label):
        url = sample(asset, f'vibe12-alpha-{asset["sha256"]}.png')
        return f'<a href="{escape(url,quote=True)}"><img src="{escape(url,quote=True)}" alt="{escape(label,quote=True)}" loading="lazy" width="172" height="172"></a>'

    def metrics(now, old):
        line = f'Gain +{100*now["boss_effect"]:.1f} · Damage {100*now["image_breakage"]:.1f}<small>Final score {100*now["final_boss_score"]:.2f}</small>'
        if old is not None:
            line += f'<small>Previous damage {100*old["image_breakage"]:.1f}</small>'
        return line

    headings = ''.join(f'<th>{escape(labels[k[0]])}<small>{k[1].title()} α{k[2]:g}</small></th>' for k in selected)
    means = ''.join('<td>'+metrics(known[k]['means'], older[k]['means'] if k in older else None)+'</td>' for k in selected)
    lines = []
    for case in cases:
        off = known[selected[0]]['cases'][case]['off']
        fields = [f'<th scope="row">{case.title()}</th>', '<td>'+image(off, case+' original')+'</td>']
        for k in selected:
            r = known[k]['cases'][case]
            fields.append('<td>'+image(r['on'],case+' '+labels[k[0]])+metrics(r,older[k]['cases'][case] if k in older else None)+'</td>')
        lines.append('<tr>'+''.join(fields)+'</tr>')
    doc = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Opening four comparison</title>
<style>:root{{color-scheme:dark;font:15px/1.45 system-ui;background:#17191d;color:#f1eee9}}main{{max-width:1250px;margin:25px auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse}}td,th{{padding:6px 8px;text-align:left;vertical-align:top;border-bottom:1px solid #48494f}}thead th{{min-width:150px;background:#23262c}}thead th:first-child{{min-width:65px}}small{{display:block;color:#bdc1ca;font-weight:400}}img{{display:block;object-fit:contain}}a{{color:#ffc59f}}</style>
<main><h1>Opening four comparison</h1><p>Eight matched RAW192 cases. The current Jev question judges damage on the finished illustration alone. Previous damage used the original and the edit together. Click a sample to inspect the full drawing. <a href="final-boss-vibe12-alpha-leaderboard.html">Live leaderboard</a></p>
<div class="scroll"><table><thead><tr><th>Case</th><th>Slider off</th>{headings}</tr></thead><tbody><tr><th>Mean</th><td></td>{means}</tr>{''.join(lines)}</tbody></table></div></main></html>'''
    output = public / 'final-boss-vibe12-opening-comparison.html'
    pending = output.with_suffix('.publish')
    pending.write_text(doc)
    pending.replace(output)
