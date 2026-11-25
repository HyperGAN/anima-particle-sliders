"""Rank measured native strengths on the same eight Final Boss images."""

from html import escape
from pathlib import Path
from statistics import fmean
from time import sleep
from urllib.request import urlopen
import argparse
import sys

SEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SEARCH.parent))

from control import ROOT, STORAGE
from publish import BOSS, DAMAGE, GAIN, ORIGINAL, SCORE, VERSION as LEGACY_VERSION, score_case as legacy_score
sys.path.insert(0, str(SEARCH.parent / 'metric-calibration'))
from review import saved, BINDING, RATINGS
from finished_image import VERSION, score_case
from compare import comparison
from case_sensitivity import ranks_by_omission
import design_selection
from optimization_worker import PRIOR, pin, read, write
from publish_optimization import PUBLIC, sample


OUTPUT = 'final-boss-vibe12-alpha-leaderboard.html'
LABELS = read(ROOT / 'recipes.json')
FIELDS = (GAIN, BOSS, DAMAGE, SCORE)
ADAPTIVE = 'leaderboard-retune-visible'
SEARCH_ROOTS = (ADAPTIVE, design_selection.RUN)
CONDITIONS = ('matched_constant', 'fixed_strength')


def panel(method, setting, cases, source, run):
    if not all(row.get(metric) is not None for row in cases.values() for metric in FIELDS):
        return None
    for row in cases.values():
        if (row[SCORE] != score_case(row) or row[GAIN] != max(0, row[BOSS] - row[ORIGINAL])):
            raise ValueError('A row differs from its original Final Boss scores')
    return {'method': method, 'profile': setting['profile'], 'alpha': setting['alpha'],
            'gains': setting['effective_strengths'], 'status': 'measured', 'cases': cases,
            'means': {metric: fmean(row[metric] for row in cases.values()) for metric in FIELDS},
            'source': pin(source), 'runs': [run]}


def previous_rows(snapshot):
    for row in snapshot['rows']:
        cases = {}
        for case, old in row['cases'].items():
            if old[SCORE] != legacy_score(old):
                raise ValueError('The previous measurements no longer reproduce')
            damage = saved(old['on']['sha256'])
            if damage is None:
                break
            current = old | {DAMAGE: damage}
            cases[case] = current | {SCORE: score_case(current)}
        if len(cases) == len(snapshot['cases']):
            yield row | {'cases': cases, 'means': {metric: fmean(case[metric] for case in cases.values())
                                                   for metric in FIELDS}}


def search_rows(standings, originals):
    for name in SEARCH_ROOTS:
        for request in sorted((SEARCH / name).glob('*/request.json')):
            method = request.parent.name
            policy = read(request)
            report = ROOT / 'tuning' / method / 'boss-effect-tuning.json'
            if policy.get('quality_metric') != DAMAGE or policy.get('jev_binding') != BINDING:
                raise ValueError('Damage-only search used a different quality metric')
            if policy['leaderboard'] != pin(report) or policy['scoring_version'] != VERSION:
                raise ValueError('Tuned candidates differ from their native evaluation')
            raw = read(report)['raw']['sha256']
            if read(Path(policy['definition']['path']))['checkpoint']['sha256'] != raw:
                raise ValueError('Tuned candidates use a different RAW adapter')
            for path in sorted(request.parent.glob('renders/*/*/validation.json')):
                trial = read(path)
                cases, setting = trial['rows'], trial['setting']
                if list(cases) != standings['cases'] or setting['cases'] != standings['cases']:
                    raise ValueError('Tuned candidates use a different comparison panel')
                for case, row in cases.items():
                    if row['off']['sha256'] != originals[case]['sha256']:
                        raise ValueError('Tuned sample has a different slider-off image')
                    render = read(Path(row['on']['path']).with_name('render.json'))
                    if render['gains'] != setting['effective_strengths'] or render['image'] != row['on']:
                        raise ValueError('Tuned sample differs from the rendered alpha')
                measured = panel(method, setting, cases, path, name)
                if measured is not None:
                    yield measured


def rankings(standings):
    originals = {case: read(PRIOR / 'shared-off' / case / 'render.json')['image'] for case in standings['cases']}
    candidates = {}
    for row in (*previous_rows(standings), *search_rows(standings, originals)):
        key = row['method'], row['profile'], row['alpha']
        previous = candidates.get(key)
        if previous is not None and any(previous['cases'][case]['on']['sha256'] != row['cases'][case]['on']['sha256']
                                         for case in standings['cases']):
            raise ValueError(f'Two alpha receipts disagree on the matched samples for {key}')
        if previous is not None:
            row['runs'] = list(dict.fromkeys((*previous['runs'], *row['runs'])))
        candidates[key] = row
    ranked = sorted(candidates.values(), key=lambda row: (-row['means'][SCORE],
                                                           row['method'], row['profile'], row['alpha']))
    tops = set()
    for row in ranked:
        method = row['method']
        row['best_for_recipe'] = method not in tops
        if row['best_for_recipe']:
            tops.add(method)
    return originals, ranked


def images(cases, values, original=False):
    photos = []
    for case in cases:
        row = values[case]
        record = row if original else row['on']
        name = case.title()
        title = name if original else (f'{name} · Final Boss score {100*row[SCORE]:.2f} · '
                                       f'gain +{100*row[GAIN]:.1f} · damage {100*row[DAMAGE]:.1f}')
        path = sample(record, f'vibe12-alpha-{record["sha256"]}.png')
        photos.append(f'<a href="{path}" title="{title}"><img src="{path}" width="58" height="58" '
                      f'loading="lazy" alt="{name}"></a>')
    return '<div class="images">' + ''.join(photos) + '</div>'


def publish():
    standings = read(STORAGE / 'alpha-standings-v1.json')
    if standings['scoring_version'] != LEGACY_VERSION:
        raise ValueError('Unexpected leaderboard score')
    originals, ranked = rankings(standings)
    cases = standings['cases']
    measured = {(r['method'], r['profile'], r['alpha']) for r in ranked}
    queued = sum((r['method'],r['profile'],r['alpha']) not in measured for r in standings['rows'])
    if queued and not (STORAGE / 'visible-damage-published.json').exists():
        print(f'Calibrating {queued} remaining eight-case settings before switching the board', flush=True)
        return
    lines = []
    queue_path = SEARCH / ADAPTIVE / 'queue.json'
    progress = read(queue_path) if queue_path.is_file() else {'trained': [], 'complete': [], 'active': {}}
    completed = set(progress['complete'])
    active = set(progress['active'].values())
    winners = [r for r in ranked if r['best_for_recipe']]
    omissions = ranks_by_omission(winners, cases)
    design_file = ROOT / "metric-calibration/design-range/latest.json"
    design = read(design_file) if design_file.is_file() else {}
    audited = {(r['method'], r['profile'], r['alpha']): r for r in design.get('rows', [])}
    selected = design_selection.selections(SEARCH/design_selection.RUN, cases)
    audited = {k:r for k,r in audited.items() if k[0] not in selected}
    for method,(tuning,trial) in selected.items():
        audited[method,tuning['profile'],tuning['alpha']] = dict(trial['design'],pairs=trial['pairs'],
                method=method,profile=tuning['profile'],alpha=tuning['alpha'],
                inputs={c:r['on']['sha256'] for c,r in trial['rows'].items()},
                case_scores={c:r[SCORE] for c,r in trial['rows'].items()})
    design_selection.publish_paths(SEARCH/design_selection.RUN, PUBLIC, LABELS, selected)
    design_option = '<option value="design">Distinct Final Boss designs</option>' if audited else ''
    design_help = (f'<p><a href="?metric=design">Distinct Boss ranking</a> · <a href="?metric=score">Per-image ranking</a> · '
                   f'{len(selected)} of {len(audited)} displayed recipes have completed distinct-design alpha tuning. '
                   'Other recipes show their measured per-image strength. <a href="final-boss-vibe12-design-alpha.html">Distinct-design search paths</a>. '
                   'Repeated designs across different characters share credit. Jev compares all 28 character pairs.</p>'
                   if audited else '')
    for rank, row in enumerate(ranked, 1):
        method, profile, alpha = row['method'], row['profile'], row['alpha']
        means = row['means']
        control = (profile, alpha) in (('constant', .6375), ('early', 2 / 3))
        candidate = ADAPTIVE in row['runs'] and not control
        shared = 'fixed' in row['runs'] or ADAPTIVE in row['runs'] and control
        source = 'Adaptive search' if candidate else 'Fixed strength' if shared else 'Earlier search'
        if candidate and method not in completed:
            source += ' · in progress'
        note = f'<small>{source}</small>'
        design_row = audited.get((method,profile,alpha))
        design_data, design_cell = '', '<td class="design"></td>'
        if design_row:
            if any(row['cases'][c]['on']['sha256'] != design_row['inputs'][c] or
                   row['cases'][c][SCORE] != design_row['case_scores'][c] for c in cases):
                raise ValueError("The design comparison does not bind this measured setting")
            if method in selected:
                note = f'<small><a href="final-boss-vibe12-design-alpha.html#{method}">Design-tuned strength</a></small>'
            design_data = f' data-design="{design_row["score"]}"'
            def pair_label(pair):
                return ' / '.join(c.title() for c in pair["cases"]) + f' · {100*pair["repeat"]:.0f}/100'
            ordered = sorted(design_row['pairs'], key=lambda p:p['repeat'])
            design_cell = (f'<td class="design">{100*design_row["score"]:.2f}<details><summary>'
                           f'Repeat {100*design_row["repeat"]:.1f}/100</summary><small>Most distinct · {pair_label(ordered[0])}</small>'
                           f'<small>Most similar · {pair_label(ordered[-1])}</small></details></td>')
        sensitivity = ''
        if row['best_for_recipe']:
            places = omissions[method]
            low, high = min(places.values()), max(places.values())
            span = str(low) if low == high else f'{low}–{high}'
            details = ''.join(f'<small>Omit {c.title()} · #{place}</small>' for c,place in places.items())
            sensitivity = f'<details><summary>7-case rank {span}</summary>{details}</details>'
        lines.append(f'<tr data-effect="{means[GAIN]}" data-score="{means[SCORE]}" data-damage="{means[DAMAGE]}" '
                     f'{design_data} data-method="{method}" data-adaptive="{int(candidate)}" '
                     f'data-fixed="{int(shared)}" data-control="{profile if control else ""}" id="{method}-{profile}-{alpha:g}">'
                     f'<td class="rank">{rank}</td><th scope="row">{escape(LABELS[method])}{note}</th>'
                     f'<td>{profile.title()} α {alpha:.5g}</td><td>{images(cases, row["cases"])}</td>'
                     f'{design_cell}<td class="effect">+{100 * means[GAIN]:.1f}</td><td>{100 * means[SCORE]:.2f}{sensitivity}</td>'
                     f'<td>{100 * means[DAMAGE]:.1f}</td></tr>')

    reference = next((r for r in ranked if (r['method'],r['profile'],r['alpha']) ==
                      ('shared-best-opening','early',1.15625)), None)
    reference_html = ''
    if reference:
        scores = reference['means']
        reference_html = (f'<tr><td>Reference</td><th scope="row">Opening four<small>Visual comparison</small></th>'
                          f'<td>Early α 1.15625</td><td>{images(cases,reference["cases"])}</td>'
                          f'<td class="design"></td><td>+{100*scores[GAIN]:.1f}</td><td>{100*scores[SCORE]:.2f}</td>'
                          f'<td>{100*scores[DAMAGE]:.1f}</td></tr>')
    comparison(ranked, standings, LABELS, cases, PUBLIC, sample)

    page = f'''<!doctype html><html lang="en"><meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" /><title>Final Boss alpha leaderboard</title>
<style>
:root{{color-scheme:dark;background:#17191d;color:#f1eee9;font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1540px;margin:30px auto;padding:0 16px}}h1{{margin:0 0 4px;font-size:1.8rem}}
p{{margin:5px 0 12px;color:#c4c8d1}}a,.effect{{color:#ffc59f}}.effect{{font-weight:650}}
.controls{{display:flex;gap:16px;flex-wrap:wrap;margin:14px 0}}select{{margin-left:5px;padding:3px 6px}}
.scroll{{overflow:auto;border:1px solid #3a3e49;border-radius:7px}}table{{width:100%;border-collapse:collapse;text-align:left;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 9px;border-bottom:1px solid #353841;vertical-align:middle}}thead{{background:#242830}}
th[scope=row]{{min-width:145px}}td:nth-child(3){{white-space:nowrap}}tr:target{{background:#33303b}}
.images{{display:flex;gap:4px;min-width:max-content}}img{{display:block;object-fit:contain;border-radius:4px}}
.design{{display:none}}.design-view .design{{display:table-cell}}
summary{{cursor:pointer;font-size:.75rem;color:#b6bdca}}
small{{display:block;font-weight:400;color:#b6bdca;font-size:.75rem}}
</style><main>
<h1>Final Boss alpha leaderboard</h1>
<p>{len(completed)}/{len(progress['trained'])} trained recipes through the damage-adjusted search · {len(active)} running · {len(ranked)} settings measured across all searches · {queued} awaiting damage ratings.</p>
<p>Jev rates rendering defects in the finished artwork on its own. The Boss score uses matched character gain and distinction, then multiplies by the undamaged fraction. Zero gain scores zero.</p>
<p>Same eight cases at RAW192. Boss effect is the average gain over slider off. Expand a seven-case rank to see how far that recipe moves when one scene is left out. Each recipe keeps its full-panel strength.</p>
<p><a href="final-boss-vibe12-alpha-tuned.html#trial-paths">Search paths</a> · <a href="final-boss-vibe12-opening-comparison.html">Compare opening four</a> · <a href="final-boss-vibe12-preference-review.html">Blind finalist comparison</a> · <a href="final-boss-vibe12-alpha-leaderboard-v1.html">Previous scoring</a> · <a href="?fixed=constant">Shared constant α0.6375</a> · <a href="?fixed=early">Shared early α2/3</a></p>
{design_help}
<div class="controls"><label>Sort by <select id="metric"><option value="score">Final Boss score</option>{design_option}<option value="effect">Boss effect</option><option value="damage">Least image damage</option></select></label>
<label>Show <select id="scope"><option value="top">Best measured per recipe</option><option value="adaptive">Current search and fixed strengths</option><option value="tradeoffs">Effect and damage trade-offs</option><option value="all">All measured strengths</option><option value="early">Shared early α2/3</option><option value="constant">Shared constant α0.6375</option></select></label></div>
<div class="scroll"><table><thead><tr><th scope="col">Rank</th><th scope="col">Recipe</th><th scope="col">Strength</th>
<th scope="col">Matched samples</th><th scope="col" class="design">Distinct Boss score</th><th scope="col">Boss effect</th><th scope="col">Final Boss score</th><th scope="col">Image damage</th></tr></thead>
<tbody><tr><td></td><th scope="row">Slider off</th><td></td><td>{images(cases, originals, True)}</td><td class="design"></td><td></td><td></td><td></td></tr>{reference_html}</tbody>
<tbody id="trials">{''.join(lines)}</tbody></table></div></main>
<script>
const body = document.getElementById('trials');
const metric = document.getElementById('metric');
const scope = document.getElementById('scope');
const rows = Array.from(body.rows);
rows.forEach((row, i) => row.initialRank = i);
const query = new URLSearchParams(location.search);
const fixed = query.get('fixed');
if (metric.querySelector('option[value="design"]') && (query.get('metric') === 'design' || (!query.has('metric') && !fixed && !location.hash))) metric.value = 'design';
if (fixed === 'early' || fixed === 'constant') scope.value = fixed;
if (location.hash && document.getElementById(location.hash.slice(1))) scope.value = 'all';
const isLatest = row => row.dataset.adaptive === '1' || row.dataset.fixed === '1';
function tradeoffs() {{
  const groups = new Map();
  for (const row of rows) {{
    const method = row.dataset.method;
    if (!groups.has(method)) groups.set(method, []);
    groups.get(method).push(row);
  }}
  const options = new Set();
  for (const candidates of groups.values()) {{
    for (const row of candidates) {{
      const effect = Number(row.dataset.effect), damage = Number(row.dataset.damage);
      const improved = candidates.some(other => {{
        const gain = Number(other.dataset.effect), harm = Number(other.dataset.damage);
        return gain >= effect && harm <= damage && (gain > effect || harm < damage);
      }});
      if (!improved) options.add(row);
    }}
  }}
  return options;
}}
function update() {{
  const key = metric.value;
  const design = key === 'design';
  if (design) scope.value = 'top';

  body.closest('table').classList.toggle('design-view', design);
  const direction = key === 'damage' ? 1 : -1;
  rows.sort((a, b) => direction * (Number(a.dataset[key] ?? -1) - Number(b.dataset[key] ?? -1)) || a.initialRank - b.initialRank);
  let rank = 0;
  const seen = new Set();
  const options = scope.value === 'tradeoffs' ? tradeoffs() : null;
  for (const row of rows) {{
    const method = row.dataset.method;
    const latest = isLatest(row);
    const representative = !seen.has(method);
    row.hidden = design ? !row.hasAttribute('data-design') : options ? !options.has(row) : scope.value === 'top' ? !representative : (scope.value === 'early' || scope.value === 'constant') ? row.dataset.control !== scope.value : scope.value === 'adaptive' && !latest;
    if (!row.hidden) {{
      seen.add(method);
      row.querySelector('.rank').textContent = ++rank;
    }}
    body.appendChild(row);
  }}
}}
metric.addEventListener('change', update);
scope.addEventListener('change', () => {{
  if (scope.value !== 'top' && metric.value === 'design') metric.value = 'score';
  update();
}});
update();
</script></html>'''
    file = PUBLIC / OUTPUT
    temporary = file.with_suffix('.publish')
    temporary.write_text(page)
    temporary.replace(file)
    with urlopen(f'http://127.0.0.1:8765/{OUTPUT}', timeout=8) as response:
        if response.read() != page.encode():
            raise ValueError('Tailnet returned a different strength leaderboard')
    write(STORAGE / 'alpha-standings.json', {'scoring_version': VERSION, 'quality_metric': DAMAGE, 'jev_binding': BINDING, 'selection_metric': SCORE, 'cases': cases, 'rows': ranked})
    if design:
        from datetime import datetime, timezone
        current = sorted(audited.values(),key=lambda r:(-r['score'],r['method']))
        write(STORAGE/'distinct-alpha-standings.json',dict(design,
              as_of=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
              rows=current, tuned_methods=sorted(selected), baseline_signature=design['signature'],
              protocol='Use completed distinct-design alpha selections where available and prior measured per-image strengths elsewhere. Both use all 28 character pairs.'))
    write(STORAGE / 'visible-damage-published.json', {'version': VERSION, 'settings': len(ranked), 'waiting': queued})
    print(f'{len(ranked)} strengths published with finished-image ratings', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    previous = None
    while True:
        inputs = [STORAGE / 'alpha-standings-v1.json']
        design = ROOT / "metric-calibration/design-range/latest.json"
        if design.is_file():
            inputs.append(design)
        if RATINGS.exists():
            inputs.append(RATINGS)
        queue = SEARCH / ADAPTIVE / 'queue.json'
        if queue.is_file():
            inputs.append(queue)
        inputs.extend((SEARCH / design_selection.RUN).glob('*/result.json'))
        for name in SEARCH_ROOTS:
            inputs.extend((SEARCH / name).glob('*/renders/*/*/validation.json'))
        revision = tuple((p, hash(p.read_bytes()) if p == queue else p.stat().st_mtime_ns) for p in sorted(inputs))
        if revision != previous:
            publish()
            previous = revision
        if not args.watch:
            break
        sleep(20)


if __name__ == '__main__':
    main()
