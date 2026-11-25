"""Publish common-strength and Jev-selected images for the same saved RAW adapters."""
from html import escape
from pathlib import Path
from statistics import fmean
import sys
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent
sys.path.insert(0,str(CAMPAIGN))
from control import CACHE, STORAGE
from optimization_worker import pin, read
from publish_optimization import PUBLIC, sample

LABELS=read(CAMPAIGN/'recipes.json')
NAMES={'ranger':'Ranger','smith':'Smith','stationer':'Stationer','guard':'Guard'}
OUTPUT='final-boss-vibe12-alpha-tuned.html'


def image(record, label):
    path=sample(record,f"vibe12-alpha-{record['sha256']}.png")
    return f'<figure><a href="{path}"><img src="{path}" alt="{escape(label)}" height="512" width="512" loading="lazy"></a><figcaption>{escape(label)}</figcaption></figure>'


def rows(method, report, definition):
    trained=method.removesuffix('-late')
    fixed=read(CAMPAIGN/'tuning'/trained/'boss-effect-tuning.json')['matched_constant']
    if report['checkpoint']!=definition['checkpoint'] or report['judge']!='jev' or report['sampler']['model']!=read(CACHE/'index.json')['identity']['model']:
        raise ValueError('Saved search differs from its verified adapter or Jev plan')
    root=ROOT/f'{method}.alpha-tune'
    by_name={x['id']:x for x in definition['cases']}
    for selected in report['selected']:
        alpha,profile=selected['alpha'],selected['profile']
        folder=root/'renders'/profile/alpha.hex()
        sections={split:read(folder/f'{split}.json') for split in ('search','reserved')}
        if any((x['alpha'],x['profile'],x['effective_strengths'])!=(alpha,profile,selected['effective_strengths']) for x in sections.values()):
            raise ValueError('Selected tuning differs from its exact render strength')
        subjects=[]
        for key,name in NAMES.items():
            match=sections[by_name[key]['split']]['cases'][key]
            reference=fixed['cases'][key]
            if match['off']['sha256']!=reference['off']['sha256'] or match['observation']['integrity']!='pass':
                raise ValueError('Saved tuned image failed the exact same-reference check')
            images=image(match['off'],'Slider off')+image(reference['on'],'Shared constant α 0.6375')+image(match['on'],f'Search result · {profile} α {alpha:g}')
            subjects.append(f'<section aria-label="{name}"><h3>{name}</h3><div class="pair">{images}</div></section>')
        yield (f'<article><h2>{escape(LABELS[trained])}</h2><p>{profile.title()} α <strong>{alpha:g}</strong> · '
               f'Jev matched gain {selected["search_gain"]:+.3f} on search cases and {selected["reserved_gain"]:+.3f} '
               f'on validation cases</p><div class="cases">{"".join(subjects)}</div></article>')


def comparison_table(methods, standings):
    scores = []
    fixed = {row['method']: row for row in standings['rankings']['matched_constant']}
    progress = read(ROOT / 'status.json')

    def formatted(value):
        return '—' if value is None else f'{100 * value:.2f}'

    for method in methods:
        receipt = ROOT / f'{method}.alpha-tune/tuning.json'
        trained = method.removesuffix('-late')
        ordinary = fixed.get(trained, {})
        before = ordinary.get('means', {}).get('final_boss_score')
        label = escape(LABELS[trained])
        if not receipt.is_file():
            result = progress.get(method)
            if result and not result['complete'] and result['receipt'].get('returncode') == 2:
                scores.append((-1, f'<tr><th scope="row">{label}</th><td>No setting qualified</td>'
                                   f'<td>{formatted(before)}</td><td>—</td><td>—</td><td>Search failed</td></tr>'))
            continue
        report = read(receipt)
        for selected in report['selected']:
            profile, alpha = selected['profile'], selected['alpha']
            reviewed = STORAGE / 'alpha-leaderboard' / method / f'{profile}-{alpha.hex()}.json'
            if not reviewed.is_file():
                continue
            checked = read(reviewed)
            if (checked['tuning_sha256'] != pin(receipt)['sha256'] or checked['selection'] != selected
                    or checked['board_cases'] != standings['cases']
                    or checked['scoring_version'] != standings['scoring_version']
                    or (ordinary and ordinary['raw_sha256'] != report['checkpoint']['sha256'])):
                raise ValueError('Reviewed scores differ from the comparison inputs')
            after = checked['means']['final_boss_score']
            status = '' if checked['status'] == 'pass' else f' · {checked["status"]}'
            change = None if before is None or after is None else 100 * (after - before)
            difference = '—' if change is None else f'{change:+.2f}'
            outcome = 'Unmeasured' if change is None else 'Improved' if change > 0 else 'Failed to improve'
            scores.append((after if after is not None else -1,
                           f'<tr><th scope="row">{label}</th><td>{profile.title()} α {alpha:g}</td>'
                           f'<td>{formatted(before)}</td><td>{formatted(after)}{status}</td>'
                           f'<td>{difference}</td><td>{outcome}</td></tr>'))
    if not scores:
        return ''
    rows = ''.join(row for _, row in sorted(scores, reverse=True))
    return ('<section><h2>Leaderboard scores on the same eight cases</h2><div class="scroll"><table>'
            '<thead><tr><th>Recipe</th><th>Selected strength</th><th>Shared constant α 0.6375</th><th>Adjusted</th>'
            f'<th>Change</th><th>Outcome</th></tr></thead><tbody>{rows}</tbody></table></div></section>')


def leaderboard_retune():
    panels = []
    for path in sorted((ROOT / 'leaderboard-retune').glob('*/result.json')):
        result = read(path)
        confirmation = path.with_name('confirmed-selection.json')
        if confirmation.is_file():
            checked = read(confirmation)
            if checked['source'] != pin(path):
                raise ValueError('The confirmed setting belongs to a different alpha search')
            result = checked
        baseline = result['baseline']
        alternative = result['selected'] or result['best_measured']
        label = escape(LABELS[result['method']])
        before = 100 * baseline['means']['final_boss_score']
        if alternative is None:
            panels.append(f'<article><h3>{label}</h3><p>Kept constant α 0.6375 at {before:.2f}. No alternative passed validation.</p></article>')
            continue
        setting = alternative['setting']
        profile, alpha = setting['profile'], setting['alpha']
        strength = f'{profile.title()} α {alpha:g}'
        after = 100 * alternative['means']['final_boss_score']
        boss = 100 * alternative['means']['boss_distinction']
        outcome = 'Improved' if result['selected'] else 'Kept the shared setting'
        subjects = []
        for key, name in NAMES.items():
            old, new = baseline['rows'][key], alternative['rows'][key]
            if old['off']['sha256'] != new['off']['sha256']:
                raise ValueError('The revised alpha search used a different slider-off image')
            images = image(old['off'], 'Slider off') + image(old['on'], 'Shared constant α 0.6375') + image(new['on'], strength)
            subjects.append(f'<section aria-label="{name}"><h3>{name}</h3><div class="pair">{images}</div></section>')
        panels.append(f'<article><h3>{label}</h3><p>{outcome} · {strength} scored {after:.2f} '
                      f'vs {before:.2f} at the shared strength · Boss distinction {boss:.1f}/100</p>'
                      f'<div class="cases">{"".join(subjects)}</div></article>')
    if not panels:
        return ''
    return ('<section><h2>Search guided by the leaderboard score</h2>'
            '<p>The search samples all three timing profiles, refines near the strongest scores and checks the finalists on eight matched cases.</p>'
            f'{"".join(panels)}</section>')


def alpha_110_samples():
    def subjects(one, higher):
        parts = []
        for key, case in higher.items():
            original = one[key]
            if original['off']['sha256'] != case['off']['sha256']:
                raise ValueError('The α 1.10 sample comparison used different originals')
            name = key.title()
            images = (image(case['off'], 'Slider off') + image(original['on'], 'Constant α 1.00')
                      + image(case['on'], 'Constant α 1.10'))
            parts.append(f'<section aria-label="{name}"><h3>{name}</h3><div class="pair">{images}</div></section>')
        return ''.join(parts)

    panels = []
    for method in ('shared-best-seed-split', 'shared-best-stock'):
        folder = ROOT / 'leaderboard-retune' / method / 'renders' / 'constant'
        paths = [(folder / alpha.hex() / 'search.json') for alpha in (1.0, 1.10)]
        if not all(path.is_file() for path in paths):
            continue
        first, second = [read(path) for path in paths]
        one, higher = first['rows'], second['rows']
        first_score = 100 * fmean(row['final_boss_score'] for row in one.values())
        second_score = 100 * fmean(row['final_boss_score'] for row in higher.values())
        score_text = f'Four search cases · {first_score:.2f} at α 1.00 · {second_score:.2f} at α 1.10'
        validation_samples = ''
        validation = paths[1].with_name('validation.json')
        if validation.is_file():
            reviewed = read(validation)
            score = 100 * fmean(row['final_boss_score'] for row in reviewed['rows'].values())
            score_text += f' · {score:.2f} on the eight leaderboard cases'
            if paths[0].with_name('validation.json').is_file():
                comparison = read(paths[0].with_name('validation.json'))['rows']
                validation_samples = ('<details><summary>Eight leaderboard images at α 1.10</summary>'
                                      f'<div class="cases">{subjects(comparison, reviewed["rows"])}</div></details>')
        panels.append(f'<article><h3>{escape(LABELS[method])}</h3><p>{score_text}</p><div class="cases">'
                      f'{subjects(one, higher)}</div>{validation_samples}</article>')
    if not panels:
        return ''
    return f'<section id="alpha-110"><h2>What α 1.10 did on the search images</h2>{"".join(panels)}</section>'


def trial_paths():
    def value(trial, name, precision):
        values = [row[name] for row in trial['rows'].values()]
        return '—' if None in values else f'{100 * fmean(values):.{precision}f}'

    panels = []
    paths = [(directory, label) for archive, label in (
        ('leaderboard-retune-damage', 'Current search · Boss effect and image damage'),
        ('leaderboard-retune-adaptive', 'Earlier adaptive search'),
        ('leaderboard-retune-centered', 'Effect search from α 1.00'),
        ('leaderboard-retune', 'Previous score search'),
    ) for directory in sorted((ROOT / archive).glob('*')) if directory.name in LABELS]
    for directory, label in paths:
        files = sorted(directory.glob('renders/*/*/search.json'), key=lambda path: path.stat().st_mtime_ns)
        if not files:
            continue
        rows = []
        for index, path in enumerate(files, 1):
            trial = read(path)
            setting = trial['setting']
            score = value(trial, 'final_boss_score', 3)
            gain = value(trial, 'boss_effect', 1)
            damage = value(trial, 'image_breakage', 1)
            rows.append(f'<tr><td>{index}</td><td>{setting["profile"].title()}</td><td>{setting["alpha"]:.7g}</td>'
                        f'<td>{score}</td><td>{gain}</td><td>{damage}</td></tr>')
        panels.append(f'<details><summary>{escape(LABELS[directory.name])} · {label}</summary><div class="scroll"><table>'
                      '<thead><tr><th>Trial</th><th>Profile</th><th>α</th><th>Final Boss score</th><th>Boss gain</th>'
                      f'<th>Image damage</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></details>')
    if not panels:
        return ''
    return ('<section id="trial-paths"><h2>Alpha trials in order</h2>'
            f'<p>Scores use the four search images. The eight-case validation is reported separately.</p>{"".join(panels)}</section>')


def main():
    plan=read(ROOT/'plan.json')
    body=[]
    methods=(*plan['methods'], 'shared-best-seed-split-late')
    table=comparison_table(methods, read(STORAGE/'leaderboard.json'))
    for method in methods:
        receipt=ROOT/f'{method}.alpha-tune/tuning.json'
        if receipt.is_file():
            body.extend(rows(method,read(receipt),read(ROOT/f'{method}.json')))
    if not body:
        raise ValueError('No qualified alpha result has been saved yet')
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Final Boss strength search</title>
<style>:root{{color-scheme:dark;background:#18191c;color:#f4eee6;font:16px/1.45 system-ui,sans-serif}}main{{width:min(1500px,calc(100% - 32px));margin:25px auto}}h1,h2,h3,p,figure{{margin:0}}h2{{font-size:1.2rem}}h3{{font-size:1rem;margin-bottom:6px}}a,strong{{color:#ffc7a6}}p{{margin:6px 0 10px}}article{{border-top:1px solid #454854;margin-top:16px;padding-top:12px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;margin-top:.4rem;text-align:left;font-variant-numeric:tabular-nums}}th,td{{padding:.4rem .8rem;border-bottom:1px solid #454854}}.cases{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:14px}}.pair{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}}img{{display:block;width:100%;height:auto;aspect-ratio:1;border-radius:4px}}figcaption{{font-size:.75rem;color:#c4c8d3;margin-top:5px}}</style>
<main><h1>Final Boss strength search</h1>
<p><a href="final-boss-vibe12-alpha-leaderboard.html">Alpha leaderboard</a> · <a href="final-boss-vibe12-leaderboard-constant.html">Shared-strength leaderboard</a> · <a href="final-boss-vibe-corrected-teachers.html">Training pairs</a> · <a href="#alpha-110">α 1.10 samples</a> · <a href="#trial-paths">Alpha trials in order</a></p>
{trial_paths()}{alpha_110_samples()}<details><summary>Archived selection policies</summary>{leaderboard_retune()}<section><h2>Original Jev search</h2>
{table}{''.join(body)}</section></details></main></html>'''
    file=PUBLIC/OUTPUT; pending=file.with_suffix('.publish');pending.write_text(page);pending.replace(file)
    url=f'http://127.0.0.1:8765/{OUTPUT}'
    with urlopen(url,timeout=10) as response:
        if response.read()!=page.encode():raise ValueError('Tailnet returned a different alpha comparison')
    print(url,flush=True)

if __name__=='__main__':
    import argparse
    from time import sleep
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    seen = None
    while True:
        root = ROOT / 'leaderboard-retune-damage'
        revision = tuple(sorted((str(path), path.stat().st_mtime_ns)
                                for path in root.glob('*/renders/*/*/validation.json'))) + tuple(sorted(
                                    (str(path), path.stat().st_mtime_ns) for path in root.glob('*/search-path.json')))
        if revision != seen:
            main()
            seen = revision
        if not args.watch:
            break
        sleep(30)
