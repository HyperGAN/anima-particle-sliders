"""Publish the per-recipe scalar scoreboard with matched outputs, training provenance and trial evidence."""

from hashlib import sha256
from html import escape
from pathlib import Path
from statistics import fmean
import fcntl, json, sys, tempfile, os
from release_tools.constant_alpha_retest import verify_search
from release_tools.final_boss_visible_damage import score_case
from release_tools.final_boss_design_scores import summarize

from locations import GALLERY_URL, TIMED_GALLERY_URL, RESULTS as here
from sample_rows import CSS, strip

web = here.parent / "sample-gallery"
bundle = web / "constant-retest"
bundle.mkdir(exist_ok=True)


def read(p):
    return json.loads(p.read_text())


def write(p, data):
    if isinstance(data, str):
        data = data.encode()
    if p.exists() and p.read_bytes() == data:
        return
    with tempfile.NamedTemporaryFile(
        dir=p.parent, prefix=".retest-", delete=False
    ) as temp:
        temp.write(data)
    os.chmod(temp.name, 0o644)
    os.replace(temp.name, p)


def js(p, obj):
    write(p, json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def link(path, txt):
    return f'<a href="{escape(path)}">{escape(txt)}</a>'


def png(r):
    url = "images/" + r["sha256"] + ".png"
    path = web / url
    if not path.exists():
        data = Path(r["path"]).read_bytes()
        assert sha256(data).hexdigest() == r["sha256"]
        write(path, data)
    return url


def fig(ref, label):
    url = png(ref)
    return f'<figure><figcaption>{escape(label)}</figcaption><a href="{url}"><img loading="lazy" width="512" height="512" src="{url}" alt="{escape(label)}"></a></figure>'


roster = read(here / "inventory.json")
cfg = read(here / "panels.json")
from lumen_studio.study.objectives import protocol

if any(cfg[key] != value for key, value in protocol().items()):
    raise ValueError(
        "The research leaderboard and native Final Boss trainer use different judge protocols"
    )
requests = {r["id"]: r["request"] for r in cfg["cases"]}
hist = read(here.parent / "product/docs/final-boss-raw192/vibe12/design-range.json")[
    "development"
]
prior = {r["method"]: r for r in hist["rows"]}
settings = read(here.parent / "product/docs/final-boss-raw192/vibe12/standings.json")[
    "settings"
]
campaign = here.parent.parent / "anima-final-boss-vibe12-20261124"


def historical(n, c):
    best = prior[n]
    row = next(
        t
        for t in settings
        if (t["method"], t["profile"], t["alpha"])
        == (n, best["profile"], best["alpha"])
    )
    sha = best["inputs"][c]
    assert sha == row["cases"][c]["on_sha256"]
    rel = Path(n) / "renders" / best["profile"] / float(best["alpha"]).hex()
    for base in [
        campaign / "strength-search" / run / rel for run in reversed(row["runs"])
    ] + [campaign / "tuning" / rel]:
        if (base / "validation.json").is_file():
            x = read(base / "validation.json")["rows"][c]["on"]
            if x["sha256"] == sha:
                return x
        if (base / c / "image.png").is_file():
            return dict(path=str(base / c / "image.png"), sha256=sha)
    raise ValueError("Missing historical image for " + n + "/" + c)


adj = read(
    here.parent / "product/docs/final-boss-raw192/vibe12/constant-leaderboard.json"
)["sliders"]
objective = sys.argv[1] if len(sys.argv) > 1 else "distinct_boss_score"
publish_lock = (here / (objective + ".publish.lock")).open("a")
fcntl.flock(publish_lock, fcntl.LOCK_EX)
is_design = objective == "distinct_boss_score"
slug = "constant-retest" if is_design else "constant-effect-retest"
unit = 100 if is_design else 1
fmt = lambda v: "—" if v is None else f"{100 * v:.2f}" if is_design else f"{v:+.4f}"


def summarize_trial(t):
    ident = t["identity"]
    assert ident["objective"] == objective and ident["gains"] == [ident["alpha"]] * 10
    assert list(t["rows"]) == cfg["panels"][ident["phase"]]
    for c, r in t["rows"].items():
        assert r["off"]["sha256"] == next(
            v["off"]["sha256"] for v in cfg["cases"] if v["id"] == c
        )
    if t["score"] is not None:
        gain = fmean(
            r["boss_distinction"] - r["original_boss_distinction"]
            for r in t["rows"].values()
        )
        assert gain == t["mean_gain"]
        if is_design:
            for pair in t["pairs"]:
                assert pair["images"] == [
                    t["rows"][c]["on"]["sha256"] for c in pair["cases"]
                ]
                assert pair["repeat"] == fmean(
                    pair[k]["value"] for k in ("forward", "reverse")
                )
            calc = summarize(
                {c: score_case(r) for c, r in t["rows"].items()}, t["pairs"]
            )
            assert calc == t["design"] and calc["score"] == t["score"]
        else:
            assert gain == t["score"]
    rows = {
        c: {
            **{k: v for k, v in r.items() if k not in ("off", "on")},
            "off_sha256": r["off"]["sha256"],
            "on_sha256": r["on"]["sha256"],
        }
        for c, r in t["rows"].items()
    }
    return dict(
        alpha=ident["alpha"],
        gains=ident["gains"],
        score=t["score"],
        mean_gain=t["mean_gain"],
        design=t["design"],
        rows=rows,
        pairs=t["pairs"],
    )


results = []
snippets = {}
inline = {}
per_recipe = []
for n, r in roster.items():
    if r["run"] in ("anime", "appeal-v2"):
        continue
    d = here / "methods" / n
    base = dict(
        method=n,
        label=r["label"],
        pg=r["pg"],
        step=r["step"],
        training=r["training"],
        raw_sha256=r["raw"]["sha256"],
        rates=r.get("rates"),
        previous_design=prior.get(n, {}).get("score"),
        score=None,
        integrity=None,
    )
    t = d / objective
    if not (t / "result.json").is_file():
        results.append(
            dict(base, state="pending", probes=len(list((t / "search").glob("*.json"))))
        )
        continue
    observed = (t / "result-observed.json").exists()
    out = read(t / ("result-observed.json" if observed else "result.json"))
    assert out["raw"] == base["raw_sha256"] and out["objective"] == objective
    path = out["trials"]
    a = verify_search(path, out.get("alpha"))
    trials = [
        summarize_trial(read(t / "search" / (val.hex() + ".json"))) for val, _ in path
    ]
    assert [(v["alpha"], v["score"]) for v in trials] == [tuple(item) for item in path]
    rank = None
    if a is not None:
        data = read(
            t / "ranking-observed.json"
            if observed
            else t / "ranking" / (a.hex() + ".json")
        )
        rank = summarize_trial(data)
        assert rank["score"] == out["score"] and set(rank["rows"]) == set(
            cfg["panels"]["ranking"]
        )
        inline[n] = strip(
            (c.title(), png(data["rows"][c]["on"])) for c in cfg["panels"]["ranking"]
        )
        cases = []
        for c, v in data["rows"].items():
            measured = (
                v["boss_distinction"] - v["original_boss_distinction"]
                if None not in (v["boss_distinction"], v["original_boss_distinction"])
                else None
            )
            info = f" · gain {measured:+.3f}" if measured is not None else ""
            old = (
                fig(
                    historical(n, c),
                    f"Historical {prior[n]['profile']} α {prior[n]['alpha']:g}",
                )
                if n in prior
                else ""
            )
            pair = (
                fig(v["off"], c.title() + " · OFF")
                + old
                + fig(
                    v["on"], f"ON · α {a:g}" + info + " · " + out["checks"][c]["status"]
                )
            )
            cases.append(
                f'<article><h4>{escape(c.title())}</h4><div class="pair boss-three">{pair}</div><details><summary>Prompt and seed {requests[c]["seed"]}</summary><p>{escape(requests[c]["prompt"])}</p></details></article>'
            )
        snippets[n] = "".join(cases)
    detail = dict(
        base,
        state=out["decision"],
        objective=objective,
        alpha=a,
        probes=len(trials),
        search_score=out.get("search_score"),
        score=out.get("score"),
        mean_gain=out.get("mean_gain"),
        integrity=out.get("integrity"),
        checks=out.get("checks"),
        original_measurement=out.get("original"),
    )
    js(
        bundle / (n + "-" + objective + ".json"),
        dict(detail, search=trials, ranking=rank, replay=read(d / "pg012-parity.json")),
    )
    results.append(detail)
candidates = [
    r
    for r in results
    if r["state"] == "measured"
    and r["training"] == "completed"
    and r["score"] is not None
]
ordered = sorted(candidates, key=lambda r: (-r["score"], r["method"]))
last = None
for i, r in enumerate(ordered, 1):
    if last != r["score"]:
        place = i
    r["rank"] = place
    last = r["score"]
others = sorted(
    (r for r in results if r not in candidates),
    key=lambda r: (r["score"] is None, -(r["score"] or 0), r["method"]),
)
count = sum(r["state"] != "pending" for r in results)


historical_public = (
    here.parent.parent / "anima-final-boss-research-raw192-20261123/sample-gallery"
)
bookmark = historical_public / "final-boss-vibe12-alpha-leaderboard.html"
timed_archive = bookmark.with_name(
    "final-boss-vibe12-alpha-leaderboard-timed-20261124.html"
)
if is_design and count == len(results) and bookmark.is_file():
    original = bookmark.read_bytes()
    if not timed_archive.exists():
        if b"Historical Final Boss alpha leaderboard" not in original:
            raise ValueError(
                "Could not pin the original timed leaderboard before switching the bookmark"
            )
        write(timed_archive, original)
timed_link = TIMED_GALLERY_URL + (
    timed_archive.name if timed_archive.is_file() else bookmark.name
)


def table(group, references=False):
    columns = [
        "Rank",
        "Training recipe",
        "Constant α",
        "Matched samples",
        "Distinct Boss / 100" if is_design else "Matched gain",
        "Boss gain",
        "Integrity",
        "RAW update",
        "Training PG",
        "Trials",
        "Search score",
        "Training status",
    ]
    if is_design:
        columns += ["Historical score", "Difference"]
    lines = []
    if references:
        panel = cfg["panels"]["ranking"]
        off = {v["id"]: v["off"] for v in cfg["cases"]}
        historical_row = prior["late-cues"]
        for label, strength, images, score in [
            (
                "Matched baseline",
                "OFF",
                strip((c.title(), png(off[c])) for c in panel),
                "—",
            ),
            (
                "Published late-cues reference",
                f"{historical_row['profile']} {historical_row['alpha']:g}",
                strip((c.title(), png(historical("late-cues", c))) for c in panel),
                fmt(historical_row["score"]) if is_design else "—",
            ),
        ]:
            lines.append(
                f'<tr class="ref"><td>—</td><th scope="row">{label}</th><td>{strength}</td><td>{images}</td><td>{score}</td><td colspan="{len(columns) - 5}"></td></tr>'
            )
    for row in group:
        name = row["method"]
        label = (
            link("#" + name, row["label"]) if name in snippets else escape(row["label"])
        )
        alpha = row.get("alpha")
        gain = row.get("mean_gain")
        fields = [
            f"{alpha:g}" if alpha is not None else "—",
            inline.get(name, "Awaiting samples"),
            fmt(row["score"]),
            f"{gain:+.4f}" if gain is not None else "—",
            row["integrity"] or row["state"],
            str(row["step"]),
            row["pg"],
            str(row["probes"]),
            fmt(row.get("search_score")),
            row["training"],
        ]
        if is_design:
            old = row["previous_design"]
            fields.extend(
                [
                    fmt(old),
                    f"{100 * (row['score'] - old):+.2f}"
                    if row["score"] is not None and old is not None
                    else "—",
                ]
            )
        score = row["score"] if row["score"] is not None else ""
        signed = gain if gain is not None else ""
        lines.append(
            f'<tr data-score="{score}" data-gain="{signed}" data-method="{escape(name)}"><td>{row.get("rank", "—")}</td><th scope="row">{label}</th>'
            + "".join(f"<td>{v}</td>" for v in fields)
            + "</tr>"
        )
    controls = ""
    if references:
        controls = (
            '<div class="board-controls" hidden><label>Sort by <select id="board-sort"><option value="score">Board score</option><option value="gain">Boss gain</option></select></label> '
            '<label>Show <select id="board-limit"><option value="all">All recipes</option><option value="10">Top 10</option><option value="5">Top 5</option></select></label></div>'
        )
    return (
        controls
        + '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{c}</th>" for c in columns)
        + "</tr></thead><tbody>"
        + "".join(lines)
        + "</tbody></table></div>"
    )


title = (
    "Final Boss constant-alpha retest"
    if is_design
    else "Final Boss matched-effect retest"
)
html = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'
    + title
    + '</title><link rel="stylesheet" href="gallery.css">'
    + CSS
    + "</head><body><header>"
    + link("index.html", "Anima samples")
    + "<nav>"
    + (
        link("training-updates.html", "Training updates")
        if (web / "training-updates.html").is_file()
        else ""
    )
    + link(
        "constant-effect-retest.html" if is_design else "constant-retest.html",
        "Archived intensity-only run" if is_design else "Distinct Boss score",
    )
    + " "
    + link("constant-alpha.html", "Shared-strength comparison")
    + " "
    + link("final-boss-raw128.html", "Failed RAW128")
    + "</nav></header><main><h1>"
    + title
    + "</h1>"
)
if not is_design:
    html += (
        "<p><strong>Archived partial experiment.</strong> The intensity-only run ended when the distinct Boss objective was confirmed. Unfinished rows were never ranked. "
        + link("constant-retest.html", "Open the default distinct Boss leaderboard")
        + "</p>"
    )
html += f"<p>{count}/{len(results)} checkpoints · four alpha-search characters · eight ranking characters · ten Euler steps.</p>"
html += "<details><summary>Scoring and experiment settings</summary>"
html += f"<p>{count} of {len(results)} Final Boss arms have completed their retest. Every row uses one saved RAW. The same alpha applies to all ten native Euler steps. Four matched characters guide the adaptive search, then the script freezes alpha and evaluates eight different characters. All eight are existing development cases. New renders use PG 0.12.0. The search brackets from α 1, tests both sides, extends a rising edge to the native limit of 5, and refines the measured leader to neighboring probes within 1/32. Completed training recipes receive numerical ranks. Stopped and screening runs appear as diagnostics. Failed or unresolved structural checks keep a candidate out of release.</p>"
html += (
    "<p>The distinct Boss score uses the original Jev character-distinction, visible-damage and design-repetition judgments. Distinction gain earns more credit for a convincing finished boss. Visible damage reduces the per-image score. Each of the 28 character pairs receives a deduction for repeated design, measured in both image orders. Search and board scores are shown on the same 0–100 scale. The Boss gain column reports signed ON minus OFF intensity.</p>"
    if is_design
    else "<p>Both search and ranking use the equally weighted signed change in Boss character intensity from the matched OFF image. Each score is on the original 0–1 intensity scale. Structural checks determine eligibility for promotion separately.</p>"
)
html += (
    "<p>"
    + link("constant-retest/" + slug + ".json", "Scores and training provenance")
    + " · "
    + link(
        timed_link,
        "Original timed leaderboard",
    )
    + "</p>"
)
html += "</details>"
html += "<h2>Completed training recipes</h2>" + table(
    ordered + [r for r in others if r["training"] == "completed"], references=True
)
html += """<script>
(()=>{const controls=document.querySelector('.board-controls');if(!controls)return;
const body=controls.nextElementSibling.querySelector('tbody'), rows=[...body.querySelectorAll('[data-method]')];
const sort=controls.querySelector('#board-sort'),limit=controls.querySelector('#board-limit');
function refresh(){const key=sort.value,ordered=[...rows].sort((a,b)=>(Number(b.dataset[key]||-Infinity)-Number(a.dataset[key]||-Infinity)));
ordered.forEach((row,i)=>{body.append(row);row.hidden=limit.value!=='all'&&i>=Number(limit.value)})}
controls.hidden=false;sort.addEventListener('change',refresh);limit.addEventListener('change',refresh)})();</script>"""
html += "<h2>Stopped runs, rate screens and version controls</h2>" + table(
    [r for r in others if r["training"] != "completed"]
)
html += "<h2>1990s anime and character appeal</h2><p>These PG 0.12.0 checkpoints use their own intended-effect rubric. Each search measures the mean ON minus OFF on four selection cases, reserves the highest measured scalar and renders four separate ranking cases. The image judge scores the ranking renders again.</p>"
neighbor_rows = []
for prior_row, key in zip(adj[1:], ["anime", "appeal-v2"], strict=True):
    study = here / "neighbor-effect-results" / key
    if not (study / "result.json").is_file():
        neighbor_rows.append(dict(slider=prior_row["slider"], state="pending"))
        html += f"<p>{escape(prior_row['slider'])} · retest in progress · {link('constant-alpha.html#' + key, 'Earlier samples')}</p>"
        continue
    result = read(study / "result.json")
    measurement = read(study / "ranking.json")
    assert (
        result["score"] == measurement["mean_gain"]
        and measurement["raw"]["sha256"] == result["raw"]["sha256"]
    )
    verify_search(result["trials"], result["alpha"])
    imgs = []
    matched = []
    for pair in measurement["pairs"]:
        c = pair["case"]
        before = c["off"]["output"]
        after = pair["on"]["output"]
        matched.append(
            dict(
                seed=c["seed"],
                prompt=c["prompt"],
                off_sha256=before["sha256"],
                on_sha256=after["sha256"],
                gain=pair["gain"],
                integrity=pair["on"]["integrity"]["status"],
                off_score=c["off"]["judgment"]["score"],
                on_score=pair["on"]["judgment"]["score"],
            )
        )
        imgs.append(
            f'<article><h4>Seed {c["seed"]}</h4><div class="pair">'
            + fig(before, "Slider OFF")
            + fig(after, "Slider ON · " + pair["on"]["integrity"]["status"])
            + "</div></article>"
        )
    info = dict(
        slider=result["slider"],
        pg=result["pg"],
        raw_sha256=result["raw"]["sha256"],
        step=result["step"],
        objective=result["objective"],
        alpha=result["alpha"],
        trials=result["trials"],
        score=result["score"],
        status=result["status"],
        cases=matched,
        frozen_reference=result["reference"],
        prior_calibration=prior_row["alpha"],
        prior_gain=prior_row["confirmation_gain"],
    )
    html += (
        "<h3>"
        + escape(result["slider"])
        + '</h3><div class="scroll"><table><thead><tr><th>Setting</th><th>Matched samples</th><th>Effect gain</th><th>Integrity</th></tr></thead><tbody>'
    )
    for side, label in (("off", "OFF"), ("on", f"ON · α {result['alpha']:g}")):
        thumbs = strip(
            (f"Seed {c['seed']}", "images/" + c[side + "_sha256"] + ".png")
            for c in matched
        )
        score = f"{result['score']:+.4f}" if side == "on" else "—"
        status = result["status"] if side == "on" else "—"
        html += f'<tr><th scope="row">{label}</th><td>{thumbs}</td><td>{score}</td><td>{status}</td></tr>'
    html += "</tbody></table></div>"
    neighbor_rows.append(info)
    js(bundle / (key + ".json"), info)
    html += (
        f'<details id="{key}"><summary>{escape(result["slider"])} · {len(result["trials"])} trials · α {result["alpha"]:g} · held-out gain {result["score"]:+.4f} · {result["status"]}</summary><p>{link("constant-retest/" + key + ".json", "Scores and image hashes")} · {link("constant-alpha.html#" + key, "Earlier samples")}</p>'
        + "".join(imgs)
        + "</details>"
    )
finalists = []
pin = here / "finalist-check/selection.json"
if is_design and pin.exists():
    selected = read(pin)
    header = '<h2 id="confirmation">Four separate production scenes</h2>'
    html += (
        header
        + "<p>The numerical leader and the highest structurally qualified recipe are reserved after ranking. Their constant alphas are compared with published timed late-cues RAW192 on the same four production scenes. The eight-case ranks stay frozen.</p>"
    )
    for entry in selected["entries"]:
        name = entry["method"]
        path = pin.parent / name / "result.json"
        if not path.exists():
            continue
        result = read(path)
        m = result["measurement"]
        reference = result["reference"]
        expected = {
            c["seed"]: c["reference_on_sha256"] for c in reference.get("cases", [])
        }
        public = dict(
            method=name,
            roles=[k for k, v in selected["roles"].items() if v == name],
            alpha=m["alpha"],
            raw_sha256=entry["raw_sha256"],
            score=m["objective_score"],
            integrity=m["status"],
            reference_score=reference.get("reference_score"),
            board_score=entry["score"],
            reference_decision=reference["decision"],
            promotion_decision=(
                "exceeds_reference"
                if entry["integrity"] == "pass"
                and m["status"] == "pass"
                and reference["decision"] == "exceeds_reference"
                else "retain_reference"
            ),
            cases=[],
        )
        html += f"<details><summary>{escape(name)} · constant α {m['alpha']:g} · {fmt(m['objective_score'])} / 100 · {m['status']}</summary>"
        for pair in m["pairs"]:
            c = pair["case"]
            off = c["off"]["output"]
            on = pair["on"]["output"]
            baseline = expected.get(c["seed"])
            public["cases"].append(
                dict(
                    seed=c["seed"],
                    prompt=c["prompt"],
                    off_sha256=off["sha256"],
                    on_sha256=on["sha256"],
                    reference_on_sha256=baseline,
                )
            )
            middle = (
                fig(
                    dict(
                        path=str(web / "images" / (baseline + ".png")), sha256=baseline
                    ),
                    "Published late-cues",
                )
                if baseline
                else ""
            )
            html += (
                f'<article><h4>Seed {c["seed"]}</h4><div class="pair boss-three">'
                + fig(off, "OFF")
                + middle
                + fig(on, "Constant ON")
                + "</div></article>"
            )
        html += "</details>"
        finalists.append(public)
    js(
        bundle / "finalist-confirmation.json",
        dict(policy=selected["protocol"], roles=selected["roles"], results=finalists),
    )
    html += (
        "<p>"
        + link(
            "constant-retest/finalist-confirmation.json",
            "Confirmation scores, seeds and hashes",
        )
        + "</p>"
    )
html += "<h2>Eight matched Final Boss images per recipe</h2>"
for r in ordered + others:
    n = r["method"]
    if n not in snippets:
        continue
    timeline = " · ".join(
        f"{a:g} ({fmt(s)})"
        for a, s in read(here / "methods" / n / objective / "selected.json")["trials"]
    )
    html += f'<details id="{n}"><summary>{escape(r["label"])} · RAW{r["step"]} · PG {r["pg"]} · α {r["alpha"]:g}</summary><p>Alpha trials in search order — {timeline}.</p><p>{link("constant-retest/" + n + "-" + objective + ".json", "Per-case measurements, pair judgments and replay hashes")}</p>{snippets[n]}</details>'
assets = [
    link("constant-retest/retest-scripts.zip", "Unattended retest scripts"),
    link("constant-retest/retest-source.json", "Source hashes"),
]
wheel_proof = bundle / "trainer-wheel.json"
if wheel_proof.is_file():
    wheel = read(wheel_proof)
    name = wheel["wheel"]
    if (
        Path(name).name != name
        or sha256((bundle / name).read_bytes()).hexdigest() != wheel["sha256"]
        or wheel["final_boss_default"] != "distinct_boss_score"
    ):
        raise ValueError("Served trainer differs from its build receipt")
    assets.extend(
        (
            link(
                "constant-retest/" + name,
                "Trainer wheel with the default design metric",
            ),
            link(
                "constant-retest/trainer-wheel.json", "Wheel SHA-256 and source commit"
            ),
        )
    )
html += "<p>" + " · ".join(assets) + "</p></main></body></html>"
if is_design and (bundle / "constant-alpha-retest.json").exists():
    html = html.replace(
        "</main>",
        "<p>"
        + link(
            "constant-retest/constant-alpha-retest.json",
            "Final ranks and sensitivity to each character",
        )
        + " · "
        + link(
            "constant-retest/constant-alpha-evidence.json.gz",
            "Offline scoring evidence",
        )
        + "</p></main>",
        1,
    )
js(
    bundle / (slug + ".json"),
    dict(
        objective=objective,
        score_display_multiplier=unit,
        replay_pg="0.12.0",
        panels=cfg["panels"],
        requests=requests,
        completed=count,
        total=len(results),
        recipes=ordered + others,
        other_sliders=neighbor_rows,
        finalist_confirmation=finalists,
    ),
)
write(web / (slug + ".html"), html)
if is_design and count == len(results) and timed_archive.is_file():
    # Keep the historical bookmark usable directly, including without JavaScript.
    mirror = html.replace(
        "<head>",
        f'<head><base href="{escape(GALLERY_URL,quote=True)}constant-retest.html">' ,
        1,
    )
    write(bookmark, mirror)
if is_design and count == len(results) and ordered and (web / "index.html").is_file():
    import re

    leader = ordered[0]
    method = leader["method"]
    category = here / "methods" / method / objective
    data = (
        read(category / "ranking-observed.json")
        if (category / "result-observed.json").exists()
        else read(category / "ranking" / (leader["alpha"].hex() + ".json"))
    )
    c = cfg["panels"]["ranking"][0]
    panel = data["rows"][c]
    middle = fig(historical(method, c), "Historical setting") if method in prior else ""
    section = (
        '<section><h2><a class="title" href="constant-retest.html">Final Boss</a></h2>'
        + f"<p>{escape(leader['label'])} · α {leader['alpha']:g} · distinct Boss {fmt(leader['score'])}/100 · structural check {leader['integrity']}</p>"
        + '<div class="pair boss-three">'
        + fig(panel["off"], c.title() + " · OFF")
        + middle
        + fig(panel["on"], "Constant ON")
        + "</div><p>"
        + link("constant-retest.html", "Rankings and matched samples")
        + " · "
        + link("final-boss-raw128.html", "Original failed RAW128")
        + "</p></section>"
    )
    idx = web / "index.html"
    updated, matches = re.subn(
        r'<section><h2><a class="title" href="(?:final-boss|constant-retest)\.html">Final Boss</a></h2>.*?</section>',
        lambda _: section,
        idx.read_text(),
        count=1,
        flags=re.S,
    )
    if matches:
        write(idx, updated)
print(f"{slug}: {count}/{len(results)}", flush=True)
