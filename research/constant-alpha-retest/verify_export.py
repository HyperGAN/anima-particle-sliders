"""Recompute each exported scalar score and verify all sampled SHA-256 links on the tailnet."""

from hashlib import sha256
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
import json, sys
from urllib.parse import urlsplit, urljoin
from urllib.request import urlopen
from release_tools.constant_alpha_retest import verify_search, verify_recipe
from release_tools.final_boss_design_scores import summarize
from release_tools.final_boss_visible_damage import score_case
from release_tools.verify_final_boss_designs import verify_orders

from locations import GALLERY_URL, RESULTS as root

web = root.parent / "sample-gallery"
home = GALLERY_URL
slug = sys.argv[1] if len(sys.argv) > 1 else "constant-retest"
full = "--complete" in sys.argv


def read(p):
    return json.loads(p.read_bytes())


def get(p):
    return urlopen(urljoin(home, p), timeout=45).read()


def proof(artifact):
    p = urlsplit(artifact).path.lstrip("/")
    body = get(artifact)
    if ":" not in artifact and not artifact.startswith("#"):
        assert body == (web / p).read_bytes(), p
    if p.startswith("images/"):
        assert sha256(body).hexdigest() == Path(p).stem


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = set()
        self.ids = set()
        self.method = None
        self.row_images = {}
        self.details = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "details":
            self.details += 1
        if tag == "tr" and "data-method" in attrs:
            assert self.details == 0
            self.method = attrs["data-method"]
            assert self.method not in self.row_images
            self.row_images[self.method] = []
        if tag == "img" and self.method is not None:
            self.row_images[self.method].append(attrs["src"])
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.add(attrs[key])

    def handle_endtag(self, tag):
        if tag == "tr":
            self.method = None
        if tag == "details":
            self.details -= 1


page = Links()
markup = get(slug + ".html")
assert markup == (web / (slug + ".html")).read_bytes()
page.feed(markup.decode())
for a in page.links:
    if a.startswith("#"):
        assert a[1:] in page.ids
with ThreadPoolExecutor(6) as pool:
    list(
        pool.map(proof, {a.split("#")[0] for a in page.links if not a.startswith("#")})
    )
summary = read(web / "constant-retest" / (slug + ".json"))
assert not full or summary["completed"] == summary["total"]
metric = summary["objective"]
design = metric == "distinct_boss_score"
catalog = read(root / "inventory.json")
assert set(v["method"] for v in summary["recipes"]) == {
    n for n, r in catalog.items() if r["run"] not in ("anime", "appeal-v2")
}
cfg = read(root / "panels.json")
from lumen_studio.study.objectives import protocol

if any(cfg[key] != value for key, value in protocol().items()):
    raise ValueError(
        "The research leaderboard and native Final Boss trainer use different judge protocols"
    )
ref = {r["id"]: r for r in cfg["cases"]}
model = cfg["model"]
assert not set(summary["panels"]["search"]) & set(summary["panels"]["ranking"])
receipt_root = (
    root.parent.parent
    / "anima-final-boss-vibe12-20261124/metric-calibration/design-range"
)
stores = [
    root / "ranking-reobservations",
    root / "ranking-reobservations/http-retry-1",
    root / "ranking-reobservations/http-retry-2",
    root / "judgments",
    root / "judgments/http-retry-1",
    root / "judgments/http-retry-2",
] + [receipt_root / d for d in ("receipts", "receipts-retry-1", "receipts-retry-2")]
receipts = {}


def verified_pair(p, rows):
    hashes = [rows[k]["on_sha256"] for k in p["cases"]]
    assert hashes == p["images"]
    for side in ("forward", "reverse"):
        key = p[side]["receipt"]
        if key not in receipts:
            for store in stores:
                path = store / (key + ".json")
                if path.exists():
                    candidate = read(path)
                    if candidate["status"] == "ok":
                        receipts[key] = candidate
                        break
        assert key in receipts
    verify_orders(dict(model=model, **cfg["repeat"]), p, hashes, receipts, "repeat")


from functools import lru_cache
from particle_sliders.alpha_tune_jev import JEV_URL, _digest, _response_values
from lumen_studio.study.design_score import STATE, QUESTIONS, damage

campaign = receipt_root.parent.parent


@lru_cache(maxsize=16000)
def png_size(path, expected):
    data = Path(path).read_bytes()
    assert sha256(data).hexdigest() == expected and data.startswith(
        bytes.fromhex("89504e470d0a1a0a")
    )
    return len(data)


def image_binding(img):
    return dict(
        media_type="image/png",
        sha256=img["sha256"],
        bytes=png_size(img["path"], img["sha256"]),
    )


def image_score(images, rubric, field, expected):
    if expected is None:
        return
    bindings = [image_binding(im) for im in images]
    if field == "damage":
        path = (
            campaign
            / "metric-calibration/ratings"
            / cfg["binding"]
            / (images[0]["sha256"] + ".json")
        )
        if path.exists():
            rec = read(path)
            assert (
                rec["image"] == images[0]["sha256"]
                and rec["binding"] == cfg["binding"]
                and damage(rec["answers"]) == rec["severity"] == expected
            )
            return
    else:
        path = (
            campaign
            / "boss-character-review"
            / ("-".join(i["sha256"] for i in images) + ".json")
        )
        if path.exists():
            rec = read(path)
            if rec["value"] is not None:
                assert (
                    [rec["off_sha"], rec["on_sha"]] == [i["sha256"] for i in images]
                    and rec["rubric"] == rubric
                    and rec["value"] == expected
                )
                return
    request = dict(endpoint=JEV_URL, model=model, **rubric, images=bindings)
    key = _digest(request)
    for store in stores:
        path = store / (key + ".json")
        if path.exists():
            rec = read(path)
            if rec["status"] != "ok":
                continue
            if (
                rec["request"] != request
                or _digest(rec["response"]) != rec["response_sha256"]
            ):
                raise ValueError("Unbound per-image judgment")
            values = _response_values(rec["response"], rubric["questions"])
            if (damage(values) if field == "damage" else values[field]) == expected:
                return
    raise ValueError(f"Missing {field} measurement {key}")


for row in summary["recipes"]:
    if row["state"] == "pending":
        continue
    n = row["method"]
    doc = read(web / "constant-retest" / (n + "-" + metric + ".json"))
    assert (
        doc["raw_sha256"] == catalog[n]["raw"]["sha256"] == doc["replay"]["raw_sha256"]
    )
    assert all(c["actual"] == c["expected"] for c in doc["replay"]["cases"])
    verify_recipe(doc, summary["panels"])
    if doc["ranking"] is not None:
        thumbnails = [
            "images/" + doc["ranking"]["rows"][c]["on_sha256"] + ".png"
            for c in summary["panels"]["ranking"]
        ]
        assert page.row_images[n] == thumbnails, (
            f"Missing inline matched samples for {n}"
        )
    for t in [*doc["search"], doc["ranking"]]:
        if t is None:
            continue
        assert t["gains"] == [t["alpha"]] * 10
        phase = "ranking" if t is doc["ranking"] else "search"
        evidence = root / "methods" / n / metric
        path = evidence / phase / (t["alpha"].hex() + ".json")
        if phase == "ranking" and (evidence / "result-observed.json").exists():
            assert (
                sha256(path.read_bytes()).hexdigest()
                == doc["original_measurement"]["ranking_sha256"]
            )
            path = evidence / "ranking-observed.json"
        original = read(path)
        assert (
            original["identity"]["raw"] == doc["raw_sha256"]
            and original["identity"]["phase"] == phase
        )
        for c, actual in original["rows"].items():
            exported = t["rows"][c]
            off, on = actual["off"], actual["on"]
            assert [image_binding(i)["sha256"] for i in (off, on)] == [
                exported["off_sha256"],
                exported["on_sha256"],
            ]
            image_score(
                [off, off],
                cfg["boss"],
                "boss_distinction",
                exported["original_boss_distinction"],
            )
            image_score(
                [off, on], cfg["boss"], "boss_distinction", exported["boss_distinction"]
            )
            if design:
                image_score(
                    [on],
                    dict(state=STATE, questions=QUESTIONS),
                    "damage",
                    exported["image_breakage"],
                )
        rows = t["rows"]
        names = summary["panels"]["ranking" if t is doc["ranking"] else "search"]
        assert list(rows) == names
        assert all(k["off_sha256"] == ref[c]["off"]["sha256"] for c, k in rows.items())
        if t["score"] is None:
            continue
        gain = fmean(
            r["boss_distinction"] - r["original_boss_distinction"]
            for r in rows.values()
        )
        assert gain == t["mean_gain"]
        if design:
            for p in t["pairs"]:
                verified_pair(p, rows)
            exact = summarize({c: score_case(r) for c, r in rows.items()}, t["pairs"])
            assert exact == t["design"] and exact["score"] == t["score"]
        else:
            assert gain == t["score"]
for key in ("anime", "appeal-v2"):
    study = root / "neighbor-effect-results" / key
    if not (study / "result.json").exists():
        assert not full, key
        continue
    product = read(web / "constant-retest" / (key + ".json"))
    assert product in summary["other_sliders"]
    selected = read(study / "selected.json")
    assert (
        selected["objective"] == product["objective"] == "matched_intended_effect"
        and selected["pg"] == product["pg"] == "0.12.0"
    )
    (saved,) = (r for r in catalog.values() if r["run"] == key)
    assert selected["raw"]["sha256"] == product["raw_sha256"] == saved["raw"]["sha256"]
    assert selected["gains"] == [product["alpha"]] * 10
    assert verify_search(product["trials"], product["alpha"]) == selected["alpha"]
    panels = read(root.parent / "native-prod" / key / "validation.json")
    assert len(panels["steps"]) == len(panels["confirm"]) == 4
    assert not {r["prompt"] for r in panels["steps"]} & {
        r["prompt"] for r in panels["confirm"]
    }

    def verify_measure(row, group, score):
        assert row["raw"] == selected["raw"] and row["gains"] == [row["alpha"]] * 10
        assert (
            row["profile"] == "constant"
            and [p["case"] for p in row["pairs"]] == panels[group]
        )
        gains = []
        for pair in row["pairs"]:
            off, on = pair["case"]["off"], pair["on"]
            for image in (off["output"], on["output"]):
                assert (
                    sha256(Path(image["path"]).read_bytes()).hexdigest()
                    == image["sha256"]
                )
            before, after = off["judgment"]["score"], on["judgment"]["score"]
            gain = None if None in (before, after) else after - before
            assert gain == pair["gain"]
            gains.append(gain)
        assert (None if None in gains else fmean(gains)) == row["mean_gain"] == score

    for alpha, score in product["trials"]:
        verify_measure(read(study / "search" / (alpha.hex() + ".json")), "steps", score)
    ranking = read(study / "ranking.json")
    verify_measure(ranking, "confirm", product["score"])
    assert (
        ranking["alpha"] == product["alpha"] and ranking["status"] == product["status"]
    )
    for pair, case in zip(ranking["pairs"], product["cases"], strict=True):
        assert (
            case["on_sha256"] == pair["on"]["output"]["sha256"]
            and case["off_sha256"] == pair["case"]["off"]["output"]["sha256"]
        )
ranked = [r for r in summary["recipes"] if r.get("rank") is not None]
assert [r["score"] for r in ranked] == sorted(
    (r["score"] for r in ranked), reverse=True
)
print(
    f"{summary['completed']}/{summary['total']} outcomes, {len(receipts)} unique pair judgments, {len(page.links)} served links verified"
)

if design and (full or summary.get("finalist_confirmation")):
    from lumen_studio.contracts import digest
    from lumen_studio.study.objectives import protocol_id
    from lumen_studio.study.reference import REFERENCES

    policy = read(Path(__file__).with_name("confirmation-policy.json"))
    selected = read(root / "finalist-check/selection.json")
    pinned = read(REFERENCES["final boss"])
    assert (
        selected["protocol"] == policy
        and policy["reference_raw"] == pinned["raw_sha256"]
    )
    assert pinned["distinct_boss"]["protocol"] == policy["protocol"] == protocol_id()
    public = read(web / "constant-retest/finalist-confirmation.json")
    intact = [r for r in ranked if r["integrity"] == "pass"]
    assert (
        selected["roles"]
        == public["roles"]
        == dict(
            leader=ranked[0]["method"],
            qualified=intact[0]["method"] if intact else None,
        )
    )
    assert {r["method"] for r in public["results"]} == {
        i for i in selected["roles"].values() if i is not None
    }
    assert public["results"] == summary["finalist_confirmation"]
    for row in public["results"]:
        name = row["method"]
        reference = {c["seed"]: c for c in pinned["cases"]}
        method = next(x for x in ranked if x["method"] == name)
        assert (
            all(row[k] == method[k] for k in ("raw_sha256", "alpha"))
            and row["board_score"] == method["score"]
        )
        assert [digest([c["prompt"], c["seed"]]) for c in row["cases"]] == policy[
            "cases"
        ]
        outcome = read(root / "finalist-check" / name / "result.json")
        m = outcome["measurement"]
        assert (
            outcome["reservation"] == digest(selected)
            and m["gains"] == [row["alpha"]] * 10
        )
        assert (
            m["objective"] == metric
            and m["objective_score"] == row["score"]
            and m["status"] == row["integrity"]
        )
        evidence = read(Path(m["design_observation"]))
        native = evidence["measurement"]
        assert (
            digest(native) == evidence["measurement_sha256"]
            and native["protocol"] == protocol_id()
        )
        assert (
            m["raw"]["sha256"] == row["raw_sha256"] and native["score"] == row["score"]
        )
        measurements = {k: score_case(v) for k, v in native["rows"].items()}
        if row["score"] is not None:
            assert summarize(measurements, native["pairs"]) == native["design"]
            assert row["reference_score"] == pinned["distinct_boss"]["score"]
        directory = root / "finalist-check" / name / "judgments"
        current = [directory, directory / "http-retry-1", directory / "http-retry-2"]
        stores[:0] = current
        for pair in native["pairs"]:
            if pair["repeat"] is not None:
                verified_pair(pair, native["rows"])
        for case, pair in zip(row["cases"], m["pairs"], strict=True):
            expected = reference[case["seed"]]
            orig = pair["case"]
            off, on = orig["off"]["output"], pair["on"]["output"]
            assert case["off_sha256"] == expected["off_sha256"] == off["sha256"]
            assert case["reference_on_sha256"] in (None, expected["on_sha256"])
            assert case["on_sha256"] == on["sha256"]
            values = native["rows"][digest([case["prompt"], case["seed"]])]
            assert [values["off_sha256"], values["on_sha256"]] == [
                case["off_sha256"],
                case["on_sha256"],
            ]
            for field, refs, rubric, value in (
                (
                    "boss_distinction",
                    [off, off],
                    cfg["boss"],
                    values["original_boss_distinction"],
                ),
                (
                    "boss_distinction",
                    [off, on],
                    cfg["boss"],
                    values["boss_distinction"],
                ),
                (
                    "damage",
                    [on],
                    dict(state=STATE, questions=QUESTIONS),
                    values["image_breakage"],
                ),
            ):
                if value is None:
                    continue
                request = dict(
                    endpoint=JEV_URL,
                    model=model,
                    **rubric,
                    images=[image_binding(im) for im in refs],
                )
                key = _digest(request)
                for store in current:
                    path = store / (key + ".json")
                    if path.is_file() and (response := read(path))["status"] == "ok":
                        assert (
                            response["request"] == request
                            and _digest(response["response"])
                            == response["response_sha256"]
                        )
                        votes = _response_values(
                            response["response"], rubric["questions"]
                        )
                        assert value == (
                            damage(votes) if field == "damage" else votes[field]
                        )
                        break
                else:
                    raise ValueError("Missing finalist judgment " + key)
    print(f"Verified {len(public['results'])} four-scene finalist comparisons")
