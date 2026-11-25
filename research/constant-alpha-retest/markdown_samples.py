"""Put repository-local matched contact sheets directly in the default Markdown board."""

from hashlib import sha256
from io import BytesIO
from PIL import Image
from locations import STUDY
import json


def update():
    root = STUDY / "product/docs"
    web = STUDY / "sample-gallery"
    target = root / "final-boss-raw192/vibe12/constant-samples"
    target.mkdir(exist_ok=True)
    bundle = web / "constant-retest"
    read = lambda path: json.loads(path.read_bytes())
    board = read(bundle / "constant-retest.json")
    if board["completed"] != board["total"]:
        return

    def sheet(hashes, description):
        image = Image.new("RGB", (len(hashes) * 100, 100))
        for index, h in enumerate(hashes):
            data = (web / "images" / (h + ".png")).read_bytes()
            assert sha256(data).hexdigest() == h
            with Image.open(BytesIO(data)) as source:
                image.paste(
                    source.convert("RGB").resize((100, 100), Image.Resampling.LANCZOS),
                    (index * 100, 0),
                )
        stream = BytesIO()
        image.save(stream, "WEBP", quality=85, method=5)
        value = stream.getvalue()
        file = target / (sha256(value).hexdigest() + ".webp")
        if not file.exists():
            file.write_bytes(value)
        url = file.relative_to(root).as_posix()
        return f"[![{description}]({url})]({url})"

    rows = [
        "| Rank | Final Boss training recipe | Constant α | Matched samples | Distinct Boss / 100 | Structural check |",
        "| ---: | --- | ---: | --- | ---: | --- |",
    ]
    names = board["panels"]["ranking"]
    ranking = [v for v in board["recipes"] if v.get("rank") is not None]
    for i, row in enumerate(ranking):
        data = read(bundle / (row["method"] + "-distinct_boss_score.json"))["ranking"][
            "rows"
        ]
        if i == 0:
            preview = sheet(
                [data[k]["off_sha256"] for k in names], "Eight matched OFF images"
            )
            rows.append(f"| — | Matched baseline | OFF | {preview} | — | — |")
        preview = sheet(
            [data[k]["on_sha256"] for k in names], row["label"] + " on eight characters"
        )
        rows.append(
            f"| {row['rank']} | {row['label']} | {row['alpha']:g} | {preview} | {100 * row['score']:.2f} | {row['integrity']} |"
        )
    page = root / "leaderboard.md"
    content = page.read_text()
    a = content.index("| Rank | Final Boss training recipe |")
    b = content.index("\n\n", a)
    page.write_text(content[:a] + "\n".join(rows) + content[b:])


if __name__ == "__main__":
    update()
