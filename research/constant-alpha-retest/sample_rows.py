"""Shared presentation of matched images in the leaderboard and duration study."""

from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile
from PIL import Image
from locations import STUDY

CSS = """<style>
main,header{max-width:1540px} .scroll{overflow:auto;border:1px solid #41434a;border-radius:7px;max-width:100%}
.scroll table{margin:0;width:100%;font-size:14px}.scroll td,.scroll th{padding:6px 8px}
.scroll thead{background:#292b30}.scroll th[scope=row]{min-width:145px;max-width:260px}
.images{display:flex;gap:4px;width:max-content}.images a,.images img{display:block}
.images img{width:58px;height:58px;object-fit:cover;border-radius:3px}.images a:focus-visible{outline:3px solid gold}
.scroll tr:hover{background:#242529}.scroll .ref{background:#202328}
@media(max-width:650px){.scroll th[scope=row]{min-width:105px}.scroll td,.scroll th{padding:4px}.images img{width:48px;height:48px}}
</style>"""


def strip(items):
    links = []
    root = STUDY / "sample-gallery"
    directory = root / "sample-thumbs"
    directory.mkdir(exist_ok=True)
    for label, url in items:
        source = root / url
        name = source.stem + ".webp"
        thumb = directory / name
        if not thumb.is_file():
            with (
                Image.open(source) as original,
                NamedTemporaryFile(dir=directory, delete=False) as tmp,
            ):
                original.convert("RGB").resize(
                    (116, 116), Image.Resampling.LANCZOS
                ).save(tmp, "WEBP", quality=82)
            Path(tmp.name).chmod(0o644)
            Path(tmp.name).replace(thumb)
        title, path = escape(label, quote=True), escape(url, quote=True)
        small = escape(thumb.relative_to(root).as_posix(), quote=True)
        links.append(
            f'<a href="{path}" title="{title}"><img loading="lazy" width="58" height="58" alt="{title}" src="{path}" srcset="{small} 1x, {small} 2x"></a>'
        )
    return '<div class="images">' + "".join(links) + "</div>"
