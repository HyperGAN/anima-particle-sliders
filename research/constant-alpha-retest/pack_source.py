"""Package the registered study driver and metric source for an equipped study host."""

import json
from hashlib import sha256
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo
from locations import CODE, STUDY

public = STUDY / "sample-gallery/constant-retest"
public.mkdir(exist_ok=True, parents=True)
names = (
    "snapshot.py",
    "sample_rows.py",
    "markdown_samples.py",
    "finalist_check.py",
    "confirmation-policy.json",
    "run.py",
    "locations.py",
    "gpu_guard.py",
    "worker.py",
    "extra.py",
    "neighbors.py",
    "inventory.py",
    "evaluate.py",
    "publish.py",
    "verify_export.py",
    "panels.json",
    "recover_ranking.py",
    "pack_source.py",
)
paths = {"cohort-retest/" + name: CODE / name for name in names}
for name in (
    "release_tools/constant_alpha_retest.py",
    "lumen_studio/study/scalar_search.py",
    "lumen_studio/study/design_score.py",
    "lumen_studio/study/objectives.py",
    "configs/anima/final-boss-design.json",
):
    paths["product/" + name] = STUDY / "product" / name
outfile = public / "retest-scripts.zip"
with ZipFile(outfile.with_suffix(".pending"), "w", ZIP_DEFLATED) as z:
    for name, p in paths.items():
        info = ZipInfo(name, date_time=(2026, 11, 25, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        z.writestr(info, p.read_bytes())
outfile.with_suffix(".pending").replace(outfile)
info = dict(
    files={name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
    sha256=sha256(outfile.read_bytes()).hexdigest(),
)
dest = public / "retest-source.json"
dest.with_suffix(".pending").write_text(json.dumps(info, indent=2) + "\n")
dest.with_suffix(".pending").replace(dest)
