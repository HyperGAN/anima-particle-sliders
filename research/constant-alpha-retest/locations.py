"""Separate committed scripts from the equipped study and its mounted campaign archive."""

import os
from pathlib import Path

CODE = Path(__file__).resolve().parent
STUDY = (
    Path(os.environ["ANIMA_RETEST_STUDY"]).resolve()
    if "ANIMA_RETEST_STUDY" in os.environ
    else (CODE.parents[2] if CODE.parent.name == "research" else CODE.parent)
)
RESULTS = STUDY / "cohort-retest"

GALLERY_URL = os.environ.get("ANIMA_GALLERY_URL", "http://127.0.0.1:8796").rstrip("/") + "/"
TIMED_GALLERY_URL = os.environ.get("ANIMA_TIMED_GALLERY_URL", "http://127.0.0.1:8765").rstrip("/") + "/"
