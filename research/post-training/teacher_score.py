"""Read a candidate teacher image the way the board reads a slider's output."""

from pathlib import Path
from statistics import fmean

from particle_sliders.alpha_tune_jev import _Image
from particle_sliders.candidate_failure import FileRecord

from lumen_studio.study import working_score as ws
from lumen_studio.study.judge import StudyJudge
from lumen_studio.study.objectives import protocol

from campaign import ROOT

judge = StudyJudge(ROOT / "working-metric" / "judgments", "Final Boss")
reader = ws.DrawingReader(ROOT / "working-search" / "drawing-receipts")
cfg = protocol()


def picture(record):
    return _Image.read(FileRecord.from_path(Path(record["path"])))


def value(answer, key):
    return None if answer is None else answer[key]


def measure(off, on):
    a, b = picture(off), picture(on)
    paired = judge._decide((a, b), ws.PAIR_STATE, ws.PAIR_QUESTIONS)
    # One neutral image never gets an answer for its own Boss reading. The score floors that
    # reading at BOSS_FLOOR anyway, and every answered scene reads far below it.
    original = value(judge._decide((a, a), **cfg["boss"]), "boss_distinction")
    row = dict(
        image=on,
        original=ws.BOSS_FLOOR if original is None else original,
        original_answered=original is not None,
        boss=value(judge._decide((a, b), **cfg["boss"]), "boss_distinction"),
        broken=reader.broken(on["path"]),
        off_broken=reader.broken(off["path"]),
        one_figure=value(paired, "one_figure"),
        same_person=value(paired, "same_person"),
        same_scene=value(paired, "same_scene"),
        figure=value(judge._decide((b,), ws.FIGURE_STATE, ws.FIGURE_QUESTIONS), "figure"),
        recolored=value(judge._decide((a, b), ws.RECOLOR_STATE, ws.RECOLOR_QUESTIONS), "recolored"),
    )
    row["quality"] = ws.quality(row["broken"], row["off_broken"], row["one_figure"], row["figure"])
    row["case"] = ws.score_case(row["boss"], row["original"], row["quality"], row["recolored"])
    return row


def repeat(left, right):
    x, y = picture(left), picture(right)
    values = [value(judge._decide(order, **cfg["repeat"]), "repeat") for order in ((x, y), (y, x))]
    return None if None in values else fmean(values)
