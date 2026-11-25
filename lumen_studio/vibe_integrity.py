"""Calibrated single-image integrity gate for unattended Anima training."""

from importlib.resources import files
import json
from pathlib import Path

from particle_sliders.alpha_tune_jev import _Image
from .study.judge import StudyJudge
from particle_sliders.candidate_failure import FileRecord

DEFECTS = {
    "no_readable_image": (
        "Has the render collapsed largely into texture, color, glare or noise, so a recognizable "
        "figure and coherent illustration are missing? A readable fantasy figure or a dim scene is normal."
    ),
    "split_or_duplication": (
        "Does the candidate visibly break into separate panels or repeat the main character "
        "because of an obvious render error?"
    ),
    "broken_structure": (
        "Is there a clear structural rendering failure, such as a duplicated head, fused or "
        "detached body parts, incoherent face or torso, torn geometry, or smeared shapes that "
        "erase structure? Require a specific visible defect. Stylized costumes, sharp armor, "
        "occlusion and cropping are normal."
    ),
}
STRUCTURE_DETAILS = {
    "accidental_parts": (
        "Is a body part obviously duplicated, attached in the wrong place or visibly separated "
        "from the figure by a drawing error? Count only visible evidence."
    ),
    "accidental_face": (
        "Is the visible face plainly malformed by extra eyes, misplaced features or disintegrating "
        "shapes? An intentional mask, stylized shading and expression are valid."
    ),
    "accidental_geometry": (
        "Does an obvious drawing error tear apart or smear the figure's contours so it ceases "
        "to read as a single formed figure? Coherent hair, armor and props can overlap."
    ),
}


def _check(judge, image):
    checks = judge._decide(
        (image,),
        "Inspect the single finished image for visible rendering failures.",
        {
            name: {"type": "noul", "instructions": prompt}
            for name, prompt in DEFECTS.items()
        },
    )
    if checks is None:
        return {"status": "unmeasured", "checks": None}
    details = None
    for name, probability in list(checks.items()):
        if 0.5 <= probability < 0.8:
            if name == "broken_structure":
                details = judge._decide(
                    (image,),
                    "Inspect the candidate for specific visible drawing errors.",
                    {
                        key: {"type": "noul", "instructions": text}
                        for key, text in STRUCTURE_DETAILS.items()
                    },
                )
                checks[name] = max(details.values()) if details else None
            else:
                independently = judge._decide(
                    (image,),
                    "Look at the actual visible structures before deciding.",
                    {name: {"type": "noul", "instructions": DEFECTS[name]}},
                )
                checks[name] = independently[name] if independently else None
    known = [p for p in checks.values() if p is not None]
    status = (
        "fail"
        if any(p >= 0.8 for p in known)
        else (
            "pass" if len(known) == len(checks) and max(known) < 0.5 else "unmeasured"
        )
    )
    outcome = {"status": status, "checks": checks}
    if details:
        outcome["structural_details"] = details
    return outcome


class FrameIntegrity:
    def __init__(self, receipts):
        self.judge = StudyJudge(Path(receipts), "Anima structural integrity")

    def check(self, image):
        return _check(self.judge, _Image.read(image))

    def calibrate(self):
        directory = files("lumen_studio").joinpath("calibration")
        fixtures = json.loads(directory.joinpath("fixtures.json").read_text())
        results = {}
        for name, fixture in fixtures.items():
            record = FileRecord.from_path(Path(directory.joinpath(fixture["file"])))
            if record.sha256 != fixture["sha256"]:
                raise ValueError(f"Changed calibration fixture {name}")
            result = self.check(record)
            results[name] = dict(
                sha256=record.sha256, expected=fixture["expected"], **result
            )
        return dict(
            passed=all(row["status"] == row["expected"] for row in results.values()),
            cases=results,
        )
