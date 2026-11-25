"""Run a pinned RAW192 retrain and its two matched-strength evaluations."""
import sys

from control import MANIFEST, PREVIOUS, ROOT, STORAGE


def run():
    import optimization_worker as harness
    from recipe_loader import prepared
    from shared_data import LABELS, parity, prepare_cache

    labels = harness.read(ROOT / "recipes.json")
    method = sys.argv[3]
    if method not in labels:
        raise ValueError("Training arm is outside the declared leaderboard")

    original_options = harness.options
    variation = harness.read(MANIFEST)["variation"]
    harness.options = lambda: original_options() | {"variation": variation}
    harness.VARIATION = variation
    harness.ROOT = ROOT
    harness.STORAGE = STORAGE
    harness.METHODS = tuple(labels)
    if method in LABELS or sys.argv[1] == "tune":
        harness.prepare_cache, harness.parity = prepare_cache, parity
    else:
        implementation, entry = prepared(method)
        harness.prepare_cache = lambda runtime, name: implementation.prepare_cache(runtime, name, entry)
        harness.parity = lambda runtime, name, cache, request: implementation.parity(runtime, name, cache, request, entry)

    import boss_review
    import image_breakage
    import tune

    boss_review.FOLDER = ROOT / "boss-character-review"
    image_breakage.CACHE = ROOT / "image-breakage-review/cases"
    tune.BREAKAGE_RULE = image_breakage.RULE = PREVIOUS / "image-breakage-rule.json"
    tune.METHODS = tuple(labels)
    harness.main()
