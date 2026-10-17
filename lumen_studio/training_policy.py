"""Pure dispatch policy for archived native training commands."""

LEGACY_TRAINING_MESSAGE = (
    "Historical gmix native training dispatch is unavailable. "
    "Use the explicit externally admitted lumen_studio.pg09_training.PG09Trainer "
    "after review of the exact installed shared package and native cache lineage. "
    "Legacy EMA, pilot, microbatch and resume protocols are not PG09 protocols."
)


class LegacyTrainingUnavailable(RuntimeError):
    pass


def legacy_training_unavailable(entrypoint):
    raise LegacyTrainingUnavailable(f"{entrypoint}: {LEGACY_TRAINING_MESSAGE}")
