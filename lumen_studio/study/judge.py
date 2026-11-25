"""Retry explicit transient Jev HTTP errors without changing an earlier receipt."""

import json
import time

from particle_sliders.alpha_tune_jev import JEV_MODEL, JEV_URL, JevJudge, _digest


class StudyJudge(JevJudge):
    def _decide(self, images, state, questions):
        values = super()._decide(images, state, questions)
        if values is not None:
            return values
        binding = dict(
            endpoint=JEV_URL,
            model=JEV_MODEL,
            state=state,
            questions=questions,
            images=[image.identity() for image in images],
        )
        key = _digest(binding)
        directory = self.receipts
        for attempt, delay in enumerate((1, 3), start=1):
            path = directory / f"{key}.json"
            if not path.is_file():
                # An in-flight request can still have reached the service.
                return None
            receipt = json.loads(path.read_text())
            if receipt["status"] != "http_error" or receipt["response"].get(
                "http_status"
            ) not in (408, 429, 500, 502, 503, 504):
                return None
            directory = self.receipts / f"http-retry-{attempt}"
            if not (directory / f"{key}.json").is_file():
                time.sleep(delay)
            values = JevJudge(directory, self.concept)._decide(images, state, questions)
            if values is not None:
                return values
        return None
