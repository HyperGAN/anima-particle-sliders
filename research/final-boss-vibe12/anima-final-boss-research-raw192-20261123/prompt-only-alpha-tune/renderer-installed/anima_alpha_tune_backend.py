"""Native alpha-tune binding for the admitted Anima Turbo RAW192 experiment."""

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from prepare import guarded, render
from replay import loaded
from particle_sliders.candidate_failure import FileRecord


class Renderer:
    def __init__(self, runtime, variation):
        self.runtime = runtime
        self.variation = variation
        sigmas = runtime.scheduler(10).sigmas.tolist()
        self.interval_widths = tuple(a - b for a, b in zip(sigmas, sigmas[1:]))
        self.sampler = {"model": runtime.identity, "sigmas": sigmas, "steps": 10,
                        "width": 512, "height": 512, "cfg": 1.}

    def render(self, request, gains, output):
        if len(gains) != 10 or any(gain < 0 or gain > 5 for gain in gains):
            raise ValueError("Anima supports ten effective strengths in [0, 5]")
        render(self.runtime, request["prompt"], request["seed"], output.parent,
               gains=list(gains), variation=self.variation)
        return FileRecord.from_path(output)


@contextmanager
def open_renderer(checkpoint, options):
    import torch
    from lumen_studio.backends.anima import TurboRuntime

    guarded()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(19 * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = TurboRuntime(Path(options["model"]), device="cuda:0", verify=True)
    try:
        variation = loaded(runtime, asdict(checkpoint), options["step"])
        yield Renderer(runtime, variation)
    finally:
        runtime.close()
