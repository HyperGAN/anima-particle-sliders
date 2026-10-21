"""One process owns Anima GPU work, with update-boundary training handoffs."""
from contextlib import contextmanager
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .contracts import atomic_json, file_hash
from .store import Store
from .models import ANIMA, get_model


class OwnershipError(RuntimeError):
    pass


class GpuLease:
    def __init__(self, directory, device):
        self.directory, self.device = Path(directory), device
        self.handle = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.handle = (self.directory / f"anima-{self.device.replace(':', '-')}.lock").open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            raise OwnershipError("Another Anima coordinator owns this GPU") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        return self

    def __exit__(self, *exc):
        if self.handle:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


def choose_work(controls, has_images, has_training):
    if controls["paused"]:
        return "render" if has_images else None
    if has_images and (not has_training or controls["render_burst"] < 4) and (
            not has_training or controls["training_updates"] >= 20):
        return "render"
    if has_training:
        return "train"
    return "render" if has_images else None


class Coordinator:
    def __init__(self, directory, model_dir, *, gpu="1", runtime_factory=None):
        self.directory, self.model_dir = Path(directory).resolve(), Path(model_dir).resolve()
        self.store = Store(self.directory / "studio.sqlite3")
        self.gpu = str(gpu)
        self.device = "cpu" if gpu == "cpu" else "cuda:0"
        self.runtime_factory = runtime_factory
        self.runtime = None
        self.loaded_checkpoints = None
        self.loaded_model = None
        self.stopping = False
        (self.directory / "images").mkdir(parents=True, exist_ok=True)

    def load_runtime(self, training=False, model_id=ANIMA.id, checkpointing=None):
        if self.runtime is not None and self.loaded_model != model_id:
            self.release_runtime()
        if self.runtime is None:
            from .runtime import create_runtime
            factory = self.runtime_factory or create_runtime
            self.runtime = factory(self.model_dir, self.device, backend=get_model(model_id).backend,
                                   checkpointing=training if checkpointing is None else checkpointing)
            self.loaded_model = model_id
        return self.runtime

    def release_runtime(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
        self.loaded_checkpoints = None
        self.loaded_model = None
        gc.collect()
        if self.device != "cpu":
            import torch
            torch.cuda.empty_cache()

    def render_one(self):
        from .particles import ParticleAdapter
        from .inference_controls import set_inference_mix
        from .runtime import Cancelled
        from PIL.PngImagePlugin import PngInfo
        job = self.store.claim()
        if job is None:
            return False
        ident, payload = job["id"], job["payload"]
        try:
            checkpoints = payload["checkpoints"]
            if self.runtime is not None and checkpoints != self.loaded_checkpoints:
                self.release_runtime()
            runtime = self.load_runtime(model_id=payload.get("model", ANIMA.id))
            if runtime.identity != payload["model_identity"]:
                if checkpoints:
                    raise ValueError("Queued model identity changed")
                from .provenance import single_image_runtime_compatibility
                single_image_runtime_compatibility(self.model_dir, payload["model_identity"], runtime.identity)
            if checkpoints != self.loaded_checkpoints:
                catalog = {c["sha256"]: c for c in self.store.catalog()}
                for variation, sha in checkpoints.items():
                    checkpoint = catalog[sha]
                    if file_hash(checkpoint["path"]) != sha:
                        raise ValueError("Selected checkpoint file changed")
                    adapter = ParticleAdapter(runtime.transformer).to(runtime.device)
                    adapter.load_export(checkpoint["path"], model_identity=runtime.identity)
                    adapter.eval().requires_grad_(False)
                    runtime.mixer.add(variation, adapter)
                self.loaded_checkpoints = checkpoints.copy()
            strengths = set_inference_mix(runtime.mixer, payload["mix"], payload["energy"])
            if strengths != payload["strengths"]:
                raise ValueError("Queued mixing contract changed")
            start = time.monotonic()
            image = runtime.render(payload["prompt"], payload["seed"], payload["width"], payload["height"],
                payload["steps"], progress=lambda p: self.store.progress(ident, p),
                cancelled=lambda: self.store.job(ident)["status"] == "cancelling")
            metadata = dict(**payload, effective_prompt=payload["prompt"], job_id=ident,
                            group_id=job["group_id"], render_seconds=time.monotonic() - start)
            if runtime.identity != payload["model_identity"]:
                metadata.update(requested_model_identity=payload["model_identity"], model_identity=runtime.identity)
            path = self.directory / "images" / f"{ident}.png"
            temp = path.with_suffix(".tmp")
            png = PngInfo()
            png.add_text("anima", json.dumps(metadata, sort_keys=True))
            image.save(temp, format="PNG", pnginfo=png)
            temp.replace(path)
            if not self.store.finish(ident, path, metadata):
                path.unlink(missing_ok=True)
            else:
                atomic_json(path.with_suffix(".json"), metadata)
        except Cancelled as exc:
            self.store.fail(ident, exc, cancelled=True)
        except Exception as exc:
            self.store.fail(ident, f"{type(exc).__name__}: {exc}")
            self.release_runtime()
        controls = self.store.controls()
        self.store.control(render_burst=controls["render_burst"] + 1)
        return True

    def train_quantum(self, run):
        from .training_policy import legacy_training_unavailable
        legacy_training_unavailable("Studio legacy training quantum")

    def _reject_legacy_runs(self):
        from .training_policy import LEGACY_TRAINING_MESSAGE
        for run in self.store.runs():
            if run["status"] == "queued":
                self.store.update_run(run["id"], "failed", run["step"],
                                      error=LEGACY_TRAINING_MESSAGE)

    def enough_memory(self, training):
        if self.gpu == "cpu" or self.runtime_factory:
            return True
        if self.runtime is None:
            # Diagnostics can release their last references after Runtime.close
            # returns. Return this process's unused allocator cache before
            # treating it as another application's occupied GPU memory.
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        result = subprocess.run(["nvidia-smi", "-i", self.gpu,
            "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
        # Do not evict or terminate another application's allocations.
        return int(result.stdout.strip()) >= (22000 if training else 10000)

    def run(self, once=False):
        if self.gpu != "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        # A lock shared across all Studio data directories, keyed by physical GPU.
        lease_key = self.gpu
        if self.gpu != "cpu":
            lease_key = subprocess.check_output(["nvidia-smi", "-i", self.gpu,
                "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        with GpuLease(Path("/tmp"), lease_key):
            self.store.recover()
            self.store.control(owner=dict(pid=os.getpid(), gpu=self.gpu, state="idle"))
            try:
                while not self.stopping:
                    queued = any(j["status"] == "queued" for j in self.store.jobs())
                    # Hold the normal ownership lease before changing queue state.
                    # Recovered legacy jobs fail before memory probes or native loads;
                    # image work keeps its existing inference-only dispatch.
                    self._reject_legacy_runs()
                    work = choose_work(self.store.controls(), queued, False)
                    if work and self.runtime is None and not self.enough_memory(work == "train"):
                        self.store.control(owner=dict(pid=os.getpid(), gpu=self.gpu, state="waiting for GPU memory"))
                        if once:
                            return
                        time.sleep(2)
                        continue
                    if work == "render":
                        self.store.control(owner=dict(pid=os.getpid(), gpu=self.gpu, state="rendering"))
                        self.render_one()
                    else:
                        if self.runtime is not None:
                            self.release_runtime()
                        owner = self.store.controls()["owner"]
                        if not isinstance(owner, dict) or owner.get("state") != "idle":
                            self.store.control(owner=dict(pid=os.getpid(), gpu=self.gpu, state="idle"))
                        if once:
                            return
                        time.sleep(.5)
                    if once:
                        return
            finally:
                self.release_runtime()
                self.store.control(owner=None)
