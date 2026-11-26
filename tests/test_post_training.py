"""Software cases for forking a trained RAW and replacing some Boss teachers."""

import json
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from lumen_studio.cache import TargetCache, save_tensor_file
from lumen_studio.contracts import atomic_json, digest, file_hash
from lumen_studio.study import boss_targets, training
from lumen_studio_scoped.training import Checkpoint


class Runtime:
    """Deterministic stand-in for the frozen model; a prompt only shifts the velocity."""

    device = torch.device("cpu")
    identity = {"model": "software"}

    def __init__(self):
        self.mixer = SimpleNamespace(scales=lambda values: torch.no_grad())

    def encode(self, prompt):
        return torch.full((1, 2, 4), sum(prompt.encode()) % 97 / 97)

    def noise(self, seed, width, height):
        return torch.randn(1, 4, 1, 8, 8, generator=torch.Generator().manual_seed(seed))

    def scheduler(self, steps):
        return SimpleNamespace(timesteps=torch.linspace(1000, 100, steps))

    def predict(self, latent, timestep, embedding):
        return latent * 0.1 + embedding.mean() + float(timestep) / 1e4

    def step(self, scheduler, velocity, timestep, latent):
        return latent - 0.1 * velocity


def late_cues_cache(tmp_path, runtime, scenes=3):
    source = tmp_path / "source"
    source.mkdir()
    recipe = dict(method="late-cues-v1", rows={})
    rows = []
    for index in range(scenes):
        name = f"scene{index}"
        Image.new("L", (16, 16), 40 * (index + 1)).save(source / f"{name}.png")
        images = []
        for side in ("neutral", "positive"):
            path = source / f"{name}-{side}.png"
            Image.new("RGB", (4, 4), (index, len(side), 9)).save(path)
            images.append(dict(output=dict(prompt=f"{name} {side}", seed=index, path=str(path),
                                           sha256=file_hash(path))))
        recipe["rows"][images[1]["output"]["sha256"]] = dict(
            neutral_sha256=images[0]["output"]["sha256"], matte=f"{name}.png",
            matte_sha256=file_hash(source / f"{name}.png"), explicit=f"{name} explicit cue")
        rows.append(dict(id=name, neutral=f"{name} neutral", positive=f"{name} positive",
                         seeds=[index], selection=dict(neutral=images[0], positive=images[1])))
    identity = dict(model=runtime.identity, split="train", variation="boss", manifest_sha256="c" * 64,
                    layout="final-boss/late-cues-v1", field_recipe=recipe)
    fingerprint = digest(identity)
    shards = []
    for row in rows:
        neutral = runtime.encode(row["neutral"])
        records = boss_targets.teacher_records(runtime, neutral, runtime.encode(row["positive"]), row["seeds"][0])
        context = boss_targets.field_context(runtime, recipe, source, row, records[0]["latent"].shape[-2:])
        for record in records:
            record["positive"] = boss_targets.target_field(
                runtime, record, record["neutral"], record["positive"], context)
        path = source / f"{row['id']}.pt"
        save_tensor_file(path, dict(fingerprint=fingerprint, row=row, seed=row["seeds"][0],
                                    embedding=neutral, positive_embedding=runtime.encode(row["positive"]),
                                    records=records))
        shards.append(dict(path=path.name, sha256=file_hash(path), count=20, row=row["id"], seed=row["seeds"][0]))
    atomic_json(source / "index.json", dict(identity=identity, input_fingerprint=fingerprint, shards=shards,
                                            fingerprint=digest(dict(input=fingerprint, shards=shards))))
    return TargetCache(source)


def replacement(tmp_path, seed=1, prompt="scene1 cleaner boss"):
    path = tmp_path / "cleaner.png"
    Image.new("RGB", (4, 4), (200, 3, 3)).save(path)
    return dict(positive=prompt, explicit=prompt + " cue",
                selected=dict(output=dict(prompt=prompt, seed=seed, path=str(path), sha256=file_hash(path))))


def test_replacing_one_teacher_keeps_every_other_field_and_the_neutral_path(tmp_path):
    runtime = Runtime()
    parent = late_cues_cache(tmp_path, runtime)
    new = replacement(tmp_path)
    cache = boss_targets.replace_boss_teachers(runtime, parent, tmp_path / "cleaner", {"scene1": new})
    assert len(cache) == len(parent) == 60 and cache.index["fingerprint"] != parent.index["fingerprint"]
    note = cache.index["identity"]["teacher_replacement"]
    assert note["parent_cache"] == parent.index["fingerprint"] and list(note["rows"]) == ["scene1"]
    changed = 0
    for index in range(60):
        old, fresh = parent[index], cache[index]
        assert torch.equal(old["embedding"], fresh["embedding"])
        if old["row"]["id"] != "scene1":
            assert all(torch.equal(old[k], fresh[k]) for k in ("latent", "neutral", "positive"))
            assert old["row"] == fresh["row"]
            continue
        assert fresh["row"]["positive"] == new["positive"]
        assert fresh["row"]["selection"]["replaced"] == old["row"]["selection"]["positive"]
        assert torch.equal(old["neutral"], fresh["neutral"]) or old["trajectory"] == "positive"
        assert (old["trajectory"] == "neutral") == torch.equal(old["latent"], fresh["latent"]) or old["position"] == 0
        assert not torch.equal(old["positive"], fresh["positive"])
        changed += 1
    assert changed == 20
    # The stored target is the recipe applied to the new teacher, as native admission recomputes it.
    record = cache[27]
    context = boss_targets.field_context(runtime, cache.index["identity"]["field_recipe"],
                                         cache.directory, record["row"], record["latent"].shape[-2:])
    teacher = runtime.predict(record["latent"], record["timestep"], runtime.encode(new["positive"]))
    assert torch.equal(record["positive"],
                       boss_targets.target_field(runtime, record, record["neutral"], teacher, context))
    again = boss_targets.replace_boss_teachers(runtime, parent, tmp_path / "cleaner", {"scene1": new})
    assert again.index == cache.index


def test_replacement_refuses_unreproduced_anchors_and_mismatched_teacher_images(tmp_path):
    runtime = Runtime()
    parent = late_cues_cache(tmp_path, runtime)
    with pytest.raises(ValueError):
        boss_targets.replace_boss_teachers(runtime, parent, tmp_path / "all",
            {f"scene{i}": replacement(tmp_path, i, f"scene{i} b") for i in range(3)})
    with pytest.raises(ValueError):
        boss_targets.replace_boss_teachers(runtime, parent, tmp_path / "seed",
                                           {"scene1": replacement(tmp_path, seed=5)})
    drifted = Runtime()
    drifted.predict = lambda latent, timestep, embedding: latent * 0.2 + embedding.mean()
    with pytest.raises(ValueError, match="not reproduced"):
        boss_targets.replace_boss_teachers(drifted, parent, tmp_path / "drift", {"scene1": replacement(tmp_path)})
    assert not any((tmp_path / name).exists() for name in ("all", "seed", "drift"))


def test_study_training_forks_once_then_resumes_the_child(tmp_path, monkeypatch):
    seen = []

    def inspect(*args, **kwargs):
        seen.append({k: kwargs[k] for k in ("mode", "cohort_id", "fork", "resume", "seed")})
        raise RuntimeError("inspected")

    monkeypatch.setattr(training, "open_training", inspect)
    cache = SimpleNamespace(index={"identity": {"variation": "x"}, "fingerprint": "fixture"})
    rates = dict(g=0.001, d=0.01, particle=0.03)
    parent = Checkpoint(str(tmp_path / "RAW000512.safetensors"), 1, "a" * 64)

    def run(root, **options):
        with pytest.raises(RuntimeError, match="inspected"):
            training.train(SimpleNamespace(root=root, runtime=object()), cache, {}, rates, 48, "fit", **options)
        return seen[-1]

    fresh = run(tmp_path / "fresh")
    child = run(tmp_path / "child", parent=parent)
    assert (fresh["mode"], fresh["fork"]) == ("fresh", None)
    assert (child["mode"], child["fork"], child["resume"]) == ("fork", parent, None)
    assert child["cohort_id"] != fresh["cohort_id"] and child["seed"] == 1
    replicate = run(tmp_path / "replicate", parent=parent, seed=2)
    assert replicate["seed"] == 2 and replicate["cohort_id"] not in (child["cohort_id"], fresh["cohort_id"])
    saved = dict(path=str(tmp_path / "FULL000012.pt"), bytes=1, sha256="b" * 64)
    arm = training.arm_path(tmp_path / "child", rates)
    atomic_json(arm / "cursor.json", dict(step=12, full=saved, rates=rates))
    resumed = run(tmp_path / "child", parent=parent)
    assert (resumed["mode"], resumed["fork"]) == ("resume", None)
    assert resumed["resume"] == Checkpoint(**saved) and resumed["cohort_id"] == child["cohort_id"]


def test_working_score_keeps_boss_ability_and_zeroes_broken_drawings(tmp_path):
    from lumen_studio.study import working_score as ws

    picture = Image.merge("RGB", [Image.effect_noise((512, 512), 80) for _ in range(3)])
    picture.save(tmp_path / "image.png")
    # Noise does not compress, so every tile exceeds the reader's limit and is cut into pieces.
    crops = ws.tiles(tmp_path / "image.png")
    assert len(crops) == 4 and all(len(pieces) == 4 for pieces in crops)
    for pieces, (left, top) in zip(crops, ((0, 0), (256, 0), (0, 256), (256, 256)), strict=True):
        for raw, (x, y) in zip(pieces, ((0, 0), (96, 0), (0, 96), (96, 96)), strict=True):
            with Image.open(ws.BytesIO(raw)) as piece:
                # Native pixels: each piece equals the same region of the source.
                box = (left + x, top + y, left + x + 160, top + y + 160)
                assert piece.size == (160, 160) and piece.tobytes() == picture.crop(box).tobytes()
            assert len(raw) <= ws.TILE_BYTES
    Image.new("RGB", (512, 512), (20, 30, 40)).save(tmp_path / "flat.png")
    flat = ws.tiles(tmp_path / "flat.png")
    assert [len(pieces) for pieces in flat] == [1, 1, 1, 1]
    with Image.open(ws.BytesIO(flat[3][0])) as tile:
        assert tile.size == (256, 256)
    with pytest.raises(ValueError):
        Image.new("RGB", (200, 200)).save(tmp_path / "small.png")
        ws.tiles(tmp_path / "small.png")
    assert ws.quality(0.05, 0.05, 1.0) == 1.0
    assert ws.quality(0.05 + ws.SPAN, 0.05, 1.0) == 0.0
    assert ws.quality(0.05, 0.05, 0.2) == 0.2  # A split figure limits a clean drawing.
    assert ws.quality(None, 0.05, 1.0) is None
    # Ability is rewarded as before; a broken drawing earns nothing, whatever its design.
    strong, weak = ws.score_case(0.9, 0.1, 1.0), ws.score_case(0.6, 0.1, 1.0)
    assert strong > weak > 0 and ws.score_case(0.95, 0.1, 0.0) == 0
    cases = dict(a=strong, b=weak)
    distinct = ws.working_score(cases, [dict(cases=["a", "b"], repeat=0.0)])
    repeated = ws.working_score(cases, [dict(cases=["a", "b"], repeat=1.0)])
    assert distinct > repeated > 0
    assert ws.working_score(dict(a=strong, b=None), [dict(cases=["a", "b"], repeat=0.0)]) is None
