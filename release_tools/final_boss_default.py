"""Resolve the Final Boss release from the first row of the distinct-design board."""


def select_default(panel, board, catalog):
    winner = panel["rows"][0]
    if winner["score"] != max(row["score"] for row in panel["rows"]):
        raise ValueError("The published distinct-design board is out of order")
    recipe = winner["method"]
    artifact = catalog["checkpoints"][recipe]
    method = board["methods"][recipe]
    if (artifact["sha256"] != method["raw_sha256"]
            or artifact["training_cache_sha256"] != method["training_cache_sha256"]):
        raise ValueError("The released artifacts differ from the measured training run")
    if "target_index" not in artifact:
        raise ValueError(f"Publish the target cache for the new winner {recipe} before releasing the board")
    settings = [s for s in board["settings"] if all(s[k] == winner[k] for k in ("method", "profile", "alpha"))]
    if len(settings) != 1:
        raise ValueError("Expected a unique measured setting for the winning row")
    setting = settings[0]
    if winner["inputs"] != {k: v["on_sha256"] for k, v in setting["cases"].items()}:
        raise ValueError("The winning design measurements and replay images differ")
    case = panel["cases"][0]
    model = board["teacher"]["model"]
    default = dict(recipe=recipe, label=method["label"], step=board["training_updates"], weights=board["weights"],
                resolution=512, steps=10, cfg=1, checkpoint_repository=catalog["repository"],
                checkpoint_revision=catalog["revision"], checkpoints={recipe: artifact},
                profile=winner["profile"], alpha=winner["alpha"], gains=setting["gains"], sigmas=method["sigmas"],
                selection_metric="distinct_boss_score", development_score=winner["score"],
                development_cases=panel["cases"], independent_score=None,
                training_model_identity=model, qualified_replay_runtime_sha256=[model["runtime_sha256"]],
                reference_case=dict(label=case, **board["evaluation_requests"][case]),
                reference_png_sha256=winner["inputs"][case])
    qualified = catalog.get("qualified_replay_models", {}).get(artifact["sha256"])
    if qualified:
        default["qualified_replay_models"] = qualified
    return default
