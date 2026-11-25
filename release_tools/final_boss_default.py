"""Resolve the Final Boss release from the first row of the distinct-design board."""


def scheduled_gains(profile, alpha, sigmas):
    """The per-interval gains that a named schedule and alpha resolve to on this sampler grid."""
    from particle_sliders.strength_schedule import interval_strengths

    from math import isfinite

    if (len(sigmas) != 11 or sigmas[0] != 1 or sigmas[-1] != 0
            or any(isinstance(value, bool) or not isfinite(value) for value in sigmas)
            or any(start <= end for start, end in zip(sigmas, sigmas[1:]))):
        raise ValueError("Expected the ten decreasing native Euler intervals from 1 to 0")
    widths = [start - end for start, end in zip(sigmas, sigmas[1:])]
    return list(interval_strengths(alpha, widths, profile=profile))


# Anima applies effective strengths in [0, 5]. See ``scheduled_render.step_gains``.
STRONGEST_GAIN = 5.0


def strongest_alpha(profile, sigmas):
    """The largest alpha whose every interval gain Anima can apply under this schedule."""
    from math import nextafter

    alpha = STRONGEST_GAIN / max(scheduled_gains(profile, 1.0, sigmas))
    while max(scheduled_gains(profile, alpha, sigmas)) > STRONGEST_GAIN:
        alpha = nextafter(alpha, 0.0)
    return alpha


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
    # Alpha, schedule, sampler grid and resolved gains are released as one setting.
    if setting["gains"] != scheduled_gains(winner["profile"], winner["alpha"], method["sigmas"]):
        raise ValueError("The winning gains differ from its schedule and alpha on the recorded sampler grid")
    if winner["inputs"] != {k: v["on_sha256"] for k, v in setting["cases"].items()}:
        raise ValueError("The winning design measurements and replay images differ")
    case = panel["cases"][0]
    model = board["teacher"]["model"]
    return dict(recipe=recipe, label=method["label"], step=board["training_updates"], weights=board["weights"],
                resolution=512, steps=10, cfg=1, checkpoint_repository=catalog["repository"],
                checkpoint_revision=catalog["revision"], checkpoints={recipe: artifact},
                profile=winner["profile"], alpha=winner["alpha"], gains=setting["gains"], sigmas=method["sigmas"],
                selection_metric="distinct_boss_score", development_score=winner["score"],
                development_cases=panel["cases"], independent_score=None,
                training_model_identity=model, qualified_replay_runtime_sha256=[model["runtime_sha256"]],
                reference_case=dict(label=case, **board["evaluation_requests"][case]),
                reference_png_sha256=winner["inputs"][case])
