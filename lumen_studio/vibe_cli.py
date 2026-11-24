"""Collect teacher targets with Anima's named or free-text vibe workflows."""
import argparse
import json
from pathlib import Path

from .vibe import AnimaVibe, main as free_text_main, workflow_graphs
from .vibe_curation import collect_portfolio, portfolio_manifest
from .vibe_profiles import profile_for
from .vibe_teacher_selection import best_teacher_round, select_collection


def main(argv=None):
    dispatch = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    dispatch.add_argument("--vibe")
    dispatch.add_argument("--out", type=Path)
    dispatch.add_argument("--score-workflow", type=Path)
    routing, _ = dispatch.parse_known_args(argv)
    profile = profile_for(routing.vibe) if routing.vibe else None
    old_collection = routing.out and (routing.out / "collection.json").is_file()
    if profile is None or old_collection or routing.score_workflow:
        return free_text_main(argv)

    parser = argparse.ArgumentParser(description=__doc__, parents=[dispatch])
    for key in ("model-dir", "prompt-workflow"):
        parser.add_argument("--" + key, type=Path, required=key == "model-dir")
    parser.add_argument("--prompt-model", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-provider", choices=("nanogpt", "typesafe"), default="nanogpt")
    for name, default in (("rounds", 1), ("pool-size", 8), ("replace", 4), ("best-of", 3),
                          ("seed", 7), ("resolution", 512)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-score", type=float, default=.75)
    parser.add_argument("--prepare-targets", action="store_true")
    parser.add_argument("--retry-pending", action="store_true")
    args = parser.parse_args(argv)
    if args.out is None or not 0 <= args.min_score <= 1:
        parser.error("--out is required and --min-score must be in [0, 1]")

    from particle_sliders_vibe import read_json, write_json
    from particle_sliders_vibe.nanoodle import NanoodleWorkflows

    prompts, score = workflow_graphs(args.prompt_model, args.judge_model, profile.concept)
    if args.prompt_workflow:
        prompts = read_json(args.prompt_workflow)
    workflows = NanoodleWorkflows(prompts, score, judge_provider=args.judge_provider)
    product = AnimaVibe(args.model_dir, workflows, device=args.device, resolution=args.resolution)
    try:
        state = collect_portfolio(product, profile, args)
        rounds, evidence = select_collection(
            state, profile, args.out, product=product, writer_model=args.prompt_model, minimum=args.min_score,
            retry_pending=args.retry_pending,
        )
        result = best_teacher_round(rounds, args.min_score)
        summary = dict(portfolio=str(args.out.resolve() / "portfolio.json"), distinct_finish=result["distinct_finish"],
                       near_duplicates=result["pair_audit"]["near_duplicates"], selection="raw",
                       selected_round=result["index"] + 1, completed_rounds=len(state["rounds"]),
                       selection_evidence=str(evidence), winning_policy=result["selection_policy"]["won_by"])
        try:
            manifest = portfolio_manifest(state | {"rounds": [result]}, args.min_score)
        except ValueError as error:
            if not str(error).startswith("no qualifying contexts"):
                raise
            print(json.dumps(summary | dict(qualified_contexts=0)))
            raise SystemExit(str(error)) from None
        destination = args.out / "training" / manifest["sha256"]
        write_json(destination / "manifest.json", manifest)
        if args.prepare_targets:
            from .cache import prepare_targets
            prepare_targets(product.get_runtime(), manifest, manifest["variation"], destination / "targets",
                            resolution=args.resolution)
        print(json.dumps(summary | dict(manifest=str(destination / "manifest.json"),
                                        targets=str(destination / "targets") if args.prepare_targets else None,
                                        qualified_contexts=len(manifest["rows"]))))
    finally:
        product.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
