# Vibe12 Final Boss research source

On the image leaderboard, **Anima default** marks the released adapter and exact strength. The page’s “Show the Anima default” link jumps to its eight samples. The host reads `storage/released-default.json`, linked to the release checkout’s `docs/final-boss-raw192/default.json`, and checks the RAW hash, gains and reference image before displaying the label.

The [method inventory](ranked-methods.json) identifies the training implementation behind every ranked RAW192 adapter. The [checkpoint catalog](../../docs/final-boss-raw192/vibe12/checkpoints.json) gives the immutable download revision, file, size, SHA-256 and training-cache fingerprint.

### Reproduce the historical timed leaderboard

From the release checkout, point `CAMPAIGN` at the equipped Vibe12 experiment directory and run the following commands together. The artifacts publisher reuses existing files, verifies all ranked RAWs and uploads the winning target cache. Supply `--target-cache PATH` if that cache lives outside the per-recipe directory. The design publisher verifies the Jev evidence and exports the historical replay to `design-default.json` alongside the board. Fresh training comparisons use the [fixed-strength leaderboard](../../docs/leaderboard.md).

```sh
BOARD=docs/final-boss-raw192/vibe12
python -m release_tools.publish_final_boss_artifacts "$CAMPAIGN" "$BOARD"
python -m release_tools.publish_final_boss_vibe12 "$CAMPAIGN" "$BOARD"
python -m release_tools.publish_final_boss_designs "$CAMPAIGN" "$BOARD"
python -m release_tools.verify_final_boss_designs "$BOARD"
```


This directory preserves the host-side programs used to train the November 24 Vibe12 cohort and run its November 25 alpha search. The 26 training recipes share a teacher pool of 12 scenes, two native trajectories, and ten denoising positions per trajectory. Each recipe specifies which fields it draws from that pool. The comparison uses saved RAW192 adapters and eight matched slider-off / slider-on images. The public leaderboard and its offline evaluation are the easier entry points for inspecting scores without the training host. [Leaderboard](../../docs/leaderboard.md)

## Where to start

| Question | Source |
|---|---|
| Which recipe modified the training data? | [26 recipe names](anima-final-boss-vibe12-20261124/recipes.json), [sampling recipes](anima-final-boss-vibe12-20261124/shared_data.py), [spatial parameters](anima-final-boss-vibe12-20261124/register_spatial_v2.py), [seed, costume, and rollout registration](anima-final-boss-vibe12-20261124/register_extra_v2.py) |
| How does background projection work? | [On the twelve scenes](anima-final-boss-vibe12-20261124/adapters/spatial_stock.py) and [exterior projection and its checks](anima-final-boss-optimization-20261123/background_orthogonal_teacher_worker.py) |
| How do we capture fields on a trained adapter’s trajectory? | [Frozen-donor capture and native parity checks](anima-final-boss-vibe12-20261124/adapters/rollout_stock.py) |
| What actually trains and saves the weights? | [Recipe binding](anima-final-boss-vibe12-20261124/cohort.py), [the original ParticleGAN update, RAW export, FULL resume and milestone checks](anima-final-boss-optimization-20261123/optimization_worker.py), [engine source lock](anima-final-boss-research-raw192-20261123/cohorts/prompt-only/product/lumen_studio/engine-source.json) |
| How are alphas selected? | [Two-sided search](anima-final-boss-vibe12-20261124/strength-search/adaptive_alpha.py), [image measurements and final selection](anima-final-boss-vibe12-20261124/strength-search/retune_final_boss_visible.py), [queue](anima-final-boss-vibe12-20261124/strength-search/adaptive_queue.py) |
| What did Jev evaluate? | [Character effect rubric](anima-final-boss-optimization-20261123/boss_review.py), [current damage rubric and score](../../release_tools/final_boss_visible_damage.py), [previous paired-image damage rubric](anima-final-boss-optimization-20261123/image-breakage-rule.json), [native panel](anima-final-boss-research-raw192-20261123/panel.json) |
| What generates the page? | [Matched samples and ranking](anima-final-boss-vibe12-20261124/strength-search/alpha_leaderboard.py), [trial-path details](anima-final-boss-vibe12-20261124/strength-search/publish.py) |

The active strength worker measures the Boss rating of the edited image and its matched original, measures visible image damage, then ranks complete panels by their mean Final Boss score. The archived training worker separately records catastrophic output checks on training milestones. Earlier strength workers are included because their scored images were reused and appear in the history.

## Inputs for native reproduction

The paths are deliberately laid out as sibling dated campaign directories. Run the host commands from a **separate experiment copy** provisioned with the original artifacts. The GPU runner reads and writes run journals, caches, and checkpoints beneath that copy. A source checkout alone is sufficient to inspect the algorithms and run the public offline evaluation. Rendering and training also require the following material:

- The verified Anima Turbo v1.1 native model, matching CUDA/PyTorch environment, product runtime and runtime overlay. The recorded stock training engine lock names Particle Sliders source commit `507168a241e237fc3d5f1032a8137196fa3c3562`, ParticleGAN `0.11.1` at `d5dec2fe005305082acf382f0939a4bdb51c7131`, and the wheel SHA. The portable model SHA is `eb5520f293d42a19df9242d8d442d5d2068e5121620814418c2b50a97de0552f`. The target-construction checkout used Anima commit `5981336de4af939e430d0940659cfddb723a1023`. These are separate training and target-construction dependencies. [Engine lock](anima-final-boss-research-raw192-20261123/cohorts/prompt-only/product/lumen_studio/engine-source.json) · [Recorded paths](anima-final-boss-vibe12-20261124/control.py)
- The frozen teacher manifest `c2cb92f3c068a1bf20ae942f81ad73e4b8c18d60d769b05c049e78888e3b228d`, its selection audit and native shards with cache fingerprint `adab80480e0fa186a10234c1055df3ca2f148a2d71665d5d01543091d801a9f8`. Scene prompts, selection evidence, source image/matte inputs, the pinned teacher-construction worktrees, and the host's seed-split dependencies supply the recipe registrations. The campaign assumes the on-disk layout recorded in its inputs. [Input hashes](source-files.json) · [Teacher binding](anima-final-boss-vibe12-20261124/freeze.py)
- The common slider-off PNGs and their render records, historical rubric and calibration fixtures, RAW checkpoint and matching scoped sidecar per method, and the fixed-strength tuning reports. A terminal comparison requires every matched sample. Optional prior alpha receipts save work when their hashes, models, settings and sample bindings match. The development/gallery archives contain further assets needed by the legacy publishing paths.
- The Particle Sliders Jev observer and its three pinned module versions, authorized access to its decision endpoint, and one installed `particle_sliders.alpha_renderers` entry for `anima-final-boss-turbo`. The bridge build source is included. Use the guarded environment to install it in the renderer location expected by the worker. Host-mounted storage, its lock directory, and the model/worker interpreter paths must resolve before admission. [Jev module hashes](anima-final-boss-research-raw192-20261123/prompt-only-alpha-tune/pr-wheel-visible/module-hashes.json) · [Renderer registration](anima-final-boss-research-raw192-20261123/prompt-only-alpha-tune/bridge/pyproject.toml)

## Recorded-host entry points

With the dependencies above in place, start in `anima-final-boss-vibe12-20261124/`. On a freshly provisioned run, create the expected output folders and run `freeze.py`, `prepare_mattes.py`, `prepare_prompts.py`, and `register_spatial_v2.py` with the recorded research environment. The fixed source registration in `register_extra_v2.py --watch` admits the donor-dependent arm as its RAW becomes available. The training dispatcher handles only its own children and leases.

```sh
python -u campaign_shared_v4.py
```

For an existing cohort with the three calibration RAW/control records, `strength-search/define.py` creates the pinned definitions once in a new destination. Preserve the existing definitions on an exact resume. Start the search queue and publishers as separate processes on the equipped host:

```sh
python -u publish.py --watch
python -u strength-search/alpha_leaderboard_v1.py --watch
python -u metric-calibration/review.py
python -u metric-calibration/check_transfer.py
python -u metric-calibration/backfill.py
python -u strength-search/adaptive_queue.py
python -u strength-search/alpha_leaderboard.py --watch
python strength-search/publish.py
```

The publishers verify their output with an HTTP read-back. The public capture uses loopback port 8765. Serve `anima-final-boss-research-raw192-20261123/sample-gallery/` on that port when exercising this path. Native inputs with path-bearing hashes must be restored on their recorded layout for exact replay. A relocated experiment needs its own pinned manifests and journals.

[Source inventory](source-files.json) lists the original SHA-256 for each captured file, a second digest where the HTTP hostname was replaced by loopback, and digests of selected external inputs. Model weights, adapter weights, FULL states, image and tensor caches, generated judge receipts, and operator credentials must be supplied separately for a native run.

## Finished-image damage calibration

The current worker starts a separate versioned alpha search. It reuses matching native renders and Boss judgments and obtains a new Jev damage rating for each unique finished image. The earlier publisher supplies historical settings for the backfill. Start the watchers and queues in separate processes. The live chart labels the user-nominated opening-four reference separately from the sorted scores. [Jev questions and score](anima-final-boss-vibe12-20261124/metric-calibration/finished_image.py) · [Calibrated search worker](anima-final-boss-vibe12-20261124/strength-search/retune_final_boss_visible.py) · [Calibration evidence](../../docs/final-boss-raw192/vibe12/visible-damage-audit.json)

## Finished-edit finalist comparison

The all-recipe score chooses the six recipes and their measured settings. From the campaign directory, run `python metric-calibration/finalist_board.py --watch` with the recorded Jev environment to publish the blind round robin when a finalist or its strength changes. Each image order is cached by model, question and image hashes. The two calibration commands are `python metric-calibration/preference_round.py` and `python metric-calibration/preference_finish.py`. Reusing the historical calibration inputs on the research host requires its archived panels. The published report and bound responses support offline replay of every decision. [Results and replay commands](../../docs/final-boss-raw192/vibe12/finalist-review.md)

## Distinct Final Boss designs

Run `python metric-calibration/design_range.py --watch` in the campaign to measure recurring boss designs at each recipe’s best per-image strength. The alpha leaderboard consumes the latest complete snapshot. The companion scripts `design_range_transfer.py` and `design_lineups.py` compare the nominated opening-four recipe and the late-cue incumbent across additional search scenes and complete character lineups. The export uses `python -m release_tools.publish_final_boss_designs CAMPAIGN docs/final-boss-raw192/vibe12` after exporting the alpha snapshot. [Formula, samples and offline replay](../../docs/final-boss-raw192/vibe12/distinct-designs.md)

## Distinct-design alpha search

Run `python strength-search/adaptive_queue.py` in the campaign for guarded two-card tuning. The default objective is `design`. The old scalar queue uses `--objective visible`. Distinct-design decisions reuse cached native frames and Jev receipts. Missing frames are rendered under the same model, RAW checkpoint, scene, seed and resolved schedule. Results in `strength-search/leaderboard-retune-design/<recipe>/tuning.json` drive the default design leaderboard. [Search protocol and verified opening-four replay](../../docs/final-boss-raw192/vibe12/alpha-tuning/README.md)
