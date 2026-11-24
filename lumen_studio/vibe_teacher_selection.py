"""Select verified teacher portfolios from saved named-vibe renders."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from statistics import fmean
from threading import Lock

from . import vibe_curation
from .contracts import atomic_json, digest, file_hash
from .vibe_observer import PAIR_STATE, decision_values
from .vibe_portfolio import DUPLICATE_THRESHOLD, distinct_finish, finish_score, pair_audit
from .vibe_profiles import yes_no
from .vibe_rewrite import RewriteUnavailable, augment_round
from .vibe_selection_search import guarded_search
from .vibe_target_search import read
from .vibe_teacher_checks import SINGLE_STATE, single_question


def qualifies(row, minimum):
    on = row["positive"]["judgment"]["score"]
    off = row["neutral"]["judgment"]["score"]
    return row["integrity"] == "pass" and row["finish_gain"] > 0 and on >= minimum and on > off


def qualified_count(round_, minimum):
    credited = set(round_["distinct_finish"]["credited"])
    return sum(row["case"] in credited and qualifies(row, minimum) for row in round_["selected"])


def best_teacher_round(rounds, min_score):
    eligible = [round_ for round_ in rounds if qualified_count(round_, min_score)]
    return max(reversed(eligible or rounds), key=lambda round_: round_["distinct_finish"]["gain"])


def _temporary_judge_error(error):
    if getattr(error, "status", None) == 429:
        return True
    message = str(error)
    return message.startswith(("Jev HTTP 502", "Jev HTTP 503", "Jev HTTP 504", "unresolved call "))


def select_collection(state, profile, directory, *, product=None, writer_model=None, minimum=.75, retry_pending=False):
    from particle_sliders_vibe import _call, collection_lock

    source = Path(directory).resolve()
    files = ("vibe_teacher_selection.py", "vibe_selection_search.py", "vibe_teacher_checks.py", "vibe_rewrite.py")
    binding = dict(input=digest(state["identity"]), minimum=minimum, writer=writer_model,
                   engine={name: file_hash(Path(__file__).with_name(name)) for name in files})
    root = source / "teacher-selection" / digest(binding)
    locks = {}
    judge, provider = state["identity"]["judge"], state["identity"]["provider"]
    near = {"near": yes_no(profile.lookalike)}
    single = single_question(profile)
    call = partial(_call, root, retry_pending=retry_pending)

    def reading(outputs, state, questions):
        inputs = dict(provider=provider, model=judge, state=state, questions=questions,
                      images=[output["sha256"] for output in outputs])
        key = digest(dict(kind="judge", inputs=inputs))
        with locks.setdefault(key, Lock()):
            paths = [output["path"] for output in outputs]
            if any(file_hash(path) != output["sha256"] for path, output in zip(paths, outputs, strict=True)):
                raise ValueError("Teacher candidate image hash changed")
            archived = source / "calls" / (key + ".json")
            record = read(archived) if archived.exists() else None
            if record and record["status"] == "done" and record["inputs"] == inputs:
                receipt = record["result"]
            else:
                receipt = call("judge", inputs, lambda _: vibe_curation.observe(
                    provider, judge, paths, state, questions,
                ))
            values = decision_values(receipt["raw"], questions)
            if values != receipt["values"]:
                raise ValueError("Saved readings differ from the typed judge response")
        return values, key

    def compare(first, second):
        records = tuple(row["positive"]["output"] for row in (first, second))
        values, key = reading(records, PAIR_STATE, near)
        return dict(near=values["near"], decision=key)

    def recheck(row):
        values = []
        for side in ("neutral", "positive"):
            record = row[side]
            alternative, _ = reading((record["output"],), SINGLE_STATE, single)
            quality = record["readings"]
            values.append(finish_score(alternative["effect"], quality["overbake"], quality["layout"]))
        return (row["case"], row["method"]), values[1] - values[0]

    def cached(path, expected, produce):
        if path.exists():
            saved = read(path)
            if saved["source_sha256"] != expected or saved["sha256"] != digest(saved["result"]):
                raise ValueError("Saved teacher selection no longer matches its source")
            return saved["result"]
        result = produce()
        target = path.with_suffix(".temporary.json") if result.get("selection_policy", {}).get("temporary_judge_failure") else path
        atomic_json(target, dict(source_sha256=expected, result=result, sha256=digest(result)))
        return result

    def rank(round_, workers, *, teachers_only=False):
        groups = [
            [row | {"observed_family": row["observed_family"] or "unclassified"} for row in group
             if not teachers_only or qualifies(row, minimum)]
            for group in round_["alternatives"].values()
        ]
        # Teacher-only fallback admits the existing positive-gain qualified rows.
        original = [row for row in round_["selected"] if not teachers_only or qualifies(row, minimum)]
        retained = {row["case"] for row in original}
        pair_rows = [row for row in round_["pair_audit"]["pairs"]
                     if row["left"] in retained and row["right"] in retained]
        pairs = dict(pairs=pair_rows, average=fmean(row["mean"] for row in pair_rows) if pair_rows else None,
                     near_duplicates=sum(row["mean"] >= DUPLICATE_THRESHOLD for row in pair_rows))
        try:
            winner = guarded_search(groups, original, pairs,
                                    lambda rows: pair_audit(rows, compare, workers),
                                    lambda rows: dict(workers.map(recheck, rows)), size=len(groups))
        except RuntimeError as error:
            if not _temporary_judge_error(error):
                raise
            return dict(index=round_["index"], selected=original, pair_audit=pairs,
                        distinct_finish=distinct_finish(original, pairs, panel_size=len(groups)),
                        selection_policy={"won_by": "incumbent", "temporary_judge_failure": str(error)})
        return dict(
            index=round_["index"], selected=winner["selected"], distinct_finish=winner["distinct_finish"],
            pair_audit=winner["audit"],
            selection_policy=dict(won_by=winner["won_by"], alternatives=winner["alternatives"],
                                  independent=winner["independent"], audit_trials=winner["total_trials"],
                                  second_read_by_case=winner["second_read_by_case"]),
        )

    with collection_lock(root), ThreadPoolExecutor(6) as workers:
        rounds = [
            cached(root / f"round-{round_['index']+1}.json", digest(round_), lambda: rank(round_, workers))
            for round_ in state["rounds"]
        ]
        pool = list(state["rounds"])
        best = best_teacher_round(rounds, minimum)
        size = best["distinct_finish"]["panel_size"]
        # A single rewrite call expands the strongest measured round when it still has gaps.
        if product is not None and writer_model and qualified_count(best, minimum) < size:
            index = best["index"]
            initial = pool[index]
            try:
                expanded = cached(root / f"round-{index+1}-expanded-pool.json", digest([initial, best]),
                                  lambda: augment_round(initial, best, profile, product, root, call,
                                                        writer_model=writer_model, judge_model=judge, provider=provider))
            except (RuntimeError, RewriteUnavailable) as error:
                if not isinstance(error, RewriteUnavailable) and not _temporary_judge_error(error):
                    raise
                atomic_json(root / f"round-{index+1}-rewrite-unavailable.json", dict(error=str(error)))
            else:
                pool[index] = expanded
                result = cached(root / f"round-{index+1}-expanded.json", digest(expanded),
                                lambda: rank(expanded, workers))
                if best_teacher_round([best, result], minimum) is result:
                    rounds[index] = result
        if not any(qualified_count(round_, minimum) for round_ in rounds):
            for index, initial in enumerate(pool):
                if not any(qualifies(row, minimum) for group in initial["alternatives"].values() for row in group):
                    continue
                def qualify():
                    option = rank(initial, workers, teachers_only=True)
                    option["selection_policy"]["mode"] = "qualification-constrained"
                    return option

                result = cached(root / f"round-{index+1}-qualified.json", digest(initial), qualify)
                if qualified_count(result, minimum):
                    rounds[index] = result
    return rounds, root
