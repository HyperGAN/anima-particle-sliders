"""Select verified teacher portfolios from saved named-vibe renders.

Each measured round moves through saved stages under ``teacher-selection``:

    RANKED --(gaps and a writer)--> EXPANDED_POOL --> EXPANDED
       |                                 |
       |                                 +--> REWRITE_UNAVAILABLE (asked again on the next run)
       +--(no round has a teacher)--> QUALIFIED

A stage is complete once its second effect read and lookalike audit were
measured. When the judge is temporarily unavailable the stage falls back to
the audited incumbent and is saved as incomplete: it is never reused, and it
never replaces a complete result.
"""
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
from pathlib import Path
from statistics import fmean

from . import vibe_curation
from .contracts import atomic_json, digest, file_hash
from .vibe import neutral_score_limit
from .vibe_calls import CallLedger, temporary
from .vibe_observer import PAIR_STATE, decision_values
from .vibe_portfolio import DUPLICATE_THRESHOLD, distinct_finish, finish_score, pair_audit
from .vibe_profiles import yes_no
from .vibe_rewrite import NothingToRewrite, RewriteUnavailable, augment_round
from .vibe_selection_search import guarded_search
from .vibe_target_search import read
from .vibe_teacher_checks import SINGLE_STATE, single_question


def qualifies(row, minimum, *, max_neutral_score=1.0):
    on = row["positive"]["judgment"]["score"]
    off = row["neutral"]["judgment"]["score"]
    return (row["integrity"] == "pass" and row["finish_gain"] > 0 and on >= minimum
            and on > off and off <= max_neutral_score)


def qualified_count(round_, minimum, *, max_neutral_score=1.0):
    credited = set(round_["distinct_finish"]["credited"])
    return sum(row["case"] in credited and qualifies(row, minimum, max_neutral_score=max_neutral_score)
               for row in round_["selected"])


def best_teacher_round(rounds, min_score, *, max_neutral_score=1.0):
    eligible = [round_ for round_ in rounds if qualified_count(round_, min_score, max_neutral_score=max_neutral_score)]
    return max(reversed(eligible or rounds), key=lambda round_: round_["distinct_finish"]["gain"])


class Stage(str, Enum):
    RANKED = "round-{n}.json"
    EXPANDED_POOL = "round-{n}-expanded-pool.json"
    EXPANDED = "round-{n}-expanded.json"
    REWRITE_UNAVAILABLE = "round-{n}-rewrite-unavailable.json"
    QUALIFIED = "round-{n}-qualified.json"

    def path(self, root, index):
        return root / self.value.format(n=index + 1)


def incomplete(result):
    """The reason a stage could not finish its independent checks, if any."""
    return result.get("selection_policy", {}).get("temporary_judge_failure")


def selection_status(deferred, unresolved):
    """Say whether selection finished its independent checks, and how to finish them.

    ``deferred`` pairs each check that could not be measured with its cause.
    ``unresolved`` lists provider calls whose outcome is unknown.
    """
    if not deferred:
        return dict(teacher_selection="complete")
    return dict(teacher_selection="incomplete", unmeasured=sorted({check for check, _ in deferred}),
                teacher_selection_issue=str(deferred[0][1]), unresolved_calls=unresolved,
                next_step=("Reconcile the recorded requests, then rerun with --retry-pending." if unresolved
                           else "Rerun the same command to repeat the calls that failed."))


def select_collection(state, profile, directory, *, product=None, writer_model=None, minimum=.75, retry_pending=False):
    from particle_sliders_vibe import collection_lock

    source = Path(directory).resolve()
    max_neutral_score = neutral_score_limit(state["identity"])
    qualifies_teacher = partial(qualifies, minimum=minimum, max_neutral_score=max_neutral_score)
    count_teachers = partial(qualified_count, minimum=minimum, max_neutral_score=max_neutral_score)
    choose_round = partial(best_teacher_round, min_score=minimum, max_neutral_score=max_neutral_score)
    files = ("vibe_teacher_selection.py", "vibe_selection_search.py", "vibe_teacher_checks.py", "vibe_rewrite.py")
    binding = dict(input=digest(state["identity"]), minimum=minimum, writer=writer_model,
                   engine={name: file_hash(Path(__file__).with_name(name)) for name in files})
    root = source / "teacher-selection" / digest(binding)
    judge, provider = state["identity"]["judge"], state["identity"]["provider"]
    near = {"near": yes_no(profile.lookalike)}
    single = single_question(profile)
    # Judgments already paid for during collection are reused from its ledger.
    call = CallLedger(root, retry_pending=retry_pending, archives=(source,))
    deferred = []

    def reading(outputs, state, questions):
        inputs = dict(provider=provider, model=judge, state=state, questions=questions,
                      images=[output["sha256"] for output in outputs])
        paths = [output["path"] for output in outputs]
        if any(file_hash(path) != output["sha256"] for path, output in zip(paths, outputs, strict=True)):
            raise ValueError("Teacher candidate image hash changed")
        receipt = call("judge", inputs, lambda _: vibe_curation.observe(provider, judge, paths, state, questions))
        values = decision_values(receipt["raw"], questions)
        if values != receipt["values"]:
            raise ValueError("Saved readings differ from the typed judge response")
        return values, call.key("judge", inputs)

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

    def cached(stage, index, expected, produce):
        path = stage.path(root, index)
        draft = path.with_suffix(".temporary.json")
        if path.exists():
            saved = read(path)
            if saved["source_sha256"] != expected or saved["sha256"] != digest(saved["result"]):
                raise ValueError("Saved teacher selection no longer matches its source")
            return saved["result"]
        result = produce()
        record = dict(source_sha256=expected, result=result, sha256=digest(result))
        if incomplete(result):
            atomic_json(draft, record)
        else:
            atomic_json(path, record)
            draft.unlink(missing_ok=True)
        return result

    def rank(round_, workers, *, teachers_only=False):
        def eligible(row):
            return (qualifies_teacher(row) if teachers_only
                    else row["neutral"]["judgment"]["score"] <= max_neutral_score)

        groups = [
            [row | {"observed_family": row["observed_family"] or "unclassified"} for row in group
             if eligible(row)]
            for group in round_["alternatives"].values()
        ]
        # Teacher-only fallback admits the existing positive-gain qualified rows.
        original = [row for row in round_["selected"] if eligible(row)]
        retained = {row["case"] for row in original}
        pair_rows = [row for row in round_["pair_audit"]["pairs"]
                     if row["left"] in retained and row["right"] in retained]
        pairs = dict(pairs=pair_rows, average=fmean(row["mean"] for row in pair_rows) if pair_rows else None,
                     near_duplicates=sum(row["mean"] >= DUPLICATE_THRESHOLD for row in pair_rows))
        try:
            winner = guarded_search(groups, original, pairs,
                                    lambda rows: pair_audit(rows, compare, workers),
                                    lambda rows: dict(workers.map(recheck, rows)), size=len(groups))
        except Exception as error:
            if not temporary(error):
                raise
            deferred.append(("second effect read", error))
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
            cached(Stage.RANKED, round_["index"], digest(round_), lambda: rank(round_, workers))
            for round_ in state["rounds"]
        ]
        pool = list(state["rounds"])
        best = choose_round(rounds)
        size = best["distinct_finish"]["panel_size"]
        # A single rewrite call expands the strongest measured round when it still has gaps.
        if product is not None and writer_model and count_teachers(best) < size:
            index = best["index"]
            initial = pool[index]
            unavailable = Stage.REWRITE_UNAVAILABLE.path(root, index)
            try:
                expanded = cached(Stage.EXPANDED_POOL, index, digest([initial, best]),
                                  lambda: augment_round(initial, best, profile, product, root, call,
                                                        writer_model=writer_model, judge_model=judge, provider=provider,
                                                        max_neutral_score=max_neutral_score))
            except NothingToRewrite as error:
                atomic_json(unavailable, dict(error=str(error)))
            except Exception as error:
                if not isinstance(error, RewriteUnavailable) and not temporary(error):
                    raise
                # The writer is asked again on the next run.
                deferred.append(("rewrite pass", error))
                atomic_json(unavailable, dict(error=str(error)))
            else:
                unavailable.unlink(missing_ok=True)
                pool[index] = expanded
                result = cached(Stage.EXPANDED, index, digest(expanded), lambda: rank(expanded, workers))
                if not incomplete(result) and choose_round([best, result]) is result:
                    rounds[index] = result
        if not any(count_teachers(round_) for round_ in rounds):
            for index, initial in enumerate(pool):
                if not any(qualifies_teacher(row) for group in initial["alternatives"].values() for row in group):
                    continue
                def qualify():
                    option = rank(initial, workers, teachers_only=True)
                    option["selection_policy"]["mode"] = "qualification-constrained"
                    return option

                result = cached(Stage.QUALIFIED, index, digest(initial), qualify)
                if count_teachers(result):
                    rounds[index] = result
    return rounds, root, selection_status(deferred, call.unresolved())
