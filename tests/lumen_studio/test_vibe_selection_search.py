from itertools import combinations

from lumen_studio.vibe_selection_search import improve, optimize


def row(case, method, gain, family):
    return dict(case=case, method=method, finish_gain=gain, observed_family=family, integrity="pass")


def judge(collisions):
    def audit(rows):
        pairs = [
            {"left": a["case"], "right": b["case"],
             "mean": .85 if frozenset((a["method"], b["method"])) in collisions else .10}
            for a, b in combinations(rows, 2)
        ]
        return dict(pairs=pairs, near_duplicates=sum(pair["mean"] >= .65 for pair in pairs))
    return audit


def test_search_branches_on_both_sides_and_certifies():
    a = row("a", "a1", .9, "crown")
    b, alt = row("b", "b1", .8, "crown"), row("b", "b2", .65, "reef")
    c = row("c", "c1", .7, "stone")
    result = optimize([[a], [b, alt], [c]], judge({frozenset(("a1", "b1"))}), size=4)
    assert result["selected"] == [a, alt, c]
    assert result["audit"]["near_duplicates"] == 0
    assert result["certified"]
    assert result["upper_bound"] == result["distinct_finish"]["gain"] == .5625


def test_bounded_search_preserves_a_feasible_result_and_bound():
    a, b = row("a", "a1", .9, "crown"), row("b", "b1", .8, "reef")
    result = optimize([[a], [b]], judge({frozenset(("a1", "b1"))}), size=2, trials=1)
    assert result["selected"] == [a]
    assert result["upper_bound"] == result["distinct_finish"]["gain"] == .45
    assert result["certified"]


def test_joint_swap_preserves_family_cap_and_crosses_an_audited_conflict():
    a, aa = row("a", "a1", .5, "reef"), row("a", "a2", .8, "crown")
    b, bb = row("b", "b1", .47, "crown"), row("b", "b2", .54, "reef")
    c = row("c", "c1", .69, "crown")
    audit = judge({frozenset(("a1", "b2"))})
    result = improve([[a, aa], [b, bb], [c]], [a, b, c], audit, size=3)
    assert set(row["method"] for row in result["selected"]) == {"a2", "b2", "c1"}
    assert result["distinct_finish"]["gain"] == round((.8+.54+.69)/3, 12)
    assert result["audit"]["near_duplicates"] == 0



def test_collection_selects_strongest_exportable_round():
    from lumen_studio.vibe_cli import best_teacher_round

    def observed(index, gain, on, off, credited=True):
        return dict(index=index, distinct_finish={"gain": gain, "credited": ["a"] if credited else []},
                    selected=[dict(case="a", integrity="pass", finish_gain=gain, positive={"judgment": {"score": on}},
                                   neutral={"judgment": {"score": off}})])

    rounds = [
        observed(0, .65, .70, .10),
        observed(1, .45, .80, .20),
        observed(2, .20, .84, .30),
        observed(3, .15, .90, .10, credited=False),
    ]
    assert best_teacher_round(rounds, .75) is rounds[1]
    assert best_teacher_round(rounds, .95) is rounds[0]


def test_guarded_search_rejects_a_second_read_regression_and_keeps_a_real_gain():
    from lumen_studio.vibe_selection_search import guarded_search

    a, proposed_a = row("a", "a1", .5, "reef"), row("a", "a2", .8, "stone")
    b, proposed_b = row("b", "b1", .47, "crown"), row("b", "b2", .54, "crown")
    independent = {("a", "a1"): .55, ("a", "a2"): .41, ("b", "b1"): .4, ("b", "b2"): .43}
    inspect = judge(set())
    incumbent = [a, b]
    result = guarded_search([[a, proposed_a], [b, proposed_b]], incumbent, inspect(incumbent), inspect,
                            lambda rows: {(r["case"], r["method"]): independent[(r["case"], r["method"])]
                                          for r in rows}, size=2)
    assert result["selected"] == [a, proposed_b]
    assert result["distinct_finish"]["gain"] == .52
    assert result["independent"][result["won_by"]] >= result["independent"]["incumbent"]
