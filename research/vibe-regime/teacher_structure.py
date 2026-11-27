"""Fill missing structure verdicts for an incomplete teacher collection."""


def inspect_saved_rounds(record, check, pool):
    known, images = {}, {}
    for row in record:
        for option in row["options"]:
            for ref, field in ((row["neutral"], "baseline_structure"), (option["image"], "candidate_structure")):
                sha = ref["sha256"]
                images[sha] = ref
                if option.get(field) in ("pass", "fail", "unmeasured"):
                    known[sha] = option[field]
    missing = [sha for sha in images if sha not in known]
    known.update(zip(missing, pool.map(check, (images[sha] for sha in missing))))
    for row in record:
        for option in row["options"]:
            option["baseline_structure"] = known[row["neutral"]["sha256"]]
            option["candidate_structure"] = known[option["image"]["sha256"]]
    return known


def restore_selections(record, saved):
    kept = {}
    for row in record:
        previous = saved.get(row["scene"])
        if not previous or row["seed"] != previous["seed"]:
            continue
        for option in row["options"]:
            if option["positive"] != previous["positive"] or option["positive"] != row["chosen"]:
                continue
            if option["baseline_structure"] == option["candidate_structure"] == "pass":
                kept[row["scene"]] = previous
            else:
                row["previous_choice"], row["chosen"] = row["chosen"], None
    return kept
