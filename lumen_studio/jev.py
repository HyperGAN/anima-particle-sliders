"""The Anima Jev rubric for visible concept edits."""


def concept_rubric(concept):
    if not isinstance(concept, str) or not concept.strip():
        raise ValueError("A nonempty concept is required for Jev scoring")
    concept = concept.strip()
    question = f"How well does the candidate convey ‘{concept}’ without breaking the image?"
    levels = (
        f"No convincing {concept} effect, or the image is visibly broken.",
        f"A weak suggestion of {concept} in a coherent image.",
        f"Clearly conveys {concept} in a coherent, intact image.",
        f"Unmistakably conveys {concept} in a coherent, intact image.",
    )
    return question, levels
