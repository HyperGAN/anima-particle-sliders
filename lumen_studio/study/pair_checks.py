"""Concept-specific questions used to admit frozen training pairs."""

PAIR_PROTOCOLS = {
    "final boss": "separate-boss-v1",
    "1990s anime": "paired-cel-effect-v1",
    "visually appealing character": "paired-character-appeal-v1",
}

EFFECT_QUESTIONS = {
    "1990s anime": (
        "Compare the two images including foreground and background. Does image 2 show a "
        "clearly stronger 1990s hand-drawn cel-animation treatment of the image itself than "
        "image 1? Look for opaque cel-painted shapes, controlled shadows, ink contours and "
        "animation backgrounds. Extra detail or animation shown only on an object inside the "
        "scene is insufficient."
    ),
    "visually appealing character": (
        "Compare the central character in the two images. Does image 2 visibly strengthen the "
        "character design itself through costume, silhouette, pose, expression or coordinated "
        "color? New outfits, framing and stylization are valid. Extra polish, scenery, or an "
        "unrelated second character alone is insufficient."
    ),
}


def pair_status(protocol, probability):
    if type(probability) not in (float, int) or not 0 <= probability <= 1:
        return "unmeasured"
    if protocol == "separate-boss-v1":
        if probability < 0.5:
            return "pass"
        return "fail" if probability >= 0.8 else "unmeasured"
    if probability >= 0.8:
        return "pass"
    return "fail" if probability < 0.5 else "unmeasured"
