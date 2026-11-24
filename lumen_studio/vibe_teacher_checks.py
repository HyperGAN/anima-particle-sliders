"""Independent image-comparison prompts for teacher selection."""
from .vibe_profiles import score_question, yes_no

STATE = "Independently compare the two unlabeled finished images as teacher targets."
QUESTIONS = {
    "anime": (
        "Does image 1 read more convincingly than image 2 as a finished hand-drawn cel-animation frame "
        "from 1990s Japan? Give priority to coherent ink contours, flat opaque paints, deliberate shadow shapes, "
        "and integrated painted scenery across the frame. Discount sharpening halos and divided layouts. "
        "Disregard period props, cast, and which genre or subject is more appealing."
    ),
    "boss": (
        "Does image 1 more successfully depict a formidable final antagonist in a single clean anime scene "
        "than image 2? Give priority to the antagonist's readable silhouette, physical construction and visual "
        "presence. Discount duplicated views, halos, or unreadable shapes. Disregard preferential color and lighting."
    ),
}


def question(profile):
    return {"preferred": yes_no(QUESTIONS["anime" if profile.family_choice else "boss"])}


def preference(in_order, reversed_order):
    return (in_order + 1 - reversed_order) / 2


SINGLE_STATE = "Rate the visual effect of this one unlabeled frame. Use the image alone."
SINGLE_READINGS = {
    "anime": score_question(
        "Could this full frame have been photographed from an actual 1990s Japanese drawn-animation production? "
        "Use the junctions between painted background and flat cel surfaces, precise line shapes and stepped shading. "
        "Where the frame is all scenery, consider its use as a background-only establishing shot. Discount period props, "
        "scan texture alone, digital watercolor-paper borders and glossy continuous gradients.",
        ("Reads as another medium or era of rendering.",
         "Occasional cel production signs with limited consistency.",
         "Several clear cel-era production choices work across most of the frame.",
         "Looks convincingly like a complete still photographed from hand-drawn 1990s Japanese animation."),
    ),
    "boss": score_question(
        "Could the central figure serve as the last antagonist in a hand-drawn role-playing game? "
        "Judge from its physical design alone, including scale impression, the outline, dress or shell construction, "
        "face and distinctive, readable shapes. A dark set, dramatic light and explosions are weak evidence on their own.",
        ("Looks like an ordinary character or incidental detail.",
         "A few imposing details without a clear final-antagonist design.",
         "A readable, distinctive design suited to a final encounter.",
         "Unmistakable final-antagonist design with unusually strong visual identity."),
    ),
}


def single_question(profile):
    return {"effect": SINGLE_READINGS["anime" if profile.family_choice else "boss"]}
