"""Visual criteria and prompt treatments for named Anima vibes."""
from dataclasses import dataclass

from .vibe_portfolio import FAMILY_MINIMUM


@dataclass(frozen=True)
class VibeProfile:
    concept: str
    scene_instruction: str
    effect: dict
    suitability: str
    lookalike: str
    families: dict[str, str]
    subject_slots: tuple[tuple[str, str], ...] = ()
    family_choice: bool = False
    family_minimum: float = FAMILY_MINIMUM

def score_question(instructions, criteria):
    return dict(type="score", instructions=instructions, criteria=list(criteria))


def yes_no(instructions):
    return dict(type="noul", instructions=instructions)


ANIME = VibeProfile(
    "1990s anime",
    "Describe one continuous shot from one camera viewpoint per neutral, with ordinary physical subjects. "
    "Vary the locations and viewpoints across scenes. Keep neutrals free of style, anime, decade, cel and film terms.",
    score_question(
        "Imagine this image as a full-bleed frame from a broadcast or theatrical hand-drawn Japanese animated film "
        "of the 1990s. How convincing is that specific visual treatment? Look for opaque, uniform cel-painted flats, "
        "sharply bounded simple shadow shapes and consistent ink contours on foreground forms where visible, plus a "
        "separately painted background or a stylized background painting suitable as a cel-animation establishing shot. "
        "Consider the visible palette and optical softness only as supporting evidence. An ink-and-watercolor drawing "
        "on textured or bordered paper, pencil sketch, poster, or illustration with watercolor bleeding throughout "
        "does not itself establish the cel animation treatment. A person need not be present.",
        (
            "The medium and overall treatment read as a different kind of image, including a sketch, watercolor on paper, "
            "glossy 3D, or photography.",
            "A few illustrative features evoke animation, but the rendered frame lacks the specific hand-drawn cel treatment.",
            "The image mostly looks like an animated cel-era frame, with clear opaque graphic shapes, controlled shadows "
            "and/or an animation background.",
            "An unmistakable, coherent 1990s hand-drawn cel-animation frame, with a convincing production palette, "
            "graphic shapes and painted treatment throughout.",
        ),
    ),
    "How successfully does this frame read as a finished hand-drawn 1990s anime shot with a coherent, "
    "distinctive cel treatment? Judge the full frame, including scenes without figures. Discount paper-like "
    "textures, decorative borders, clipped highlights, crunchy detail and disconnected views.",
    "Do these use nearly the same specific animation treatment? Compare line weight and contour shapes, "
    "how cel shadows are cut, the density and brush treatment of the background, and the separation of "
    "foreground cels from the setting. The shared 1990s medium alone is weak evidence. Different subjects, "
    "camera angles, settings or colors alone can leave the treatment the same.",
    {
        "theatrical": "Fine restrained contour ink, opaque softly colored cels with crisp shadow shapes, and "
                      "richly layered, atmospheric gouache background painting with deep spatial perspective",
        "broadcast": "Medium-weight consistent outlines, clear two-tone cel shadows, and simplified matte "
                     "painted backgrounds with economical descriptive detail",
        "technical": "Thin technical contour lines, angular forms and hard geometric cel shadow boundaries, "
                     "with precisely drafted backgrounds and finely inked details",
        "romantic": "Delicate tapered colored lines, opaque pastel cels with curved shadow shapes, and "
                    "lightly painted soft-focus settings, in a clean full-bleed frame",
        "nocturne": "Deep pools of angular ink shadow, hard low-key cel lighting with small clean highlights, "
                    "and a restrained painted background with strong separation between dark layers",
        "comedy": "Bold bouncy rounded outlines, large uncomplicated opaque color shapes and sparse "
                  "hard-edged shadows, with chunky simplified background shapes",
    },
    subject_slots=(
        ("environment", "Only unoccupied environments. Keep the setting itself the focus and include no visible people."),
        ("object", "Only still lifes or close views of inanimate objects. Include no people, body parts or animals."),
        ("animal", "Only scenes centered on an animal in a plausible setting. Include no people or human figures."),
        ("person", "Only scenes with one or more people engaged in an ordinary activity, in a recognizable setting."),
    ),
    family_choice=True,
    family_minimum=.40,
)

FINAL_BOSS_NEUTRAL_MAX = .35

FINAL_BOSS = VibeProfile(
    "Final Boss",
    "Each neutral shows one ordinary adult in everyday clothes doing a recognizable activity. "
    "Describe the person's appearance, pose and surroundings in one continuous shot. "
    "The neutral character is a civilian with a plain silhouette, without armor, regalia, weapons, "
    "monstrous features or an imposing antagonist role. In the positive prompt, keep the same person, "
    "pose, framing, setting and lighting, and convey Final Boss through a distinctive costume and silhouette.",
    score_question(
        "How strongly does the main character visibly read as a formidable final video-game boss "
        "from their own costume, accessories, expression or silhouette? Ignore the setting, lighting, "
        "theatrical effects and image prettiness.",
        (
            "The person looks ordinary or generic, with no convincing final-boss cue.",
            "The person has a weak or ambiguous visual cue of a formidable antagonist.",
            "Several visible character-design cues make the person clearly read as a formidable final boss.",
            "The person's own iconic, imposing design makes them unmistakable as a formidable final boss.",
        ),
    ),
    "How successfully does this scene make the main figure look like a formidable final video-game boss "
    "with a clear original costume silhouette and a usable single-scene finish? Discount a distant, ordinary "
    "figure, duplicate poses, design-sheet layouts, clipped glare, oversharpened textures and chaotic shapes.",
    "Are these nearly the same distinctive boss costume or visual design? Compare silhouette, headpiece, "
    "major silhouette shapes and dominant garment or armor materials. Ignore the anime medium, the viewer-facing "
    "pose, background, and species or facial identity. Color changes alone can still be a near duplicate. "
    "Visibly different shapes and construction are distinct designs.",
    {
        "silk": "wide layered crimson silk sleeves with a draped bell silhouette and a formal folding fan",
        "foundry": "a heavy square industrial automaton silhouette with a round porthole face and rectangular brass plating",
        "oracle": "a tall pointed ivory hood and indigo fabric that forms a narrow triangular oracle silhouette",
        "reef": "prominent aquatic fan fins at the shoulders and smooth shell or fishlike materials",
        "root": "tree-root antlers and a broad dark bark and moss mantle that dominate the silhouette",
        "eclipse": "circular eclipse or crescent geometry worn around the shoulders of an amethyst ceremonial robe",
        "crown": "an upright iron crown with a black armored coat and wide structured shoulders",
    },
)


def profile_for(vibe):
    return {"1990s anime": ANIME, "final boss": FINAL_BOSS}.get(" ".join(vibe.casefold().split()))
