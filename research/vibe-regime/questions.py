"""The board's questions with the vibe phrase in place of the Final Boss wording."""

from lumen_studio.study import working_score as ws

from library import CHARACTER_VERSION, PHRASE_VERSION, WARDROBE_VERSION, WRITER_VERSION, article


def wardrobe_questions(vibe):
    a = article(vibe)
    effect = dict(
        state=("Image_1 is the original illustration. Image_2 is the edited illustration. Judge whether the "
               f"edited person has become an unmistakable {vibe}, from visible design of the character alone."),
        questions={"effect": {"type": "noul", "instructions": (
            f"Score {vibe} distinction from 0 to 1. 0 means no discernible {vibe} character. 0.25 means ordinary "
            f"clothing with no {vibe} cue. 0.50 means a credible hint of {a}, with some {vibe} cues. 0.75 means a "
            f"distinctive {vibe} with a recognizable, purposeful silhouette and signature costume details. 1.0 means "
            f"a visually unmistakable {vibe}. Judge the outfit, silhouette, headwear, accessories, and other visible "
            "evidence. Ignore scene, lighting, crop, and image polish.")}})
    # Read on one image during training, where only the gain over OFF and the structural check matter.
    rubric = {"type": "score", "instructions": (
        f"How strongly does the main character visibly read as {a} from their own costume, accessories, "
        "expression or silhouette? Ignore the setting, lighting, theatrical effects and image prettiness."),
        "criteria": [f"The person looks ordinary or generic, with no convincing {vibe} cue.",
                     f"The person has a weak or ambiguous visual cue of {a}.",
                     f"Several visible character-design cues make the person clearly read as {a}.",
                     f"The person's own iconic design makes them unmistakable as {a}."]}
    pair_state = ("Image 1 is the original illustration of one person. Image 2 is an edit meant to give that "
                  f"same person {a} design while keeping the picture.")
    return effect, rubric, pair_state


def literal_questions(vibe):
    target = f'“{vibe}”'
    evidence = (f'Look at the main character for visible evidence of {target}. For a physical trait, look for '
                'the trait itself, including its location and number when specified. A print or prop depicting a '
                'physical trait earns no credit for the trait itself. For a role or style, '
                'look for its defining features. Ignore unrelated details, lighting and image polish.')
    effect = dict(
        state=f'Image 1 is the original. Rate how clearly the main character in image 2 shows {target}.',
        questions={"effect": {"type": "noul", "instructions": (
            f'{evidence} Score 0 for absent, 0.25 for ambiguous, 0.50 for recognizable but weak, '
            '0.75 for clearly present and 1 for unmistakable.')}})
    rubric = {"type": "score", "instructions": evidence, "criteria": [
        f'No visible evidence of {target}.', f'Weak or ambiguous evidence of {target}.',
        f'{target} is clearly visible.', f'{target} is unmistakable on the character.']}
    pair_state = f'Image 1 is the original. Image 2 should show {target} on the same person in the same scene.'
    return effect, rubric, pair_state


def scene_questions(vibe):
    target = f'“{vibe}”'
    evidence = (f'Look across the whole image for {target}. Follow the requested placement, quantity and '
                'extent. For a feature requested everywhere, look for repeated visible instances throughout '
                'the picture. A picture of a physical feature printed on clothing earns no credit for the '
                'feature itself, unless the request calls for a print. For a role or style, judge its visible features.')
    effect = dict(state=f'Image 1 is the original. How clearly does image 2 show {target}?',
                  questions={"effect": {"type": "noul", "instructions": (
                      f'{evidence} Score 0 for absent, 0.25 for ambiguous, 0.50 for weak or partial, '
                      '0.75 for clear and 1 for unmistakable at the requested extent.')}})
    rubric = {"type": "score", "instructions": evidence, "criteria": [
        f'{target} is absent.', f'There is weak or partial evidence of {target}.',
        f'{target} is clearly visible.', f'{target} is unmistakable at the requested extent.']}
    return effect, rubric, f'Image 1 is the original. Image 2 should show {target} in the same scene.'


def build(vibe, *, version=WRITER_VERSION):
    builders = {WARDROBE_VERSION: wardrobe_questions, CHARACTER_VERSION: literal_questions,
                PHRASE_VERSION: scene_questions, WRITER_VERSION: scene_questions}
    effect, rubric, pair_state = builders[version](vibe)
    repeat = dict(
        state="Compare the central character designs in these two illustrations from the same visual style.",
        questions={"repeat": {"type": "score", "instructions": (
            f"How much is the same underlying {vibe} design repeated across the characters? Consider the silhouette, "
            "body construction, headwear, costume shapes and signature visual idea together. Different people can be "
            f"put in essentially the same {vibe} design. Distinct designs can share a kind of garment: look for "
            "individual design ideas. Differences in setting, lighting, hue, camera or pose alone say little about "
            "the design. Ignore beauty and rendering damage; those are judged separately."),
            "criteria": ["Clearly different character designs with different signature forms",
                         "Mostly different designs with a few shared motifs",
                         "Related designs with substantial overlap and individual differences",
                         f"Mostly the same {vibe} template with minor design differences",
                         "Effectively the same character design or a repeat of the same illustration"]}})
    return dict(effect=effect, repeat=repeat, pair=dict(state=pair_state, questions=ws.PAIR_QUESTIONS),
                figure=dict(state=ws.FIGURE_STATE, questions=ws.FIGURE_QUESTIONS),
                recolor=dict(state=ws.RECOLOR_STATE, questions=ws.RECOLOR_QUESTIONS),
                rubric=rubric)
