"""Split the stronger Anima teacher prompt into its recorded role and costume edits."""


ROLE_BEFORE_COSTUME = "The character has the unmistakable silhouette of the final boss of an anime role-playing game."
ROLE_AFTER_COSTUME = "The character has the commanding, deliberate presence of the decisive villain."
ARMOR_CREST_START = "A tall open-faced helmet carries "
BACKGROUND_START = " Background: "


def split_teacher_cues(generic, compound):
    role_sentences = [sentence + " " for sentence in (ROLE_BEFORE_COSTUME, ROLE_AFTER_COSTUME)]
    if any(compound.count(sentence) != 1 or sentence in generic for sentence in role_sentences):
        raise ValueError("The stronger prompt must contain the two recorded role sentences exactly once")
    crest_only = compound.replace(role_sentences[0], "", 1).replace(role_sentences[1], "", 1)
    original_subject, separator, background = generic.partition(BACKGROUND_START)
    crest_subject, crest_separator, crest_background = crest_only.partition(BACKGROUND_START)
    if (not separator or not crest_separator or crest_background != background
            or not crest_subject.startswith(original_subject + " " + ARMOR_CREST_START)):
        raise ValueError("The crest cue must leave the original subject and background descriptions in place")
    crest = crest_subject[len(original_subject) + 1:]
    if crest.count(".") != 1 or not crest.endswith(".") or compound.count(crest + " ") != 1:
        raise ValueError("The stronger prompt must contain one recorded crest sentence")
    return {"role-cue35": compound.replace(crest + " ", "", 1), "crest-cue35": crest_only}
