"""The paired OFF/ON questions shared by the working-edit scripts."""

STATE = (
    "Image 1 is the original illustration of one person. Image 2 is an edit meant to give that "
    "same person a final-boss design while keeping the picture."
)
QUESTIONS = {
    "same_person": {
        "type": "noul",
        "instructions": (
            "Is the central character of image 2 recognizably the same individual as in image 1, with "
            "their face and hair still visible, now wearing a different design? Answer near zero when the "
            "face is hidden, missing or replaced, when the person has become a faceless cloak, creature, "
            "statue, silhouette or abstract shape, or when the head can no longer be found."
        ),
    },
    "one_figure": {
        "type": "noul",
        "instructions": (
            "Does image 2 show exactly one central character, as image 1 does? Answer near zero when "
            "there are two or more figures, a duplicated or split figure, or no clear figure at all."
        ),
    },
    "same_scene": {
        "type": "noul",
        "instructions": (
            "Is image 2 still set in the same place as image 1, with the same surroundings visible "
            "behind the character? A change of costume, pose or camera distance is fine. Answer near zero "
            "when the surroundings were replaced, washed in one color or lost."
        ),
    },
}
