# A character vibe from its phrase alone

`regime.py` trains and confirms a slider for a character vibe given only the phrase. Nobody writes prompts for the vibe, and no language model proposes scenes. Every choice is fixed in advance or made by a measured score.

```sh
python=/ml2/particlesliders-working/anima-pg012-adult-sliders-20261125/env/bin/python
cd research/post-training
ANIMA_POST_TRAINING_ROOT=<empty run directory> ANIMA_POST_TRAINING_PRODUCT=<this checkout> \
  ANIMA_POST_TRAINING_LEASE=training PYTHONPATH=. $python guard.py 0 ../vibe-regime/regime.py pirate captain
```

Rerun the same command to resume. Rendered images and judge answers are kept and reused.

## What it does

1. **Teaches on a fixed scene grid.** `library.py` pairs 44 people with 44 places and gives each one of five framings in turn, so three scenes in five are portraits. Twenty-four scenes teach, sixteen are held back to confirm, and four are watched for structural failure while training.
2. **Writes each treatment by replacing the clothing.** The treatment is the scene's own sentence with the clothing clause swapped for one of three fixed wardrobe clauses that take the phrase.
3. **Keeps the treatment worth teaching.** Each candidate is read the way the board reads a slider's output: the vibe is there, the drawing is one whole finished figure, and the scene kept its colors. The highest-scoring one is kept if it scores 0.35 or more and shows one transformed person.
4. **Builds late-cues targets and trains from scratch** with stock EMPF for 512 updates at G 0.00075, D 0.009 and particle 0.03, with no EMA.
5. **Chooses the checkpoint and one constant alpha on four characters.** Only checkpoints that passed their own structural check are candidates.
6. **Ranks on eight other characters and confirms on the sixteen held-back scenes.** Neither panel chose anything.

The judge questions are the working Boss score's, with the phrase in place of the Final Boss wording (`questions.py`).

## First run: pirate captain

[Result with every reading](../../docs/vibe-regime/pirate-captain.json)

| | |
| --- | --- |
| Teachers kept | 24 of 24, mean worth 76.8 |
| Training | 512 updates, no structural failure |
| Chosen | update 512 at constant alpha 1.875, search score 44.83 |
| Eight ranking characters | 39.42, 8 of 8 working, structural check 8 pass |
| Sixteen held-back scenes | 38.40, 14 of 16 working, structural check 13 pass and 3 unresolved |

The first attempt kept 2 of 24 scenes. Its treatments appended the vibe to the person's ordinary clothes and said the scene was unchanged. The frozen model then drew a second figure in costume beside the person, or a before-and-after sheet. Replacing the clothing in the same sentence fixed it with no other change.

## What the run shows is still open

- **The slider needs a high strength.** At strength 1 it reaches about a third of the teachers' effect, and the search chose 1.875. At that strength a waist-up portrait becomes a three-quarter or full-length shot.
- **Every character gets nearly the same coat and hat.** The repeat reading is 70 to 76, against 25 to 48 for Final Boss. For a single named costume that sameness is the vibe, so the repeat discount understates this slider. A vibe should say whether it wants variety.
- **One vibe is one test.** The wardrobe clauses and the scene grid are not specific to pirates, but no second phrase has been run.
