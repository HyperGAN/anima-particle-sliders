# A character vibe from its phrase alone

`regime.py` trains a slider for a character vibe given only the phrase. Nobody writes prompts for the vibe, and no language model proposes scenes. Every choice is fixed in advance or made by a measured score.

```sh
python=/ml2/particlesliders-working/anima-pg012-adult-sliders-20261125/env/bin/python
cd research/post-training
ANIMA_POST_TRAINING_ROOT=<empty run directory> ANIMA_POST_TRAINING_PRODUCT=<this checkout> \
  ANIMA_POST_TRAINING_LEASE=training PYTHONPATH=. $python guard.py 0 ../vibe-regime/regime.py pirate captain
```

The run leaves `slider/<vibe>.safetensors` with its two sidecars, `slider/use.json` with the constant alpha to use, and `slider/samples.jpg` showing twelve characters OFF and ON. Rerun the same command to resume. Rendered images and judge answers are kept and reused.

## What it does by default

1. **Teaches on a fixed scene grid.** `library.py` pairs 44 people with 44 places and gives each one of five framings in turn, so three scenes in five are portraits. Twenty-four scenes teach and sixteen are held back.
2. **Writes each treatment by replacing the clothing.** The treatment is the scene's own sentence with the clothing clause swapped for one of three fixed wardrobe clauses that take the phrase.
3. **Keeps the treatment worth teaching.** Each candidate is read the way the board reads a slider's output: the vibe is there, the drawing is one whole finished figure, and the scene kept its colors. The highest-scoring one is kept if it scores 0.35 or more and shows one transformed person.
4. **Builds late-cues targets and trains from scratch** with stock EMPF for 448 updates at G 0.00075, D 0.009 and particle 0.03, with no EMA. Nothing is rendered or judged while it trains.
5. **Searches one constant alpha** for the finished slider on four characters, then saves the slider and its samples.

The judge questions are the working Boss score's, with the phrase in place of the Final Boss wording (`questions.py`).

| Setting | Effect |
| --- | --- |
| `ANIMA_REGIME_UPDATES=640` | Train longer, or continue a finished run from its last update. The slider then works at a lower alpha. |
| `ANIMA_REGIME_MEASURE=1` | Also rank on eight other characters, confirm on the sixteen held-back scenes and run the structural check on both. Neither panel chooses anything. |
| `ANIMA_REGIME_CURVE=128,256` | Measure saved checkpoints of a finished run one at a time. |

## Why 448 updates

One pirate captain run was saved ten times and each checkpoint measured with its own alpha search and the eight ranking characters. [Every reading](../../docs/vibe-regime/pirate-captain-by-updates.json)

| Updates | Alpha it needs | Ranking score | Working | Score at alpha 1.0 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1.69 | 26.4 | 3 of 8 | 4.7 |
| 256 | 2.0 | 38.0 | 8 of 8 | 0.0 |
| 320 | 1.5 | 30.9 | 6 of 8 | 0.6 |
| 384 | 1.875 | 39.6 | 6 of 8 | 2.9 |
| 448 | 1.94 | 43.8 | 8 of 8 | 21.6 |
| 512 | 1.875 | 39.4 | 8 of 8 | 18.7 |
| 576 | 1.5 | 36.9 | 7 of 8 | 30.5 |
| 640 | 1.125 | 43.2 | 7 of 8 | 33.3 |
| 704 | 1.625 | 41.8 | 6 of 8 | 27.1 |
| 768 | 1.375 | 45.1 | 8 of 8 | 30.0 |

448 is the earliest checkpoint that is good by every reading: all eight characters work, it scores as well as any later one at its own alpha, and it is the first where the hat appears at alpha 1.0. It needs an alpha near 1.9, and at that alpha a waist-up portrait becomes a three-quarter or full-length shot. From 576 on the slider works near alpha 1.1 to 1.5 and keeps the framing. On the card shared during these runs an update took 11 to 15 seconds, so 448 updates is 80 to 115 minutes of training and the fixed work around it is about 14 minutes.

## First run: pirate captain, measured in full at 512 updates

[Result with every reading](../../docs/vibe-regime/pirate-captain.json)

| | |
| --- | --- |
| Teachers kept | 24 of 24, mean worth 76.8 |
| Training | 512 updates, no structural failure |
| Chosen | update 512 at constant alpha 1.875, search score 44.83 |
| Eight ranking characters | 39.42, 8 of 8 working, structural check 8 pass |
| Sixteen held-back scenes | 38.40, 14 of 16 working, structural check 13 pass and 3 unresolved |

The first attempt kept 2 of 24 scenes. Its treatments appended the vibe to the person's ordinary clothes and said the scene was unchanged. The frozen model then drew a second figure in costume beside the person, or a before-and-after sheet. Replacing the clothing in the same sentence fixed it with no other change.

## Training longer lowers the working strength

The same run continued to 768 updates chose update 640 at constant alpha 1.125. [Result](../../docs/vibe-regime/pirate-captain-768.json) · [Each saved checkpoint measured](../../docs/vibe-regime/pirate-captain-by-updates.json)

| | 512 updates | Continued to 768 |
| --- | ---: | ---: |
| Chosen | update 512, alpha 1.875 | update 640, alpha 1.125 |
| Eight ranking characters | 39.42, 8 working | 43.23, 7 working |
| Sixteen held-back scenes | 38.40, 14 working | 36.94, 14 working |
| Same scene, ranking and held-back | 36 and 27 | 48 and 45 |
| Score at alpha 1.0 on the search characters | 18.7 | 33.3 |

The scores are level. What changed is the strength: at 1.125 a waist-up portrait stays a waist-up portrait, where at 1.875 it became a three-quarter or full-length shot. Measured one checkpoint at a time, 256 updates is the first that scores well (38.0 on the ranking characters, at alpha 2.0), and 128 is not enough (26.4).

## What the run shows is still open

- **How many updates a vibe needs for a strength near 1.** For this vibe 640 of 24 scenes was enough. One run does not fix the number.
- **Every character gets nearly the same coat and hat.** The repeat reading is 70 to 76, against 25 to 48 for Final Boss. For a single named costume that sameness is the vibe, so the repeat discount understates this slider. A vibe should say whether it wants variety.
- **One vibe is one test.** The wardrobe clauses and the scene grid are not specific to pirates, but no second phrase has been run.
