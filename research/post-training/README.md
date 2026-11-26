# Post-training a trained Final Boss slider

These scripts fork one trained RAW into several children, give each the same number of
additional stock ParticleGAN updates and rank every child against its untouched parent.
[Method and measured results](../../docs/post-training/README.md)

Run them from this directory on the equipped study host, with this checkout at
`<campaign>/product`. `policy.json` pins the parent RAW and FULL, its frozen target cache,
the rates, the added budget and the teacher rules. Outputs go to `<campaign>`.

```sh
python=/ml2/particlesliders-working/anima-pg012-adult-sliders-20261125/env/bin/python
PYTHONPATH=. $python guard.py 1 teachers.py            # measure teachers, build replacement targets
PYTHONPATH=. $python guard.py 1 fit.py control         # fork, same targets
PYTHONPATH=. $python guard.py 1 fit.py cleaner         # fork, one cleaner teacher
PYTHONPATH=. $python guard.py 1 fit.py cleanest        # fork, least-damaged teachers
ANIMA_POST_TRAINING_PRODUCT=<parent study>/product \
  PYTHONPATH=. $python guard.py 1 fit.py continued     # exact FULL resume, same targets
PYTHONPATH=. $python guard.py 0 working_search.py parent-0 control-48   # alpha by the working Boss score, then eight ranking characters
PYTHONPATH=. $python working_structure.py              # structural gate at each chosen alpha
PYTHONPATH=.:../.. $python publish_working.py          # gallery/working-leaderboard.html
```

`teachers.py` first prints the measured gain and damage of the twelve current teachers and
names the six with the most damage. Write three proposals for each into `alternatives.json`,
then run it again. It renders them from the same neutral prompt and seed, applies each rule in
`policy.json` and rebuilds the late-cues fields for the selected scenes. Every unchanged scene
is regenerated through the same code and must equal the parent cache bit for bit.

`fit.py` forks through `open_training(mode="fork", fork=<parent RAW>)`. The exact
continuation instead resumes the parent FULL, which binds the trainer source that wrote it,
so it runs from the parent study's own checkout. Each arm reports at every quarter of the
added budget and stops on a structural failure before another update.

`working_search.py` re-picks each RAW's alpha from the strengths `calibrate.py` rendered, refines
around the leader and ranks the held-out characters. Name a RAW as `<arm>-<added updates>`.

`calibrate.py` and `publish.py` produce the earlier board under the distinct Boss score.
`calibrate.py` also proves that a forked child's RAW0 draws the parent's ranking images and
OFF images byte for byte. `guard.py` holds the study's per-card leases and enforces the same
entry floors and owned caps as the duration study. An exit code of 75 means the lease is busy.
