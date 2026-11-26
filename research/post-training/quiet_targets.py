"""Rebuild the parent's late-cues targets with less of the teacher's change to the surroundings."""

import sys

from lumen_studio.cache import TargetCache
from lumen_studio.study.boss_targets import quiet_boss_background

from campaign import PARENT_STUDY, POLICY, ROOT, open_runtime

cache = TargetCache(PARENT_STUDY / POLICY["parent"]["cache"], verify=True)
assert cache.index["fingerprint"] == POLICY["parent"]["cache_fingerprint"]
rt = open_runtime()
try:
    for name in sys.argv[1:]:
        spec = POLICY["targets"][name]
        built = quiet_boss_background(rt, cache, ROOT / "targets" / name, spec["background"], spec.get("dose", 1.0))
        print(name, spec, built.index["fingerprint"], flush=True)
finally:
    rt.close()
