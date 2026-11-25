from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root.parents[1]/'anima-final-boss-optimization-20261123'))
import calibration_guard
calibration_guard.ROOT=root
calibration_guard.plan=lambda _:{}
gpu, attempt, method, *mode=sys.argv[1:]
if mode == ['leaderboard-visible']:
    raise SystemExit(calibration_guard.run('retune-visible', int(gpu), attempt, (method,),
                                           worker_name='retune_final_boss_visible.py'))
if mode == ['leaderboard-damage']:
    raise SystemExit(calibration_guard.run('retune-damage', int(gpu), attempt, (method,),
                                           worker_name='retune_final_boss_damage.py'))
if mode == ['leaderboard']:
    raise SystemExit(calibration_guard.run('retune-adaptive', int(gpu), attempt, (method,),
                                           worker_name='retune_final_boss_adaptive.py'))
if mode == ['leaderboard-centered']:
    raise SystemExit(calibration_guard.run('retune-centered', int(gpu), attempt, (method,),
                                           worker_name='retune_final_boss_centered.py'))
if mode == ['leaderboard-ascending']:
    raise SystemExit(calibration_guard.run('retune', int(gpu), attempt, (method,),
                                           worker_name='retune_final_boss.py'))
if mode:
    raise ValueError('Unknown strength search mode')
raise SystemExit(calibration_guard.run('tune',int(gpu),attempt,(method,)))
