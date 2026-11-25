"""Run the Anima retrain with the measured shared-card host envelope."""
import sys
from control import ROOT
import shared_capacity_guard_probe as captured

if __name__ == '__main__':
    stage,gpu,attempt,method=sys.argv[1:]
    captured.ROOT=ROOT
    captured.plan=lambda _:{}
    raise SystemExit(captured.run(stage,int(gpu),attempt,(method,)))
