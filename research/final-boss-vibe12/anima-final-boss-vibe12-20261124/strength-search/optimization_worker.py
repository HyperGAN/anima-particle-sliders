"""Run the same installed alpha-tune command with a new RAW192 definition."""
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
archived=root.parents[1]/'anima-final-boss-research-raw192-20261123'
sys.path.insert(0,str(archived))
from run_alpha_tune import main

stage, attempt, method=sys.argv[1:]
sys.argv=[sys.argv[0],stage,attempt,str(root/f'{method}.json')]
raise SystemExit(main())
