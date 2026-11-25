"""Print quarter checkpoints and owned job transitions for the shared-teacher run."""

import json
from time import sleep

from control import ROOT


def main():
    seen = set()
    while True:
        for path in sorted(ROOT.glob("cohorts/*/milestones/RAW*.json")):
            if path in seen:
                continue
            saved = json.loads(path.read_text())
            if saved["step"] not in (48, 96, 144, 192):
                raise ValueError("Unexpected training milestone")
            print(path.parents[1].name, saved["step"], saved["gain"],
                  [x["integrity"]["status"] for x in saved["cases"]], flush=True)
            seen.add(path)
        for kind in ("train", "tune"):
            for path in sorted(ROOT.glob(f"{kind}-*.terminal.json")):
                if path in seen:
                    continue
                saved = json.loads(path.read_text())
                print(path.name, saved.get("returncode"), saved.get("reason"), flush=True)
                seen.add(path)
        sleep(30)


if __name__ == "__main__":
    main()
