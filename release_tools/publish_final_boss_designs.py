"""Export the cross-character design review, matched sample strips and replayable decisions."""
import argparse
import gzip
import json
from pathlib import Path

from release_tools.final_boss_default import select_default
from release_tools.publish_final_boss_vibe12 import read
from release_tools.verify_final_boss_designs import verify_panel, verify_orders


def export(campaign, folder):
    source=campaign/"metric-calibration/design-range"
    tuned=campaign/"storage/distinct-alpha-standings.json"
    dev=read(tuned if tuned.is_file() else source/"latest.json")
    screen=read(source/"search-scene-check.json")
    lineups=read(source/"lineups.json")
    board=read(folder/"standings.json")
    settings={(s["method"],s["profile"],s["alpha"]):s for s in board["settings"]}
    labels=read(campaign/"recipes.json")
    receipts={}
    for report in (dev,screen,lineups):
        if report is lineups:
            pairs=report["groups"].values()
            stores=[source/"lineup-receipts"]
        else:
            pairs=[p for r in report["rows"] for p in r["pairs"]]+report["controls"]
            stores=[source/"receipts",source/"receipts-retry-1",source/"receipts-retry-2"]
        for pair in pairs:
            for order in ("forward","reverse"):
                key=pair[order]["receipt"]
                if key in receipts:
                    continue
                for store in stores:
                    path=store/(key+".json")
                    if path.exists():
                        proof=read(path)
                        if proof["status"]=="ok":
                            receipts[key]=proof
                            break
                if key not in receipts:
                    raise ValueError("Missing Jev decision "+key)
    verify_panel(dev,receipts,board)
    verify_panel(screen,receipts)
    for pair in lineups["groups"].values():
        verify_orders(lineups,pair,pair["sheet_images"],receipts,"preference")
    catalog=read(folder/"checkpoints.json")
    default=select_default(dev, board, catalog)
    evidence={"development":dev,"search_scenes":screen,"lineups":lineups}
    (folder/"design-range.json").write_text(json.dumps(evidence,indent=2)+"\n")
    (folder/"design-receipts.json.gz").write_bytes(gzip.compress(json.dumps(receipts,sort_keys=True,separators=(",",":")).encode(),mtime=0))
    by_base=sorted(dev["rows"],key=lambda r:(-r["base"],r["method"]))
    opening=next(r for r in dev["rows"] if r["method"]=="shared-best-opening")
    late=next(r for r in dev["rows"] if r["method"]=="late-cues")
    leader=dev["rows"][0]
    lines=["# Distinct Final Boss designs across characters","",dev["as_of"],"",
           f'**Default** · {labels[leader["method"]]} · {leader["profile"]} α {leader["alpha"]:g} · '
           f'{100*leader["score"]:.2f}/100 on eight development characters. [Replay settings and training targets](../default.json)',"",
           f'Opening-four exposure places **{dev["rows"].index(opening)+1} of {len(dev["rows"])}** across matched characters, '
           f'compared with position {by_base.index(opening)+1} on the per-image score. Its average design repetition is '
           f'{100*opening["repeat"]:.1f}/100, compared with {100*late["repeat"]:.1f}/100 for late cues. '
           f'{len(dev.get("tuned_methods",[]))} of these recipes have completed the distinct-design alpha search. Remaining recipes enter at their best measured per-image strength.', "",
           "Jev compares the character designs in each pair of finished images. It looks for repeated silhouettes, body construction, masks, costume shapes and visual motifs. "
           "A palette change or a different background alone carries little weight. On each pair, distinct bosses earn both per-image scores. "
           "If the same design repeats, the pair earns the stronger score. Jev’s continuous repetition rating interpolates between those outcomes. "
           "The per-image scores already account for matched Final Boss gain and visible damage. Zero-effect edits contribute zero.","",
           "Every row uses the same eight cases and the same RAW192 budget. Jev rates all 28 pairs in both image orders. Its repetition rating is an ordinal score on a 0–100 scale. "
           "The current alpha search uses this cross-character score on four search characters and on eight development characters for the final choice. [Saved alpha and search evidence](alpha-tuning/README.md)","",
           "| Rank | Recipe | Schedule and α | Distinct Boss score ↑ | Per-image score ↑ | Design repetition ↓ | Matched samples | RAW192 |",
           "| ---: | --- | --- | ---: | ---: | ---: | --- | --- |"]
    for rank,r in enumerate(dev["rows"],1):
        sample=settings[r["method"],r["profile"],r["alpha"]]["samples"]
        artifact=catalog["checkpoints"][r["method"]]
        download=f'https://huggingface.co/{catalog["repository"]}/resolve/{catalog["revision"]}/{artifact["file"]}'
        lines.append(f'| {rank} | {labels[r["method"]]} | {("Design-tuned · " if r["method"] in dev.get("tuned_methods",[]) else "")}{r["profile"]} {r["alpha"]:g} | {100*r["score"]:.2f} | '
                     f'{100*r["base"]:.2f} | {100*r["repeat"]:.1f} | [![Eight edits]({sample})]({sample}) | [RAW]({download}) |')
    lines+=["","## How the result depends on the characters","",
            "In a separate blind check Jev compared two four-character lineups as whole sets. "
            "It rated which lineup produced more individually recognizable Final Boss designs. Positive margins favor opening four over late cues. "
            "The two development partitions reuse the same eight images. The four search scenes are separate from those eight and were used during alpha selection.","",
            "| Matched characters | Opening-four preference margin |", "| --- | ---: |"]
    for group in lineups["groups"].values():
        lines.append(f'| {", ".join(c.title() for c in group["cases"])} | {100*group["opening_margin"]:+.1f} |')
    extra={r["method"]:r for r in screen["rows"]}
    lines+=["",f'On the four search scenes, mean design repetition is {100*extra["shared-best-opening"]["repeat"]:.1f}/100 for opening four '
            f'and {100*extra["late-cues"]["repeat"]:.1f}/100 for late cues. The development advantage in variety depends on the characters.',"",
            "Three preselected same-design examples from a separate historical confirmation set used different native renders of the same characters. "
            "Three different-design examples used the same art style. All six landed on the expected side of the scale in both image orders. "
            "These controls check whether the judge recognizes a repeat. Agreement with human rankings on new lineups remains unmeasured.","",
            "[Exact rubric, case pairs and lineups](design-range.json) · [Jev decision receipts](design-receipts.json.gz) · [Per-image rankings](README.md)","",
            "```sh","python -m release_tools.verify_final_boss_designs docs/final-boss-raw192/vibe12","```",""]
    (folder/"distinct-designs.md").write_text("\n".join(lines))
    (folder.parent/"default.json").write_text(json.dumps(default,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign",type=Path)
    p.add_argument("folder",type=Path)
    args=p.parse_args()
    export(args.campaign,args.folder)
