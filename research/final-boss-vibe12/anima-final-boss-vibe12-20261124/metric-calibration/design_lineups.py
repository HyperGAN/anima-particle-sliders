"""Blindly compare the opening-four and late-cue lineups as complete sets of character designs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from design_range import OUT, ROOT, harness, judge
from preference_round import QUESTIONS as PAIR_QUESTIONS

STATE = "Image_1 and image_2 show alternative lineups. Corresponding tiles depict edits of the same four source characters."
QUESTION = {"preference":dict(PAIR_QUESTIONS["preference"], instructions=(
    "Which lineup gives its characters more individually recognizable Final Boss designs? Consider whether "
    "each encounter has its own visual identity, silhouette or signature motif, and whether a viewer could "
    "tell these opponents apart without relying on the background or palette. Minor variations on one stock "
    "boss costume add little range. Ordinary people do not count as successful Final Boss designs. "
    "Judge the drawings shown; choose the middle if the sets offer equivalent design range."),
    criteria=["Image_1 clearly more distinct","Image_1 slightly more distinct","Equivalent design range",
              "Image_2 slightly more distinct","Image_2 clearly more distinct"])}
METHODS = ("shared-best-opening","late-cues")


def main():
    judge()
    from particle_sliders.alpha_tune_jev import JevJudge, _Image, _digest, JEV_MODEL, JEV_URL
    from particle_sliders.candidate_failure import FileRecord
    client = JevJudge(OUT / "lineup-receipts","Individually recognizable Final Boss designs")
    d=harness.read(ROOT/"storage/alpha-standings.json")
    rows={m:next(r for r in d["rows"] if r["best_for_recipe"] and r["method"]==m) for m in METHODS}
    screen={}
    for m,r in rows.items():
        path=ROOT/"strength-search/leaderboard-retune-visible"/m/"renders"/r["profile"]/r["alpha"].hex()/"search.json"
        screen[m]=harness.read(path)["rows"]
    names=d["cases"]
    groups={"first-four":names[:4],"last-four":names[4:],"alternating-a":names[::2],"alternating-b":names[1::2],
            "search-scenes":list(screen[METHODS[0]])}
    def sheet(records):
        hashes=[r["sha256"] for r in records]
        path=OUT/"lineups"/(_digest(hashes)+".png")
        path.parent.mkdir(exist_ok=True,parents=True)
        grid=Image.new("RGB",(640,640),(28,28,28))
        for i,r in enumerate(records):
            if harness.pin(Path(r["path"]))["sha256"]!=r["sha256"]:
                raise ValueError("Unbound lineup input")
            with Image.open(r["path"]) as source:
                art=source.convert("RGB")
            art.thumbnail((320,320),Image.Resampling.LANCZOS)
            grid.paste(art,((i%2)*320+(320-art.width)//2,(i//2)*320+(320-art.height)//2))
        grid.save(path)
        ref=harness.pin(path)
        return _Image.read(FileRecord(str(path),path.stat().st_size,ref["sha256"])),hashes
    def decide(a,b):
        binding=dict(endpoint=JEV_URL,model=JEV_MODEL,state=STATE,questions=QUESTION,images=[x.identity() for x in (a,b)])
        values=client._decide((a,b),STATE,QUESTION)
        if values is None:
            raise ValueError("Missing lineup comparison")
        return {"value":values["preference"],"receipt":_digest(binding)}
    results={}
    with ThreadPoolExecutor(6) as pool:
        jobs=[]
        for name,cases in groups.items():
            panes=[]
            for m in METHODS:
                source=screen[m] if name=="search-scenes" else rows[m]["cases"]
                panes.append(sheet([source[c]["on"] for c in cases]))
            (a,ah),(b,bh)=panes
            jobs.append((name,ah,bh,[a.identity()["sha256"],b.identity()["sha256"]],pool.submit(decide,a,b),pool.submit(decide,b,a)))
        for name,ah,bh,sheet_images,forward,reverse in jobs:
            ab,ba=forward.result(),reverse.result()
            results[name]={"cases":groups[name],"opening":ah,"late":bh,"sheet_images":sheet_images,"forward":ab,"reverse":ba,
                           "opening_margin":ba["value"]-ab["value"]}
    output={"state":STATE,"questions":QUESTION,"model":JEV_MODEL,"methods":METHODS,
            "settings":{m:{k:r[k] for k in ("profile","alpha")} for m,r in rows.items()},"groups":results}
    harness.write(OUT/"lineups.json",output)
    print({name:round(r["opening_margin"]*100,2) for name,r in results.items()},flush=True)


if __name__=="__main__":
    main()
