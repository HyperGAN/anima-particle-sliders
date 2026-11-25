"""Check clean and deliberately damaged versions of unseen art using the frozen rubric."""
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from review import HERE, ROOT, assess, harness


def main():
    board = harness.read(ROOT / "storage/alpha-standings-v1.json")
    images = {}
    # Independent of the opening-four calibration preference. Reproducible defects.
    for name in ("smith", "violinist", "guard"):
        original = board["rows"][0]["cases"][name]["off"]
        images[name + "-original"] = original
        with Image.open(original["path"]) as source:
            for effect in ("scanline", "erasure"):
                image = source.convert("RGB")
                w,h = image.size
                if effect == "scanline":
                    copy = image.copy()
                    for y in range(h//9, 2*h//3, h//24):
                        shift = (y // (h//24) % 3 + 1) * w//9
                        image.paste(copy.crop((0, y, w-shift, y+h//32)), (shift,y))
                else:
                    image.paste((255,255,255), (w//5,h//6,4*w//5,3*h//5))
                out = HERE / "transfer" / f"{name}-{effect}.png"
                out.parent.mkdir(exist_ok=True)
                image.save(out)
                images[f"{name}-{effect}"] = harness.pin(out)
    with ThreadPoolExecutor(4) as pool:
        results=dict(zip(images,pool.map(assess,images.values())))
    print(results,flush=True)
    harness.write(HERE / "transfer.json",{"images": {k:v['sha256'] for k,v in images.items()}, "damage":results})
    assert all(results[f"{c}-{s}"] > results[c+"-original"] for c in ("smith","violinist","guard")
               for s in ("scanline","erasure"))


if __name__ == '__main__':
    main()
