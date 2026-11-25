"""Score each unique finished image once as new eight-case results arrive."""
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from review import assess, saved, judge, harness, ROOT, HERE, BINDING


def main():
    check = harness.read(HERE / "audit.json")
    if (check.get("binding") != BINDING or check["damage"]["broken"] <= .65 or check["damage"]["clean"] >= .35):
        raise ValueError("Expected the calibrated rubric")
    judge()
    with ThreadPoolExecutor(12) as pool:
        while True:
            images = {}
            # A single snapshot keeps the evaluated recipes on the same score version.
            for row in harness.read(ROOT / "storage/alpha-standings-v1.json")["rows"]:
                for case in row["cases"].values():
                    image = case["on"]
                    images[image["sha256"]] = image
            pending = [v for k,v in images.items() if saved(k) is None]
            if pending:
                print(f"{len(pending)} of {len(images)} unique finishes to judge", flush=True)
                for n,v in enumerate(pool.map(assess, pending), 1):
                    if n % 100 == 0:
                        print(n, "reviewed", flush=True)
            sleep(10)


if __name__ == "__main__":
    main()
