"""Run the previous U²-Net foreground procedure on the frozen neutral renders."""
import json
from hashlib import file_digest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from control import MANIFEST, ROOT


def sha(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def main():
    cv2.setNumThreads(2)
    source = Path.home() / '.u2net/u2net.onnx'
    model_sha256 = sha(source)
    net = cv2.dnn.readNetFromONNX(str(source))
    manifest = json.loads(MANIFEST.read_text())
    output = ROOT / 'mattes'
    output.mkdir(exist_ok=True)
    mattes = {}
    for row in manifest['rows']:
        neutral = row['selection']['neutral']['output']
        if sha(neutral['path']) != neutral['sha256']:
            raise ValueError(f"Neutral render changed: {row['id']}")
        with Image.open(neutral['path']) as image:
            small = np.asarray(image.convert('RGB').resize((320, 320), Image.Resampling.LANCZOS), np.float32)
        normalized = (small / 255 - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        net.setInput(normalized.transpose(2, 0, 1)[None])
        prediction = net.forward(net.getUnconnectedOutLayersNames()[0])[0, 0]
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite foreground mask')
        mask = cv2.resize((prediction >= .55).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count < 2:
            raise ValueError(f"No character region in {row['id']}")
        foreground = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        coverage = float(foreground.mean())
        if not .07 <= coverage <= .78:
            raise ValueError(f"Unusable foreground region {coverage:.3f} in {row['id']}")
        expanded = cv2.dilate(foreground, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)))
        soft = cv2.GaussianBlur(expanded.astype(np.float32), (31, 31), 6)
        soft[soft < .025] = 0
        destination = output / f"{row['id']}.png"
        Image.fromarray(np.round(soft * 255).astype(np.uint8)).save(destination)
        mattes[row['id']] = dict(path=str(destination), sha256=sha(destination), coverage=coverage,
                                soft_mean=float(soft.mean()), neutral_sha256=neutral['sha256'])
    (output / 'index.json').write_text(json.dumps(dict(manifest=manifest['sha256'], model_sha256=model_sha256, mattes=mattes), indent=2) + '\n')
    print(f"Verified {len(mattes)} masks, coverage {min(m['coverage'] for m in mattes.values()):.1%}–{max(m['coverage'] for m in mattes.values()):.1%}", flush=True)


if __name__ == '__main__':
    main()
