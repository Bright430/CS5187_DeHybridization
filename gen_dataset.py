"""Pass 3: build the stimulus set and its ground truth.

For each pair (E, M) and each sigma, writes
    data/truth/{k}_E.png, {k}_M.png          aligned sources (grayscale, 512)
    data/hybrid/{k}_s{sigma}.png             LP(M) + HP(E)

Run:  python gen_dataset.py
"""

import io
import json
import os
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common

PARQUET = "data/celeba_hq_1k5.parquet"
SIGMAS = [3.0, 4.0, 6.0, 8.0, 12.0]
ALPHA = 0.7      # balanced setting from alpha_sweep.py (margin ~ 0 at zero blur)


def main():
    pairs = json.load(open("data/pairs.json"))
    os.makedirs("data/truth", exist_ok=True)
    os.makedirs("data/hybrid", exist_ok=True)

    need = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    table = pq.read_table(PARQUET, columns=["image"])
    print(f"reading {len(need)} rows for {len(pairs)} pairs")

    faces = {}
    for i in need:
        raw = table.column("image")[i].as_py()["bytes"]
        img = np.array(Image.open(io.BytesIO(raw)).convert("L"))
        A = common.align(img)
        if A is None:
            raise RuntimeError(f"face {i} failed to align on the second pass")
        faces[i] = A

    for k, p in enumerate(pairs):
        E, M = faces[p["a"]], faces[p["b"]]
        cv2.imwrite(f"data/truth/{k:02d}_E.png", common.to_u8(E))
        cv2.imwrite(f"data/truth/{k:02d}_M.png", common.to_u8(M))
        for s in SIGMAS:
            H = common.hybrid(E, M, sigma=s, alpha=ALPHA)
            cv2.imwrite(f"data/hybrid/{k:02d}_s{s:g}.png", common.to_u8(H))

    with open("data/stimulus_meta.json", "w") as f:
        json.dump({"alpha": ALPHA, "sigmas": SIGMAS, "pairs": len(pairs)}, f, indent=1)

    print(f"wrote {len(pairs)} pairs x {len(SIGMAS)} sigmas")


if __name__ == "__main__":
    main()
