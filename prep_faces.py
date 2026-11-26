"""Pass 1: decode, align, embed every face in the CelebA-HQ shard.

Outputs
-------
data/meta.npz        index, embedding, geometry, valid flag
data/aligned/*.png   a subset of aligned faces, used to build the PCA face prior

Run:  python prep_faces.py
"""

import io
import os
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arcface
import common

PARQUET = "data/celeba_hq_1k5.parquet"
BASIS_N = 300


def main():
    os.makedirs("data/aligned", exist_ok=True)
    pf = pq.ParquetFile(PARQUET)

    embs, geos, valid, idx = [], [], [], []
    saved = 0
    i = 0
    for batch in pf.iter_batches(batch_size=50):
        for row in batch.to_pylist():
            img = np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("L"))
            A = common.align(img)
            ok = A is not None
            if ok:
                try:
                    e = arcface.embed(A)
                    g = common.geometry(A)
                except Exception:
                    ok, e, g = False, None, None
            embs.append(e if ok else np.zeros(512, np.float32))
            geos.append(g if ok else np.zeros(6, np.float32))
            valid.append(ok)
            idx.append(i)
            if ok and saved < BASIS_N:
                cv2.imwrite(f"data/aligned/{i:05d}.png", common.to_u8(A))
                saved += 1
            i += 1
            if i % 200 == 0:
                print(f"  {i} done, {sum(valid)} usable")

    np.savez_compressed("data/meta.npz",
                        idx=np.array(idx), emb=np.stack(embs),
                        geo=np.stack(geos), valid=np.array(valid))
    print(f"faces: {len(idx)}, usable: {sum(valid)}, basis saved: {saved}")


if __name__ == "__main__":
    main()
