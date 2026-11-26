"""Reproduce the retrieval-margin evidence requested for item 2.

Run:  python review_retrieval.py
Needs: data/truth/*.png, data/hybrid/*_s6.png, arcface.py, common.py.
CPU only; takes a few minutes.

Prints
  1) pairwise cosine range over the 60 source faces (gallery degeneracy)
  2) cross-pair retrieval margin = own-source similarity minus the best
     similarity to ANY face from a different pair, for H / LP / HP
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
import arcface
import common

SIGMA = 6.0


def main():
    gallery = [
        cv2.imread(f"data/truth/{k:02d}_{tag}.png", 0).astype(np.float32) / 255
        for k in range(30) for tag in ("E", "M")
    ]
    g = arcface.embed_many(gallery)
    s = g @ g.T
    pair = np.repeat(np.arange(30), 2)

    within = np.array([s[i, j] for i in range(60) for j in range(i + 1, 60)
                       if pair[i] == pair[j]])
    cross = np.array([s[i, j] for i in range(60) for j in range(i + 1, 60)
                      if pair[i] != pair[j]])
    print(f"gallery: cross-pair min {cross.min():.3f} max {cross.max():.3f} "
          f"mean {cross.mean():.3f} sd {cross.std():.3f} "
          f"p99 {np.percentile(cross, 99):.3f}")
    print(f"gallery: within-pair min {within.min():.3f} max {within.max():.3f} "
          f"mean {within.mean():.3f}")

    for name in ("H", "LP", "HP"):
        margins = []
        for k in range(30):
            H = cv2.imread(f"data/hybrid/{k:02d}_s6.png", 0).astype(np.float32) / 255
            LP, HP = common.band_split(H, SIGMA)
            W = arcface.warp_matrix(H)
            img = {"H": H, "LP": LP, "HP": np.clip(HP + 0.5, 0, 1)}[name]
            e = arcface.embed_with(img, W)
            sim = g @ e
            own = max(sim[2 * k], sim[2 * k + 1]) if name == "H" else (
                sim[2 * k + 1] if name == "LP" else sim[2 * k])
            wrong = np.delete(sim, [2 * k, 2 * k + 1]).max()
            margins.append(own - wrong)
        m = np.asarray(margins)
        print(f"{name}: margin mean {m.mean():.3f} sd {m.std(ddof=1):.3f} "
              f"min {m.min():.3f} max {m.max():.3f} "
              f"positive in {(m > 0).sum()}/30 pairs")


if __name__ == "__main__":
    main()
