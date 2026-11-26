"""Rank-1 identification: is the identity retrievable from a single band?

Gallery = the 60 source faces (30 pairs). Query = a band of the hybrid image.
This avoids the reconstruction problem entirely: we ask only "who is it?",
which is what the identity information is for.

Run:  python rank1.py
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arcface
import common

SIGMA = 6.0


def main():
    gallery, pair_ids, tags = [], [], []
    for k in range(30):
        for tag in ("E", "M"):
            gallery.append(cv2.imread(f"data/truth/{k:02d}_{tag}.png", 0).astype(np.float32) / 255)
            pair_ids.append(k)
            tags.append(tag)
    ge = arcface.embed_many(gallery)
    pair_ids = np.array(pair_ids)
    tags = np.array(tags)

    rows = []
    for k in range(30):
        H = cv2.imread(f"data/hybrid/{k:02d}_s6.png", 0).astype(np.float32) / 255
        W = arcface.warp_matrix(H)
        LP, HP = common.band_split(H, SIGMA)
        # the query we expect: high band -> E of this pair, low band -> M of this pair
        for name, img, want in (("H", H, None), ("LP", LP, "M"), ("HP", HP + 0.5, "E")):
            e = arcface.embed_with(np.clip(img, 0, 1), W)
            sims = ge @ e
            order = np.argsort(-sims)
            top = order[0]
            rows.append({
                "pair": k, "comp": name,
                "top1_pair": int(pair_ids[top]), "top1_tag": str(tags[top]),
                "top1_sim": float(sims[top]),
                "rank_of_own_pair": int(np.where(pair_ids[order] == k)[0][0]) + 1,
            })

    df = pd.DataFrame(rows)
    df.to_csv("results_rank1.csv", index=False)

    print("=== rank-1 over 60 gallery identities ===")
    for comp in ("H", "LP", "HP"):
        d = df[df["comp"] == comp]
        acc = (d["top1_pair"] == d["pair"]).mean()
        print(f"{comp:3s} correct pair: {acc:5.1%}    median rank of own pair: {d['rank_of_own_pair'].median():.0f}")

    d = df[df["comp"] == "HP"]
    print("HP: top-1 tag distribution:", d["top1_tag"].value_counts().to_dict())
    d = df[df["comp"] == "LP"]
    print("LP: top-1 tag distribution:", d["top1_tag"].value_counts().to_dict())


if __name__ == "__main__":
    main()
