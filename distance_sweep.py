"""The core figure: does the composite switch identity as it is blurred?

We hold the generation setting fixed (sigma_gen = 6) and apply increasing
viewing blur. Large sigma_view = standing further away.

Run:  python distance_sweep.py
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arcface

SIGMA_GEN = 6.0
VIEW = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]


def main():
    rows = []
    for k in range(30):
        H = cv2.imread(f"data/hybrid/{k:02d}_s{SIGMA_GEN:g}.png", 0).astype(np.float32) / 255.0
        E = cv2.imread(f"data/truth/{k:02d}_E.png", 0).astype(np.float32) / 255.0
        M = cv2.imread(f"data/truth/{k:02d}_M.png", 0).astype(np.float32) / 255.0
        eE, eM = arcface.embed(E), arcface.embed(M)
        W = arcface.warp_matrix(H)
        for sv in VIEW:
            img = H if sv == 0 else cv2.GaussianBlur(H, (0, 0), sv)
            e = arcface.embed_with(img, W)
            rows.append({"pair": k, "view": sv,
                         "cos_E": arcface.cos(e, eE), "cos_M": arcface.cos(e, eM)})
        if k % 5 == 0:
            print(f"  pair {k}")

    df = pd.DataFrame(rows)
    df["margin"] = df["cos_M"] - df["cos_E"]
    df.to_csv("results_distance.csv", index=False)
    print(df.groupby("view")[["cos_E", "cos_M", "margin"]].mean().round(3).to_string())
    print("\nshare of pairs seen as Marilyn:")
    print(df.assign(m=df["margin"] > 0).groupby("view")["m"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
