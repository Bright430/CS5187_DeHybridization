"""Baseline: how much identity survives in each band after a linear split?

For every hybrid image we embed
    H      the composite
    LP     blurred low band      (~ blurred M)
    HP      high band + 0.5      (~ edges of E)
and compare each with the two ground-truth identities.

Run:  python baseline.py
"""

import glob
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arcface
import common


def main():
    rows = []
    files = sorted(glob.glob("data/hybrid/*.png"))
    print(f"{len(files)} hybrid images")

    cache = {}
    for f in files:
        m = re.match(r"data/hybrid/(\d+)_s([\d.]+)\.png", f.replace("\\", "/"))
        k, sigma = int(m.group(1)), float(m.group(2))
        H = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        if k not in cache:
            E = cv2.imread(f"data/truth/{k:02d}_E.png", 0).astype(np.float32) / 255.0
            M = cv2.imread(f"data/truth/{k:02d}_M.png", 0).astype(np.float32) / 255.0
            cache[k] = (arcface.embed(E), arcface.embed(M))
        eE, eM = cache[k]

        LP, HP = common.band_split(H, sigma)
        W = arcface.warp_matrix(H)      # one geometry for all components
        for name, img in (("H", H), ("LP", LP), ("HP", HP + 0.5)):
            e = arcface.embed_with(img, W)
            rows.append({"pair": k, "sigma": sigma, "comp": name,
                         "cos_E": arcface.cos(e, eE), "cos_M": arcface.cos(e, eM)})

    df = pd.DataFrame(rows)
    df["margin"] = np.where(df["comp"] == "LP", df["cos_M"] - df["cos_E"],
                            df["cos_E"] - df["cos_M"])
    df["correct"] = df["margin"] > 0
    df.to_csv("results_baseline.csv", index=False)

    print("\n=== mean identity cosine by component ===")
    piv = df.groupby(["comp", "sigma"])[["cos_E", "cos_M", "margin", "correct"]].mean()
    print(piv.round(3).to_string())

    print("\n=== accuracy (which identity wins) ===")
    acc = df.groupby(["comp", "sigma"])["correct"].mean().unstack()
    print(acc.round(3).to_string())


if __name__ == "__main__":
    main()
