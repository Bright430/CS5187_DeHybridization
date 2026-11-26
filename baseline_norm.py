"""Same as baseline.py, but each band is contrast-normalised before embedding.

Why: raw bands carry very different energy (the low band loses contrast as sigma
grows, the high band gains it), so a raw comparison measures contrast, not
information. Standardising every component to the same mean/std makes the
numbers comparable across bands and across sigmas.

Run:  python baseline_norm.py
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


def norm(x, mean=0.5, std=0.22):
    x = x.astype(np.float32)
    s = x.std() + 1e-6
    return np.clip((x - x.mean()) / s * std + mean, 0, 1)


def main():
    rows = []
    files = sorted(glob.glob("data/hybrid/*.png"))
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
        W = arcface.warp_matrix(H)
        for name, img in (("H", H), ("LP", LP), ("HP", HP + 0.5)):
            e = arcface.embed_with(norm(img), W)
            rows.append({"pair": k, "sigma": sigma, "comp": name,
                         "cos_E": arcface.cos(e, eE), "cos_M": arcface.cos(e, eM)})

    df = pd.DataFrame(rows)
    df["margin"] = np.where(df["comp"] == "LP", df["cos_M"] - df["cos_E"],
                            df["cos_E"] - df["cos_M"])
    df.to_csv("results_baseline_norm.csv", index=False)

    print("=== contrast-normalised identity cosine ===")
    print(df.groupby(["comp", "sigma"])[["cos_E", "cos_M", "margin"]].mean().round(3).to_string())
    print("\n=== accuracy ===")
    print(df.assign(c=df["margin"] > 0).groupby(["comp", "sigma"])["c"].mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
