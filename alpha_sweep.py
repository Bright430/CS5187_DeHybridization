"""Alpha sweep: balance the two bands so the composite is ambiguous at zero blur.

For each alpha we build the hybrid at sigma = 6 and measure, at viewing blur 0,
the identity cosine to both sources. The balanced alpha (margin closest to 0)
is used for the recovery experiments.

Run:  python alpha_sweep.py
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arcface
import common

ALPHAS = [0.3, 0.4, 0.5, 0.7, 1.0]
SIGMA = 6.0


def main():
    E = [cv2.imread(f"data/truth/{k:02d}_E.png", 0).astype(np.float32) / 255 for k in range(30)]
    M = [cv2.imread(f"data/truth/{k:02d}_M.png", 0).astype(np.float32) / 255 for k in range(30)]
    eE = arcface.embed_many(E)
    eM = arcface.embed_many(M)

    rows = []
    for a in ALPHAS:
        Hs = [common.hybrid(E[k], M[k], SIGMA, a) for k in range(30)]
        Ws = [arcface.warp_matrix(h) for h in Hs]
        eH = arcface.embed_many(Hs, Ws)
        for k in range(30):
            rows.append({"alpha": a, "pair": k,
                         "cos_E": float(eH[k] @ eE[k]), "cos_M": float(eH[k] @ eM[k])})
        print(f"  alpha={a} done")

    df = pd.DataFrame(rows)
    df["margin"] = df["cos_M"] - df["cos_E"]
    df.to_csv("results_alpha.csv", index=False)
    print(df.groupby("alpha")[["cos_E", "cos_M", "margin"]].mean().round(3).to_string())
    print("\nshare read as E:", df.assign(c=df["margin"] < 0).groupby("alpha")["c"].mean().round(3).to_dict())
    print("mean |margin|:", df.groupby("alpha")["margin"].apply(lambda s: s.abs().mean()).round(3).to_dict())


if __name__ == "__main__":
    main()
