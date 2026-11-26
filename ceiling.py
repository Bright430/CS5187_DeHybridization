"""Oracle ceiling: how much identity survives the face subspace itself?

Project the GROUND TRUTH faces onto the same 150-dim basis used by the
recovery step. Anything lost here is lost by the prior, not by the band
decomposition, so it bounds every linear-subspace reconstruction.

Run:  python ceiling.py
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arcface
import recover_full as rf


def main():
    mu, comps = rf.build_basis()
    s2 = (comps ** 2).sum(axis=1)          # squared column norms

    rows = []
    for k in range(30):
        E = cv2.imread(f"data/truth/{k:02d}_E.png", 0).astype(np.float32) / 255
        M = cv2.imread(f"data/truth/{k:02d}_M.png", 0).astype(np.float32) / 255
        eE, eM = arcface.embed(E), arcface.embed(M)
        WE, WM = arcface.warp_matrix(E), arcface.warp_matrix(M)
        for name, T, tgt, oth, W in (("E", E, eE, eM, WE), ("M", M, eM, eE, WM)):
            b = T.ravel() - mu
            c = (comps @ b) / s2             # least-squares coefficients
            X = np.clip((mu + comps.T @ c).reshape(512, 512), 0, 1)
            e = arcface.embed_with(X, W)
            rows.append({"pair": k, "which": name, "target": float(e @ tgt),
                         "other": float(e @ oth),
                         "psnr": float(10 * np.log10(1 / np.mean((T - X) ** 2)))})

    df = pd.DataFrame(rows)
    df.to_csv("results_ceiling.csv", index=False)
    g = df.groupby("which")[["target", "other", "psnr"]].mean()
    g["acc"] = df.groupby("which").apply(lambda d: (d.target > d.other).mean())
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
