"""Full recovery experiment: all 30 pairs, 150-dim face subspace.

Reports, for each band:
  raw       - identity of the band itself, no reconstruction (upper reference)
  pca       - least-squares inversion inside the face subspace
  ceiling   - subspace projection of the ground truth (best the basis can do)

Run:  python recover_full.py
"""

import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arcface
import common

SIGMA = 6.0
K = 150


def build_basis(n_faces=280, k=K):
    files = sorted(glob.glob("data/aligned/*.png"))[:n_faces]
    X = np.stack([cv2.imread(f, 0).astype(np.float32).ravel() / 255 for f in files])
    mu = X.mean(0)
    Xc = X - mu
    G = Xc @ Xc.T / (len(X) - 1)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:k]
    u = V[:, order].T @ Xc
    u /= np.sqrt((u ** 2).mean(axis=1, keepdims=True))
    s = (Xc @ u.T).std(axis=0)
    return mu.astype(np.float32), (u * s[:, None]).astype(np.float32)


def op_cols(mu, comps, mode):
    def op(x):
        g = cv2.GaussianBlur(x.reshape(512, 512), (0, 0), SIGMA).ravel()
        return g if mode == "low" else x - g

    return np.stack([op(p) for p in comps], 1), op(mu)


def solve(A, b, rel=1e-4):
    AtA = A.T @ A
    AtA += rel * np.trace(AtA) / len(AtA) * np.eye(len(AtA), dtype=np.float32)
    return np.linalg.solve(AtA, A.T @ b)


def main():
    mu, comps = build_basis()
    print(f"basis: {comps.shape[1]} components from 280 aligned faces", flush=True)

    ops = {}
    for mode in ("low", "high"):
        A, m = op_cols(mu, comps, mode)
        ops[mode] = (A, m)

    rows = []
    for k in range(30):
        E = cv2.imread(f"data/truth/{k:02d}_E.png", 0).astype(np.float32) / 255
        M = cv2.imread(f"data/truth/{k:02d}_M.png", 0).astype(np.float32) / 255
        H = cv2.imread(f"data/hybrid/{k:02d}_s6.png", 0).astype(np.float32) / 255
        L, B = common.band_split(H, SIGMA)
        W = arcface.warp_matrix(H)
        eE, eM = arcface.embed(E), arcface.embed(M)

        # bands as-is
        for name, img, tgt in (("low", L, eM), ("high", np.clip(B + 0.5, 0, 1), eE)):
            e = arcface.embed_with(img.astype(np.float32), W)
            rows.append({"pair": k, "method": name + "_raw",
                         "target": float(e @ tgt), "other": float(e @ (eE if name == "low" else eM))})

        # subspace inversion
        for name, y, tgt in (("low", L.ravel(), eM), ("high", B.ravel(), eE)):
            A, m = ops[name]
            c = solve(A, y - m)
            X = np.clip((mu + comps.T @ c).reshape(512, 512), 0, 1)
            e = arcface.embed_with(X, W)
            rows.append({"pair": k, "method": name + "_pca",
                         "target": float(e @ tgt), "other": float(e @ (eE if name == "low" else eM))})

        # ceiling: project the ground truth
        for name, T, tgt in (("low", M, eM), ("high", E, eE)):
            c = comps @ (T.ravel() - mu)
            X = np.clip((mu + comps.T @ c).reshape(512, 512), 0, 1)
            e = arcface.embed_with(X, W)
            rows.append({"pair": k, "method": name + "_ceiling",
                         "target": float(e @ tgt), "other": float(e @ (eE if name == "low" else eM))})
        if k % 5 == 0:
            print(f"  {k}/30", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results_recover_full.csv", index=False)
    g = df.groupby("method")[["target", "other"]].mean()
    g["acc"] = df.groupby("method").apply(lambda d: (d.target > d.other).mean())
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
