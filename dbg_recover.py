"""Quick check: does a bigger, correctly scaled face subspace help?

Run:  python dbg_recover.py
"""

import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arcface
import common

SIGMA = 6.0


def build_basis(n_faces=280, k=150):
    files = sorted(glob.glob("data/aligned/*.png"))[:n_faces]
    X = np.stack([cv2.imread(f, 0).astype(np.float32).ravel() / 255 for f in files])
    mu = X.mean(0)
    Xc = X - mu
    G = Xc @ Xc.T / (len(X) - 1)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:k]
    u = V[:, order].T @ Xc                      # k x D
    u /= np.sqrt((u ** 2).mean(axis=1, keepdims=True))
    s = (Xc @ u.T).std(axis=0)                   # data scale of each direction
    comps = u * s[:, None]
    return mu.astype(np.float32), comps.astype(np.float32)


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
    for k in (40, 150):
        mu, comps = build_basis(k=k)
        Al, ml = op_cols(mu, comps, "low")
        Ah, mh = op_cols(mu, comps, "high")
        out = {"low": [], "high": []}
        for i in range(5):
            E = cv2.imread(f"data/truth/{i:02d}_E.png", 0).astype(np.float32) / 255
            M = cv2.imread(f"data/truth/{i:02d}_M.png", 0).astype(np.float32) / 255
            H = cv2.imread(f"data/hybrid/{i:02d}_s6.png", 0).astype(np.float32) / 255
            L, B = common.band_split(H, SIGMA)
            W = arcface.warp_matrix(H)
            eE, eM = arcface.embed(E), arcface.embed(M)
            for mode, A, m, y in (("low", Al, ml, L.ravel()), ("high", Ah, mh, B.ravel())):
                c = solve(A, y - m)
                X = np.clip((mu + comps.T @ c).reshape(512, 512), 0, 1)
                e = arcface.embed_with(X, W)
                out[mode].append((float(e @ eE), float(e @ eM)))
        for mode in out:
            a = np.array(out[mode]).mean(0)
            print(f"k={k:3d} {mode:4s}  cos_E={a[0]:.3f}  cos_M={a[1]:.3f}")


if __name__ == "__main__":
    main()
