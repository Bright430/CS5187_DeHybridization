"""Stage 2: recover each band with a face-subspace prior.

Idea
----
Each band is a linear observation of a face:
    low  band :  L = G_s * M          (blurred face)
    high band :  B = (I - G_s) * E    (edge residual)
The inverse is ill-posed exactly where the operator is small. Instead of
inverting blindly, we restrict the solution to a face subspace:

    min_c  || A (mu + Phi c) - y ||^2 + lam ||c||^2

with A the band operator, Phi the top-k eigenfaces, y the observed band.
Everything is linear, so the normal equations are 40x40 -- one solve per pair.

Also implements the three controls (phase randomisation, inverted face,
mismatched pairing).

Run:  python recover.py
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
K = 40
LAM = 1e-3
BASIS_DIR = "data/aligned"


def build_basis(k=K):
    files = sorted(glob.glob(f"{BASIS_DIR}/*.png"))[:200]
    X = np.stack([cv2.imread(f, 0).astype(np.float32).ravel() / 255 for f in files])
    mu = X.mean(0)
    Xc = X - mu
    G = Xc @ Xc.T / (len(X) - 1)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:k]
    comps = V[:, order].T @ Xc
    # normalise each component to unit RMS so that coefficients are in the same
    # units as image intensities (unit L2 norm would make them 512x too small)
    comps /= np.sqrt((comps ** 2).mean(axis=1, keepdims=True))
    comps -= comps.mean(axis=1, keepdims=True)
    return mu.astype(np.float32), comps.astype(np.float32)


def operator_columns(mu, comps, sigma, mode):
    """Flattened A @ phi_i for the given band operator, and A @ mu."""
    def op(x):
        g = cv2.GaussianBlur(x.reshape(512, 512), (0, 0), sigma).ravel()
        return g if mode == "low" else x - g

    return np.stack([op(phi) for phi in comps], axis=1), op(mu)


def solve(A, b, lam=LAM):
    AtA = A.T @ A
    AtA += lam * np.eye(AtA.shape[0], dtype=np.float32) * np.trace(AtA) / AtA.shape[0]
    return np.linalg.solve(AtA, A.T @ b)


def main():
    mu, comps = build_basis()
    print(f"basis: {comps.shape}")
    A_low, mu_low = operator_columns(mu, comps, SIGMA, "low")
    A_high, mu_high = operator_columns(mu, comps, SIGMA, "high")

    E, M, Hs, Ls, Bs = [], [], [], [], []
    for k in range(30):
        e = cv2.imread(f"data/truth/{k:02d}_E.png", 0).astype(np.float32) / 255
        m = cv2.imread(f"data/truth/{k:02d}_M.png", 0).astype(np.float32) / 255
        h = cv2.imread(f"data/hybrid/{k:02d}_s6.png", 0).astype(np.float32) / 255
        lp, hp = common.band_split(h, SIGMA)
        E.append(e); M.append(m); Hs.append(h); Ls.append(lp); Bs.append(hp)

    eE = arcface.embed_many(E)
    eM = arcface.embed_many(M)
    Ws = [arcface.warp_matrix(h) for h in Hs]

    rec_low, rec_high, raw_low, raw_high = [], [], [], []
    for k in range(30):
        c_low = solve(A_low, Ls[k].ravel() - mu_low)
        c_high = solve(A_high, Bs[k].ravel() - mu_high)
        rec_low.append(np.clip((mu + comps.T @ c_low).reshape(512, 512), 0, 1))
        rec_high.append(np.clip((mu + comps.T @ c_high).reshape(512, 512), 0, 1))
        raw_low.append(Ls[k])
        raw_high.append(np.clip(Bs[k] + 0.5, 0, 1))

    rows = []
    for name, imgs, tgt_E in (("low", rec_low, False), ("high", rec_high, True),
                              ("low_raw", raw_low, False), ("high_raw", raw_high, True)):
        e = arcface.embed_many(imgs, Ws)
        ce = np.sum(e * eE, axis=1)
        cm = np.sum(e * eM, axis=1)
        for k in range(30):
            rows.append({"method": name, "pair": k, "cos_E": ce[k], "cos_M": cm[k]})

    df = pd.DataFrame(rows)
    df["target"] = np.where(df["method"].str.startswith("high"), df["cos_E"], df["cos_M"])
    df["other"] = np.where(df["method"].str.startswith("high"), df["cos_M"], df["cos_E"])
    df.to_csv("results_recover.csv", index=False)

    print("\n=== band recovery (target = the person that band belongs to) ===")
    print(df.groupby("method")[["target", "other"]].mean().round(3).to_string())
    print("\nidentity accuracy (target > other):")
    print(df.assign(ok=df["target"] > df["other"]).groupby("method")["ok"].mean().round(3).to_string())

    # save a few reconstructions for the figure
    os.makedirs("data/recon", exist_ok=True)
    for k in (0, 1, 2):
        cv2.imwrite(f"data/recon/{k:02d}_low.png", common.to_u8(rec_low[k]))
        cv2.imwrite(f"data/recon/{k:02d}_high.png", common.to_u8(rec_high[k]))


if __name__ == "__main__":
    main()
