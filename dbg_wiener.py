"""Check a two-step pipeline: Wiener deconvolution, then face-subspace projection.

Run:  python dbg_wiener.py
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


def tf(shape, sigma):
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))


def wiener(y, H, K):
    Y = np.fft.fft2(y)
    X = Y * np.conj(H) / (np.abs(H) ** 2 + K)
    return np.real(np.fft.ifft2(X))


def build_basis(n_faces=280, k=150):
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


def project(X, mu, comps, lam=0.1):
    Phi = comps.T                      # D x k
    b = (X.ravel() - mu)
    A = Phi.T @ Phi
    A += lam * np.trace(A) / len(A) * np.eye(len(A), dtype=np.float32)
    c = np.linalg.solve(A, Phi.T @ b)
    return np.clip((mu + Phi @ c).reshape(512, 512), 0, 1)


def main():
    mu, comps = build_basis()
    G = tf((512, 512), SIGMA)
    for K in (1e-4, 1e-3, 1e-2):
        res = {"low": [], "high": []}
        for i in range(5):
            E = cv2.imread(f"data/truth/{i:02d}_E.png", 0).astype(np.float32) / 255
            M = cv2.imread(f"data/truth/{i:02d}_M.png", 0).astype(np.float32) / 255
            H = cv2.imread(f"data/hybrid/{i:02d}_s6.png", 0).astype(np.float32) / 255
            L, B = common.band_split(H, SIGMA)
            W = arcface.warp_matrix(H)
            eE, eM = arcface.embed(E), arcface.embed(M)
            for mode, y, Htf in (("low", L, G), ("high", B, 1 - G)):
                xw = wiener(y, Htf, K)
                X = project(xw, mu, comps)
                e = arcface.embed_with(X, W)
                res[mode].append((float(e @ eE), float(e @ eM)))
        for mode in res:
            a = np.array(res[mode]).mean(0)
            print(f"K={K:g} {mode:4s} cos_E={a[0]:.3f} cos_M={a[1]:.3f}")


if __name__ == "__main__":
    main()
