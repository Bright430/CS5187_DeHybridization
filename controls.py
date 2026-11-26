"""Controls: is the identity in the high band real, or an artefact?

1. phase randomisation - keep the amplitude spectrum, destroy the structure
2. inverted face        - same statistics, wrong orientation
3. chance level         - cosine between unrelated identities

Run:  python controls.py
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


def phase_randomize(x, seed=0):
    F = np.fft.fft2(x)
    rng = np.random.default_rng(seed)
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, F.shape))
    return np.real(np.fft.ifft2(np.abs(F) * phase))


def main():
    rows = []
    for k in range(30):
        E = cv2.imread(f"data/truth/{k:02d}_E.png", 0).astype(np.float32) / 255
        H = cv2.imread(f"data/hybrid/{k:02d}_s6.png", 0).astype(np.float32) / 255
        eE = arcface.embed(E)
        W = arcface.warp_matrix(H)
        _, B = common.band_split(H, SIGMA)

        variants = {
            "raw": B + 0.5,
            "phase_random": phase_randomize(B, seed=k) + 0.5,
            "inverted": (lambda e: e - cv2.GaussianBlur(e, (0, 0), SIGMA))(np.rot90(E, 2)) + 0.5,
        }
        for name, img in variants.items():
            e = arcface.embed_with(np.clip(img, 0, 1), W)
            rows.append({"pair": k, "variant": name, "cos_E": float(e @ eE)})

    df = pd.DataFrame(rows)
    df.to_csv("results_controls.csv", index=False)
    print(df.groupby("variant")["cos_E"].agg(["mean", "std"]).round(3).to_string())

    # chance level: cosine between different people
    E = [cv2.imread(f"data/truth/{k:02d}_E.png", 0).astype(np.float32) / 255 for k in range(30)]
    emb = arcface.embed_many(E)
    S = emb @ emb.T
    off = S[~np.eye(30, dtype=bool)]
    print(f"\nchance level (different people): mean {off.mean():.3f}  sd {off.std():.3f}")


if __name__ == "__main__":
    main()
