"""Figures for the baseline results.

fig1_baseline.png  identity cosine vs viewing blur (mean +- SEM) and the
                   share of pairs that read as the low-frequency source
fig2_strip.png     one pair rendered at four viewing blurs
"""

import os
import sys

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common


def crossover_per_pair(df):
    """Linear interpolation of the view blur where cos_M - cos_E crosses 0."""
    out = []
    for k, g in df.groupby("pair"):
        g = g.sort_values("view")
        v, m = g["view"].values, g["margin"].values
        for i in range(1, len(v)):
            if m[i - 1] < 0 <= m[i]:
                t = -m[i - 1] / (m[i] - m[i - 1])
                out.append(v[i - 1] + t * (v[i] - v[i - 1]))
                break
    return np.array(out)


def main():
    df = pd.read_csv("results_distance.csv")
    views = sorted(df["view"].unique())
    g = df.groupby("view")[["cos_E", "cos_M"]]
    mean, sem = g.mean(), g.sem()
    share = df.assign(m=df["margin"] > 0).groupby("view")["m"].mean()

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), dpi=160)
    ax[0].errorbar(views, mean["cos_E"], yerr=sem["cos_E"], marker="o", capsize=3,
                   label="cos to high-freq source (E)", color="#2b6cb0")
    ax[0].errorbar(views, mean["cos_M"], yerr=sem["cos_M"], marker="s", capsize=3,
                   label="cos to low-freq source (M)", color="#c53030")
    ax[0].set_xlabel("viewing blur  sigma_view  (px at 512)")
    ax[0].set_ylabel("identity cosine")
    ax[0].set_title("Hybrid image switches identity with blur")
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].grid(alpha=.25)

    ax[1].plot(views, share.values, marker="o", color="#2f855a")
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_xlabel("viewing blur  sigma_view")
    ax[1].set_ylabel("share of pairs seen as M")
    ax[1].set_title("Fraction of pairs that flip")
    ax[1].grid(alpha=.25)
    ax[1].axhline(0.5, ls="--", lw=.8, color="gray")

    fig.tight_layout()
    fig.savefig("fig1_baseline.png")

    cr = crossover_per_pair(df)
    if len(cr):
        print(f"crossover: mean {cr.mean():.2f}  median {np.median(cr):.2f}  "
              f"n={len(cr)}/30  (sigma_view, px at 512)")

    # qualitative strip
    k = 0
    H = cv2.imread(f"data/hybrid/{k:02d}_s6.png", 0)
    tiles = []
    for sv in (0, 4, 8, 16):
        im = H if sv == 0 else cv2.GaussianBlur(H, (0, 0), sv)
        t = cv2.copyMakeBorder(im, 24, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(t, f"sigma={sv}", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .5, 255, 1, cv2.LINE_AA)
        tiles.append(t)
    cv2.imwrite("fig2_strip.png", np.hstack(tiles))
    print("wrote fig1_baseline.png, fig2_strip.png")


if __name__ == "__main__":
    main()
