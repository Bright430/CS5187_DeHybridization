"""Stage-2 figure: band identity + controls, and the recovery ceiling.

Run:  python figures_stage2.py
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

import arcface
import common

SIGMA = 6.0


def band_metrics():
    E = [cv2.imread(f"data/truth/{k:02d}_E.png", 0).astype(np.float32) / 255 for k in range(30)]
    M = [cv2.imread(f"data/truth/{k:02d}_M.png", 0).astype(np.float32) / 255 for k in range(30)]
    eE, eM = arcface.embed_many(E), arcface.embed_many(M)

    rows = []
    for comp in ("LP", "HP", "H"):
        imgs, Ws = [], []
        for k in range(30):
            H = cv2.imread(f"data/hybrid/{k:02d}_s6.png", 0).astype(np.float32) / 255
            LP, HP = common.band_split(H, SIGMA)
            imgs.append({"LP": LP, "HP": np.clip(HP + 0.5, 0, 1), "H": H}[comp])
            Ws.append(arcface.warp_matrix(H))
        e = arcface.embed_many(imgs, Ws)
        ce, cm = np.sum(e * eE, axis=1), np.sum(e * eM, axis=1)
        tgt, oth = (cm, ce) if comp == "LP" else (ce, cm)
        for k in range(30):
            rows.append({"comp": comp, "pair": k, "target": tgt[k], "other": oth[k]})
    df = pd.DataFrame(rows)
    df.to_csv("results_bands.csv", index=False)
    return df


def main():
    df = band_metrics()
    ctrl = pd.read_csv("results_controls.csv")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), dpi=160)

    labels = ["low band\n(-> M)", "high band\n(-> E)", "composite"]
    tgt = [df[df.comp == c]["target"].mean() for c in ("LP", "HP", "H")]
    oth = [df[df.comp == c]["other"].mean() for c in ("LP", "HP", "H")]
    x = np.arange(3)
    ax[0].bar(x - .18, tgt, .36, label="correct identity", color="#2b6cb0")
    ax[0].bar(x + .18, oth, .36, label="the other person", color="#e2e8f0", edgecolor="#94a3b8")
    ax[0].set_xticks(x, labels, fontsize=8)
    ax[0].set_ylabel("identity cosine")
    ax[0].set_title("Identity survives in each band  (sigma=6)")
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].grid(axis="y", alpha=.25)
    for i, v in enumerate(tgt):
        ax[0].text(i - .18, v + .02, f"{v:.2f}", ha="center", fontsize=7)

    order = ["raw", "phase_random", "inverted"]
    names = ["raw high band", "phase randomised", "face inverted"]
    m = [ctrl[ctrl.variant == v]["cos_E"].mean() for v in order]
    s = [ctrl[ctrl.variant == v]["cos_E"].std() for v in order]
    ax[1].bar(np.arange(3), m, yerr=s, capsize=4,
              color=["#2b6cb0", "#dd6b20", "#dd6b20"])
    ax[1].axhline(0.010, ls="--", lw=.9, color="gray")
    ax[1].text(2.45, 0.03, "chance", fontsize=7, color="gray", ha="right")
    ax[1].set_xticks(np.arange(3), names, fontsize=8)
    ax[1].set_ylabel("cosine to the true identity")
    ax[1].set_title("Controls: the signal is structural")
    ax[1].grid(axis="y", alpha=.25)

    fig.tight_layout()
    fig.savefig("fig3_stage2.png")

    print(df.groupby("comp")[["target", "other"]].mean().round(3).to_string())
    print("\naccuracy:", {c: round(float((g.target > g.other).mean()), 3)
                          for c, g in df.groupby("comp")})


if __name__ == "__main__":
    main()
