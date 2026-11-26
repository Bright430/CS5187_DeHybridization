"""Figures for the report: method pipeline and a qualitative panel.

Run:  python figures_paper.py
"""

import os
import sys

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arcface
import common

SIGMA = 6.0
ACC = "#1f4e79"
GREY = "#8a8a8a"


def pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 2.35), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=ACC, fs=7.4, bold=False):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                       fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color="#111")

    def arrow(x0, y0, x1, y1, style="-|>", ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, lw=1.0, color=GREY, linestyle=ls,
                                    shrinkA=1, shrinkB=1))

    # top row: construction
    box(0.15, 1.95, 1.55, 0.75, "source E\n(high-freq carrier)", fs=6.8)
    box(0.15, 0.35, 1.55, 0.75, "source M\n(low-freq carrier)", fs=6.8)
    box(2.35, 1.15, 1.75, 0.8, "hybrid\nH = G*M + a(E - G*E)", fc="#eef3f8", fs=6.8)
    arrow(1.70, 2.32, 2.35, 1.72)
    arrow(1.70, 0.72, 2.35, 1.38)

    # viewing
    box(4.65, 1.15, 1.45, 0.8, "viewing blur\nG_v (sigma_v)", fs=6.8)
    arrow(4.10, 1.55, 4.65, 1.55)
    box(6.60, 1.15, 1.55, 0.8, "perceived\nidentity", fc="#eef3f8", fs=6.8)
    arrow(6.10, 1.55, 6.60, 1.55)
    ax.text(7.4, 0.95, "E at close range\nM at a distance", ha="center", va="top",
            fontsize=6.2, color="#444", style="italic")

    # analysis row
    box(2.35, 0.05, 1.75, 0.62, "band split\nLP / HP", fs=6.8)
    arrow(3.2, 1.15, 3.2, 0.67, ls=":")
    box(4.65, 0.05, 1.45, 0.62, "recover\neach band", fs=6.8)
    arrow(4.10, 0.36, 4.65, 0.36)
    box(6.60, 0.05, 1.55, 0.62, "identity test\n(ArcFace)", fs=6.8)
    arrow(6.10, 0.36, 6.60, 0.36)
    ax.text(9.0, 1.55, "analysis\n(no human data)", ha="center", va="center",
            fontsize=6.2, color=ACC, style="italic")

    fig.tight_layout(pad=0.1)
    fig.savefig("fig_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def pick_pair():
    d = pd.read_csv("results_distance.csv")
    piv = d.pivot_table(index="pair", columns="view", values="margin")
    if 0.0 in piv.columns and 6.0 in piv.columns:
        score = (piv[6.0] - piv[0.0]).abs()
    else:
        score = piv.iloc[:, -1] - piv.iloc[:, 0]
    return int(score.idxmax())


def qualitative(k):
    E = cv2.imread(f"data/truth/{k:02d}_E.png", 0).astype(np.float32) / 255
    M = cv2.imread(f"data/truth/{k:02d}_M.png", 0).astype(np.float32) / 255
    H = cv2.imread(f"data/hybrid/{k:02d}_s6.png", 0).astype(np.float32) / 255
    LP, HP = common.band_split(H, SIGMA)

    views = [0, 2, 4, 8, 16]
    imgs = [E, M, H, LP, np.clip(HP + 0.5, 0, 1)] + [
        cv2.GaussianBlur(H, (0, 0), s) if s else H for s in views]

    titles = ["E (source)", "M (source)", "H (hybrid)", "LP of H", "HP of H"] + \
             [f"H, sigma_v={s}" if s else "H, sharp" for s in views]

    fig, axes = plt.subplots(2, 5, figsize=(7.0, 2.85), dpi=200)
    for ax, im, t in zip(axes.ravel(), imgs, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=6.4, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#bbb")
    fig.tight_layout(pad=0.15, h_pad=0.4)
    fig.savefig("fig_qual.png", bbox_inches="tight")
    plt.close(fig)
    return k


def recovery_fig():
    d = pd.read_csv("results_recover_final.csv")
    order = ["low_raw", "low_pca", "low_ceiling", "high_raw", "high_pca", "high_ceiling"]
    names = ["raw LP", "PCA invert", "ceiling", "raw HP", "PCA invert", "ceiling"]
    m = [d[d.method == k].target.mean() for k in order]
    s = [d[d.method == k].target.std() for k in order]
    colors = ["#9dbfd9", "#c9d9e6", "#e8eef4", "#9dbfd9", "#c9d9e6", "#e8eef4"]

    fig, ax = plt.subplots(figsize=(3.5, 2.3), dpi=200)
    x = np.arange(6)
    ax.bar(x, m, yerr=s, capsize=3, color=colors, edgecolor="#5a7a94", lw=.7, width=.68)
    ax.set_xticks(x, names, fontsize=6.2, rotation=20, ha="right")
    ax.set_ylabel("cosine to the true identity", fontsize=7)
    ax.set_ylim(0, .72)
    ax.grid(axis="y", alpha=.25)
    ax.tick_params(labelsize=6.4)
    ax.axvline(2.5, color="#ccc", lw=.8)
    ax.text(1, .66, "low band  (-> M)", fontsize=6.4, ha="center", color=ACC)
    ax.text(4, .66, "high band  (-> E)", fontsize=6.4, ha="center", color=ACC)
    for i, v in enumerate(m):
        ax.text(i, v + s[i] + .015, f"{v:.2f}", ha="center", fontsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig("fig_recover.png", bbox_inches="tight")
    plt.close(fig)


def blur_fig():
    """Single-column figure: identity cosine vs viewing blur."""
    d = pd.read_csv("results_distance.csv")
    g = d.groupby("view")[["cos_E", "cos_M"]]
    mean, sem = g.mean(), g.sem()
    views = list(mean.index)
    share = d.assign(m=d["margin"] > 0).groupby("view")["m"].mean()

    fig, ax = plt.subplots(figsize=(3.45, 2.5), dpi=220)
    ax.axvspan(0, 2.2, color="#f2f5f8", zorder=0)
    ax.errorbar(views, mean["cos_E"], yerr=sem["cos_E"], marker="o", ms=3.2, lw=1.1,
                capsize=2, label="cos to E (high-freq source)", color=ACC)
    ax.errorbar(views, mean["cos_M"], yerr=sem["cos_M"], marker="s", ms=3.2, lw=1.1,
                capsize=2, label="cos to M (low-freq source)", color="#a33a2a")
    ax.axvline(2.19, ls=":", lw=.9, color="#555")
    ax.text(2.35, .60, r"crossover $\sigma_v\approx2.2$", fontsize=5.8, color="#333")
    ax.set_xlabel(r"viewing blur $\sigma_v$ (px at 512)", fontsize=7)
    ax.set_ylabel("identity cosine", fontsize=7)
    ax.tick_params(labelsize=6.4)
    ax.set_xlim(-0.6, 16.6)
    ax.set_ylim(0, .68)
    ax.grid(alpha=.2)
    ax.legend(fontsize=5.9, frameon=False, loc="center right")
    ax.set_title("perceived identity flips with distance", fontsize=7.4)
    fig.tight_layout(pad=0.2)
    fig.savefig("fig_blur.png", bbox_inches="tight")
    plt.close(fig)


def band_fig():
    """Full-width figure: band identity (left) and controls (right)."""
    b = pd.read_csv("results_bands.csv")
    c = pd.read_csv("results_controls.csv")

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.45), dpi=220,
                           gridspec_kw={"width_ratios": [1.05, 1]})

    comps = ["LP", "HP", "H"]
    names = ["LP\n($\\to$ M)", "HP\n($\\to$ E)", "H\n(composite)"]
    tgt = [b[b.comp == k].target.mean() for k in comps]
    oth = [b[b.comp == k].other.mean() for k in comps]
    x = np.arange(3)
    ax[0].bar(x - .18, tgt, .34, color=ACC, label="correct person")
    ax[0].bar(x + .18, oth, .34, color="#dce3ea", edgecolor="#8fa3b5", lw=.6,
              label="the other person")
    ax[0].set_xticks(x, names, fontsize=6.6)
    ax[0].set_ylabel("identity cosine", fontsize=7)
    ax[0].set_title("identity survives in either band", fontsize=7.4)
    ax[0].legend(fontsize=6, frameon=False)
    ax[0].grid(axis="y", alpha=.2)
    ax[0].set_ylim(0, .68)
    for i, v in enumerate(tgt):
        ax[0].text(i - .18, v + .015, f"{v:.2f}", ha="center", fontsize=5.9)

    order = ["raw", "phase_random", "inverted"]
    lbl = ["raw HP", "phase\nrandomised", "inverted\nface"]
    m = [c[c.variant == v].cos_E.mean() for v in order]
    s = [c[c.variant == v].cos_E.std() for v in order]
    ax[1].bar(np.arange(3), m, yerr=s, capsize=3, width=.6,
              color=[ACC, "#c8794b", "#c8794b"], edgecolor="#8a6a52", lw=.5)
    ax[1].axhline(0.010, ls="--", lw=.8, color="#777")
    ax[1].text(2.42, .035, "chance", fontsize=5.9, color="#666", ha="right")
    ax[1].set_xticks(np.arange(3), lbl, fontsize=6.6)
    ax[1].set_ylabel("cosine to the true identity", fontsize=7)
    ax[1].set_title("controls: the cue is structural", fontsize=7.4)
    ax[1].grid(axis="y", alpha=.2)
    ax[1].set_ylim(-.06, .68)
    ax[1].tick_params(labelsize=6.4)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig("fig_band.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    k = pick_pair()
    print("qualitative pair:", k)
    qualitative(k)
    recovery_fig()
    blur_fig()
    band_fig()
    print("wrote fig_pipeline.png, fig_qual.png, fig_recover.png, fig_blur.png, fig_band.png")
