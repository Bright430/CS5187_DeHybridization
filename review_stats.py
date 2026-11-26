"""Reproduce the statistics requested by the reviewer for items 1 and 2.

Reads only the existing result CSVs (no model inference), so it runs in seconds.
Run:  python review_stats.py
Seed is fixed so every number is reproducible.

Outputs
  1) bootstrap 95% CIs for the crossover proportion and crossover blur
  2) paired sign / Wilcoxon tests on the own-minus-other cosine of Table II
"""

import csv

import numpy as np
from scipy import stats

SEED = 20261125
B = 20000
VIEWS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0])


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def crossover_bootstrap():
    rows = load("results_distance.csv")
    m = np.full((30, len(VIEWS)), np.nan)
    for r in rows:
        m[int(r["pair"]), np.where(VIEWS == float(r["view"]))[0][0]] = (
            float(r["cos_M"]) - float(r["cos_E"])
        )

    def stats_once(idx):
        x = m[idx]
        cross = []
        for i in range(len(idx)):
            for j in range(len(VIEWS) - 1):
                if x[i, j] <= 0 < x[i, j + 1]:
                    t = (0 - x[i, j]) / (x[i, j + 1] - x[i, j])
                    cross.append(VIEWS[j] + t * (VIEWS[j + 1] - VIEWS[j]))
                    break
        return (
            len(cross) / len(idx),
            float(np.mean(cross)) if cross else np.nan,
            float(np.median(cross)) if cross else np.nan,
        )

    point = stats_once(np.arange(30))
    rng = np.random.default_rng(SEED)
    boot = np.array([stats_once(rng.integers(0, 30, 30)) for _ in range(B)])
    print(f"[crossover] point: {point[0]:.1%}, mean {point[1]:.3f} px, median {point[2]:.3f} px")
    for j, name in enumerate(("proportion", "mean crossover", "median crossover")):
        lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
        print(f"[crossover] {name}: 95% CI [{lo:.3f}, {hi:.3f}]")


def paired_tests():
    rows = load("results_bands.csv")
    d = {}
    for r in rows:
        d.setdefault(r["comp"], []).append(float(r["target"]) - float(r["other"]))
    for comp, x in d.items():
        x = np.asarray(x)
        w = stats.wilcoxon(x).pvalue
        s = stats.binomtest(int((x > 0).sum()), len(x), 0.5).pvalue
        print(
            f"[{comp}] own-other mean {x.mean():.4f}, wins {int((x > 0).sum())}/{len(x)}, "
            f"Wilcoxon p={w:.3g}, sign p={s:.3g}"
        )


if __name__ == "__main__":
    crossover_bootstrap()
    paired_tests()
