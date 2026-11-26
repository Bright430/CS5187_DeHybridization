"""Pass 2: choose 30 pairs of *different* people with a matching face layout.

Two constraints:
1. identity cosine < 0.10  (they must be different people)
2. small geometry distance  (same scale / roll / feature layout, otherwise the
   hybrid image shows obvious ghosting)

Run:  python select_pairs.py
"""

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

META = "data/meta.npz"
N_PAIRS = 30
W = np.array([2.0, 4.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # geometry weights


def main():
    m = np.load(META)
    idx, emb, geo, valid = m["idx"], m["emb"], m["geo"], m["valid"]
    keep = np.where(valid)[0]
    print(f"{len(keep)} usable faces")

    ids = [int(i) for i in keep]
    E, G = emb[keep], geo[keep]
    sim = E @ E.T
    dg = np.sqrt((((G[:, None, :] - G[None, :, :]) * W) ** 2).sum(-1))

    cands = []
    for a, b in itertools.combinations(range(len(ids)), 2):
        if sim[a, b] < 0.10:
            cands.append((float(dg[a, b]), a, b))
    cands.sort()
    print(f"{len(cands)} candidate pairs with cos < 0.10")

    used, pairs = set(), []
    thr = 0.35
    while len(pairs) < N_PAIRS and thr < 2.0:
        for d, a, b in cands:
            if d > thr:
                break
            if a in used or b in used:
                continue
            used.update((a, b))
            pairs.append({"a": ids[a], "b": ids[b], "geom": round(d, 3),
                          "cos": round(float(sim[a, b]), 3)})
            if len(pairs) == N_PAIRS:
                break
        if len(pairs) < N_PAIRS:
            thr += 0.15

    print(f"selected {len(pairs)} pairs (geometry threshold {thr:.2f})")
    print("geometry distance: mean %.3f  max %.3f" %
          (np.mean([p["geom"] for p in pairs]), np.max([p["geom"] for p in pairs])))
    print("identity cosine  : mean %.3f  max %.3f" %
          (np.mean([p["cos"] for p in pairs]), np.max([p["cos"] for p in pairs])))

    os.makedirs("data", exist_ok=True)
    with open("data/pairs.json", "w") as f:
        json.dump(pairs, f, indent=1)


if __name__ == "__main__":
    main()
