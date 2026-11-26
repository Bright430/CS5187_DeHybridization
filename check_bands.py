"""Minimal feasibility check for the de-hybridization project.

Question: after a linear band split of a hybrid image, how much identity is left
in each component?  This is the baseline the whole project is measured against.

Run:  python check_bands.py
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arcface
import hybrid_demo as hd


def main():
    sigma = 6.0
    E = hd.align_face("einstein.jpg")
    M = hd.align_face("monroe2.jpg")
    H = hd.make_hybrid(M, E, sigma)           # LP(M) + HP(E)

    LP = cv2.GaussianBlur(H, (0, 0), sigma)   # ~ blurred Monroe
    HP = H - LP                               # ~ edges of Einstein

    emb = {"E": arcface.embed(E), "M": arcface.embed(M)}
    emb["LP"] = arcface.embed(np.clip(LP, 0, 1))
    emb["HP+0.5"] = arcface.embed(np.clip(HP + 0.5, 0, 1))
    emb["H"] = arcface.embed(H)

    print("component   cos(E)   cos(M)   margin(M-E)")
    for k in ("H", "LP", "HP+0.5"):
        ce, cm = arcface.cos(emb[k], emb["E"]), arcface.cos(emb[k], emb["M"])
        print(f"{k:10s} {ce:8.3f} {cm:8.3f} {cm - ce:10.3f}")

    # sanity: the two source identities must be clearly different
    print("\nsource E vs M cosine:", round(arcface.cos(emb["E"], emb["M"]), 3))

    cv2.imwrite("chk_LP.png", (np.clip(LP, 0, 1) * 255).astype(np.uint8))
    cv2.imwrite("chk_HP.png", (np.clip(HP + 0.5, 0, 1) * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
