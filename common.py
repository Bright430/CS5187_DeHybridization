"""Shared utilities for the de-hybridization project.

All faces are aligned to 512x512 with the eye midpoint at (0.5, 0.42).
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = 512
FACE_CROP = 1.75
FACE_LIFT = 0.08
EYE_Y = 0.42

_det = None


def detector():
    global _det
    if _det is None:
        _det = cv2.FaceDetectorYN.create("yunet.onnx", "", (320, 320), 0.7, 0.3, 5000)
    return _det


def detect(img):
    """Return the largest face row from YuNet, or None."""
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img
    h, w = bgr.shape[:2]
    d = detector()
    d.setInputSize((w, h))
    _, faces = d.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


def align(img, size=OUT):
    """Crop around the detected face, rescale, then centre the eye midpoint.

    Returns float32 [0,1] image, or None if no face was found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    f = detect(gray)
    if f is None:
        return None
    x, y, fw, fh = f[:4]
    h, w = gray.shape
    side = min(FACE_CROP * fh, 1.6 * w, 1.6 * h)
    s = size / side
    M = np.float32([[s, 0, -(x + fw / 2) * s + size / 2],
                    [0, s, -(y + fh / 2 - FACE_LIFT * fh) * s + size / 2]])
    out = cv2.warpAffine(gray, M, (size, size), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    f2 = detect(out)
    if f2 is not None:
        mid = f2[4:8].reshape(2, 2).mean(axis=0)
        dx, dy = size * 0.5 - mid[0], size * EYE_Y - mid[1]
        out = cv2.warpAffine(out, np.float32([[1, 0, dx], [0, 1, dy]]), (size, size),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out.astype(np.float32) / 255.0


def geometry(img):
    """Pose descriptor used to pair two faces with matching layout.

    All lengths are normalised by the eye distance, so the descriptor is
    invariant to face size.
    """
    f = detect((np.clip(img, 0, 1) * 255).astype(np.uint8))
    if f is None:
        return None
    lm = f[4:14].reshape(5, 2).astype(np.float32)
    eyes = lm[:2][np.argsort(lm[:2, 0])]
    mouth = lm[3:5][np.argsort(lm[3:5, 0])]
    d = np.linalg.norm(eyes[1] - eyes[0]) + 1e-6
    mid = eyes.mean(axis=0)
    ang = np.arctan2(eyes[1][1] - eyes[0][1], eyes[1][0] - eyes[0][0])
    return np.array([
        d / OUT,                                  # apparent face scale
        ang,                                      # head roll
        (lm[2][0] - mid[0]) / d, (lm[2][1] - mid[1]) / d,      # nose
        (mouth.mean(axis=0)[0] - mid[0]) / d,
        (mouth.mean(axis=0)[1] - mid[1]) / d,
    ], dtype=np.float32)


def hybrid(E, M, sigma=6.0, alpha=1.0):
    """LP(M) + alpha * HP(E)."""
    lp_m = cv2.GaussianBlur(M, (0, 0), sigma)
    hp_e = E - cv2.GaussianBlur(E, (0, 0), sigma)
    return np.clip(lp_m + alpha * hp_e, 0, 1)


def band_split(H, sigma=6.0):
    lp = cv2.GaussianBlur(H, (0, 0), sigma)
    return lp, H - lp


def sigma_to_cpd(sigma, size=OUT, px_per_deg=30.0):
    """Half-amplitude cutoff of a Gaussian in cycles/degree.

    f_c = sqrt(ln2) / (sqrt(2) * pi * sigma) cycles/pixel, then scaled by the
    display resolution. px_per_deg must be fixed by the experiment geometry
    (e.g. a 15 cm wide image seen at 50 cm is about 30 px/deg at 512 px).
    """
    f_px = np.sqrt(np.log(2)) / (np.sqrt(2) * np.pi * sigma)
    return f_px * px_per_deg


def to_u8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)
