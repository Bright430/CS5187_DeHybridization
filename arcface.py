"""ArcFace (buffalo_l w600k_r50) wrapper: aligned face image -> 512-d embedding.

We keep our own alignment (YuNet landmarks -> ArcFace 112x112 template), so
insightface itself is not required -- only onnxruntime.
"""

import cv2
import numpy as np
import onnxruntime as ort

MODEL = "models/w600k_r50.onnx"
# ArcFace 5-point template in image coordinates (left-to-right)
TPL = np.float32([[38.29, 51.69], [73.53, 51.50], [56.02, 71.73],
                  [41.55, 92.37], [70.73, 92.20]])

_sess = None
_det = None


def _session():
    global _sess
    if _sess is None:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    return _sess


def _detector(img_shape):
    global _det
    if _det is None:
        _det = cv2.FaceDetectorYN.create("yunet.onnx", "", (320, 320), 0.6, 0.3, 5000)
    h, w = img_shape
    _det.setInputSize((w, h))
    return _det


def landmarks(img):
    """Return the 5 YuNet landmarks (image-left to image-right for the pairs)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    h, w = rgb.shape[:2]
    _, faces = _detector((h, w)).detect(rgb)
    if faces is None or len(faces) == 0:
        raise RuntimeError("no face")
    f = max(faces, key=lambda r: r[2] * r[3])
    lm = f[4:14].reshape(5, 2).astype(np.float32)
    eyes = lm[:2][np.argsort(lm[:2, 0])]
    mouth = lm[3:5][np.argsort(lm[3:5, 0])]
    return np.stack([eyes[0], eyes[1], lm[2], mouth[0], mouth[1]])


def to_112(img):
    M = cv2.estimateAffinePartial2D(landmarks(img), TPL, method=cv2.LMEDS)[0]
    return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def warp_matrix(img):
    """Similarity transform from an image to the 112x112 ArcFace template.

    Compute it once on the composite (or ground truth) and reuse it for every
    band-decomposed version: blurred or edge-only images are not detectable as
    faces, but they share the geometry of the image they came from.
    """
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.estimateAffinePartial2D(landmarks(img), TPL, method=cv2.LMEDS)[0]


def embed_with(img, M, flip=True):
    """Embed a face using a precomputed warp matrix (see warp_matrix)."""
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crop = cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    return _forward(crop, flip)


def _forward(crop, flip=True):
    rgb = np.repeat(crop[:, :, None], 3, axis=2)
    blob = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None]
    sess = _session()
    name = sess.get_inputs()[0].name
    if flip:
        e = sess.run(None, {name: blob})[0][0]
        e2 = sess.run(None, {name: blob[:, :, :, ::-1].copy()})[0][0]
        e = e + e2
    else:
        e = sess.run(None, {name: blob})[0][0]
    return e / np.linalg.norm(e)


def embed_many(images, Ws=None, flip=True, batch=16):
    """Embed a list of images, optionally with per-image warp matrices.

    Batching matters: on CPU this is roughly 5x faster than one call per image.
    """
    crops = []
    for i, img in enumerate(images):
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        M = Ws[i] if Ws is not None else warp_matrix(img)
        crops.append(cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE).astype(np.float32))

    sess = _session()
    name = sess.get_inputs()[0].name
    out = []
    for s in range(0, len(crops), batch):
        chunk = np.stack(crops[s:s + batch])
        rgb = np.repeat(chunk[:, :, :, None], 3, axis=3)
        blob = ((rgb - 127.5) / 127.5).transpose(0, 3, 1, 2)
        e = sess.run(None, {name: blob})[0]
        if flip:
            e = e + sess.run(None, {name: blob[:, :, :, ::-1].copy()})[0]
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        out.append(e)
    return np.concatenate(out, axis=0)


def embed(img, flip=True):
    """L2-normalised embedding of a face image (gray or BGR, any size, [0,1] or uint8)."""
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crop = to_112(img).astype(np.float32)
    return _forward(crop, flip)


def cos(a, b):
    return float(np.dot(a, b))
