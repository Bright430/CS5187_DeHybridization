"""Hybrid image demo: Monroe/Einstein, both frequency orders.

Classic "Marilyn-Einstein" hybrid (Oliva, Torralba & Schyns, SIGGRAPH 2006):
    H = LP_sigma(Marilyn) + HP_sigma(Einstein)
Close up  -> sharp edges belong to Einstein  -> you see Einstein.
Far away  -> only LP survives               -> you see Marilyn.

The wording in the course handout ("LP on Einstein, HP on Marilyn") is the
opposite order; the script also builds that version so the two can be compared.

"Viewing distance" is simulated by blurring H with increasing sigma: a larger
sigma means fewer cycles per degree on the retina, i.e. standing further away.
"""

import cv2
import numpy as np

OUT = 512
SIGMA = 6.0
FACE_CROP = 1.75     # crop side = FACE_CROP * face height
FACE_LIFT = 0.08     # move the crop up a bit so the forehead is not cut
EYE_Y = 0.42         # target y of the eye line in the aligned image


def align_face(path, size=OUT):
    """Crop a square around the detected face and rescale it to size x size."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    detector = cv2.FaceDetectorYN.create("yunet.onnx", "", (320, 320), 0.7, 0.3, 5000)
    detector.setInputSize((w, h))
    _, faces = detector.detect(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    if faces is None or len(faces) == 0:
        raise RuntimeError("no face found in " + path)
    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])[:4]
    side = min(FACE_CROP * fh, 1.6 * w, 1.6 * h)
    s = size / side
    M = np.float32([[s, 0, -(x + fw / 2) * s + size / 2],
                    [0, s, -(y + fh / 2 - FACE_LIFT * fh) * s + size / 2]])
    out = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    # second pass: translate so that the eye midpoint hits the template
    detector.setInputSize((size, size))
    _, f2 = detector.detect(cv2.cvtColor(out, cv2.COLOR_GRAY2BGR))
    if f2 is not None and len(f2):
        r = max(f2, key=lambda q: q[2] * q[3])
        eyes = r[4:8].reshape(2, 2)
        mid = eyes.mean(axis=0)
        dx = size * 0.5 - mid[0]
        dy = size * EYE_Y - mid[1]
        out = cv2.warpAffine(out, np.float32([[1, 0, dx], [0, 1, dy]]), (size, size),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out.astype(np.float32) / 255.0


def lp(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def make_hybrid(low_img, high_img, sigma=SIGMA):
    hp = high_img - lp(high_img, sigma)
    return np.clip(lp(low_img, sigma) + hp, 0, 1)


def report(name, hybrid, low_img, high_img):
    print(f"\n{name}")
    print("blur   corr(low-freq src)   corr(high-freq src)   SSIM(low)  SSIM(high)")
    for s in (0, 4, 8, 12, 16):
        hb = lp(hybrid, s) if s else hybrid
        lo = lp(low_img, s) if s else low_img
        hi = lp(high_img, s) if s else high_img
        print(f"{s:4.0f}   {np.corrcoef(hb.ravel(), lo.ravel())[0,1]:18.4f}   "
              f"{np.corrcoef(hb.ravel(), hi.ravel())[0,1]:20.4f}   "
              f"{ssim(hb, lo):8.4f}   {ssim(hb, hi):9.4f}")


def ssim(a, b):
    """Global SSIM (no windowing) with the standard constants."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    C1, C2 = (0.01 * 255) ** 2 / 255 ** 2, (0.03 * 255) ** 2 / 255 ** 2
    return ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / \
           ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))


def main():
    einstein = align_face("einstein.jpg")
    monroe = align_face("monroe2.jpg")

    classic = make_hybrid(monroe, einstein)      # LP Monroe + HP Einstein
    swapped = make_hybrid(einstein, monroe)      # what the handout literally says

    cv2.imwrite("out_hybrid.jpg", (classic * 255).astype(np.uint8))
    cv2.imwrite("out_hybrid_swapped.jpg", (swapped * 255).astype(np.uint8))
    cv2.imwrite("out_einstein_aligned.jpg", (einstein * 255).astype(np.uint8))
    cv2.imwrite("out_monroe_aligned.jpg", (monroe * 255).astype(np.uint8))

    report("CLASSIC: LP Monroe + HP Einstein", classic, monroe, einstein)
    report("SWAPPED: LP Einstein + HP Monroe", swapped, einstein, monroe)

    tiles = []
    for name, im in [("Einstein aligned", einstein), ("Monroe aligned", monroe),
                     ("Hybrid (classic)", classic),
                     ("Classic, sigma=8", lp(classic, 8)),
                     ("Classic, sigma=12", lp(classic, 12)),
                     ("Classic, sigma=16", lp(classic, 16)),
                     ("Hybrid (swapped)", swapped),
                     ("Swapped, sigma=12", lp(swapped, 12)),
                     ("Swapped, sigma=16", lp(swapped, 16))]:
        t = cv2.copyMakeBorder((im * 255).astype(np.uint8), 28, 4, 4, 4,
                               cv2.BORDER_CONSTANT, value=0)
        cv2.putText(t, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
        tiles.append(t)
    cv2.imwrite("out_contact_sheet.png", np.vstack([np.hstack(tiles[i:i + 3])
                                                    for i in range(0, 9, 3)]))


if __name__ == "__main__":
    main()
