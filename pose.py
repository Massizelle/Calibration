import glob
import os
import time

import numpy as np
import cv2 as cv

def read_native(fname):
    img = cv.imread(fname, cv.IMREAD_COLOR | cv.IMREAD_IGNORE_ORIENTATION)
    if img is not None and img.shape[0] > img.shape[1]:
        print(f"[ROTATION] {fname} : image en portrait remise en paysage")
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    return img

IMAGES_GLOB = "images/*.jpg"
OUT_DIR = "resultats"

data = np.load(os.path.join(OUT_DIR, "calibration.npz"))
K, dist = data["mtx"], data["dist"]
S = float(data["square_size_mm"])
PATTERN_SIZE = tuple(int(v) for v in data["pattern_size"])

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * S

L = 3 * S
axis = np.float32([[L, 0, 0], [0, L, 0], [0, 0, -L]])
cube = np.float32([[0, 0, 0], [0, L, 0], [L, L, 0], [L, 0, 0],
                   [0, 0, -L], [0, L, -L], [L, L, -L], [L, 0, -L]])

def pt(p):
    return tuple(int(round(v)) for v in p.ravel())

def draw_axes(img, origin, pts):
    o = pt(origin)
    img = cv.line(img, o, pt(pts[0]), (255, 0, 0), 5)
    img = cv.line(img, o, pt(pts[1]), (0, 255, 0), 5)
    img = cv.line(img, o, pt(pts[2]), (0, 0, 255), 5)
    return img

def draw_cube(img, pts):
    pts = np.int32(pts).reshape(-1, 2)
    img = cv.drawContours(img, [pts[:4]], -1, (0, 255, 0), -3)
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(pts[i]), tuple(pts[j]), (255, 0, 0), 3)
    img = cv.drawContours(img, [pts[4:]], -1, (0, 0, 255), 3)
    return img

METHODS = {name: getattr(cv, flag) for name, flag in [
    ("ITERATIVE (Levenberg-Marquardt)", "SOLVEPNP_ITERATIVE"),
    ("EPNP (linéaire, n points)", "SOLVEPNP_EPNP"),
    ("SQPNP", "SOLVEPNP_SQPNP"),
    ("IPPE (mire plane)", "SOLVEPNP_IPPE"),
] if hasattr(cv, flag)}

stats = {m: [] for m in METHODS}

for fname in sorted(glob.glob(IMAGES_GLOB)):
    img = read_native(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, PATTERN_SIZE, None)
    if not ret:
        continue
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    ok, rvec, tvec, inliers = cv.solvePnPRansac(objp, corners2, K, dist)
    if not ok:
        continue

    R, _ = cv.Rodrigues(rvec)
    C = -R.T @ tvec
    print(f"{os.path.basename(fname):30s} distance caméra-mire = {np.linalg.norm(tvec):7.1f} mm"
          f"   C = {np.round(C.ravel(), 1)}")

    imgpts, _ = cv.projectPoints(axis, rvec, tvec, K, dist)
    cv.imwrite(os.path.join(OUT_DIR, "axes_" + os.path.basename(fname)),
               draw_axes(img.copy(), corners2[0], imgpts))

    imgpts, _ = cv.projectPoints(cube, rvec, tvec, K, dist)
    cv.imwrite(os.path.join(OUT_DIR, "cube_" + os.path.basename(fname)),
               draw_cube(img.copy(), imgpts))

    for name, flag in METHODS.items():
        t0 = time.perf_counter()
        ok, rv, tv = cv.solvePnP(objp, corners2, K, dist, flags=flag)
        dt = (time.perf_counter() - t0) * 1000
        if not ok:
            continue
        proj, _ = cv.projectPoints(objp, rv, tv, K, dist)

        err = np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners2.reshape(-1, 2)) ** 2, axis=1)))
        stats[name].append((err, dt))

print("\n===== Comparaison des algorithmes PnP (moyenne sur les images) =====")
print(f"{'Méthode':35s} {'RMS reproj. (px)':>17s} {'Temps (ms)':>11s}")
for name, v in stats.items():
    if v:
        a = np.array(v)
        print(f"{name:35s} {a[:, 0].mean():17.4f} {a[:, 1].mean():11.3f}")