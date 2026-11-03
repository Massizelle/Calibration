import glob
import json
import os

import numpy as np
import cv2 as cv

def read_native(fname):
    img = cv.imread(fname, cv.IMREAD_COLOR | cv.IMREAD_IGNORE_ORIENTATION)
    if img is not None and img.shape[0] > img.shape[1]:
        print(f"[ROTATION] {fname} : image en portrait remise en paysage")
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    return img

IMAGES_GLOB = "images/*.jpg"
PATTERN_SIZE = (8, 5)
SQUARE_SIZE_MM = 28.5
SENSOR_W_MM, SENSOR_H_MM = 6.854, 5.141
OUT_DIR = "resultats"

os.makedirs(OUT_DIR, exist_ok=True)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

objpoints, imgpoints, used = [], [], []
img_size = None

for fname in sorted(glob.glob(IMAGES_GLOB)):
    img = read_native(fname)
    if img is None:
        print(f"[ILLISIBLE] {fname}")
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = gray.shape[::-1]
    elif gray.shape[::-1] != img_size:
        print(f"[IGNORÉE]  {fname} : résolution différente des autres")
        continue

    ret, corners = cv.findChessboardCorners(gray, PATTERN_SIZE, None)
    if not ret:
        print(f"[ÉCHEC]    {fname} : damier non détecté")
        continue

    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    objpoints.append(objp)
    imgpoints.append(corners2)
    used.append(fname)

    vis = cv.drawChessboardCorners(img.copy(), PATTERN_SIZE, corners2, ret)
    cv.imwrite(os.path.join(OUT_DIR, "coins_" + os.path.basename(fname)), vis)
    print(f"[OK]       {fname}")

if len(used) < 10:
    print(f"\nAttention : seulement {len(used)} images valides. Vise au moins 15.")
if not used:
    raise SystemExit("Aucune image exploitable.")

(rms, K, dist, rvecs, tvecs,
 std_int, std_ext, per_view) = cv.calibrateCameraExtended(objpoints, imgpoints, img_size, None, None)

std_int = std_int.ravel()
per_view = per_view.ravel()
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

fovx, fovy, focal_mm, principal_pt_mm, aspect = cv.calibrationMatrixValues(
    K, img_size, SENSOR_W_MM, SENSOR_H_MM)
pixel_w_um = SENSOR_W_MM / img_size[0] * 1000
pixel_h_um = SENSOR_H_MM / img_size[1] * 1000

print("\n================ RÉSULTATS ================")
print(f"Images utilisées       : {len(used)}")
print(f"Résolution             : {img_size[0]} x {img_size[1]} px")
print(f"Erreur RMS globale     : {rms:.4f} px")
print("Matrice intrinsèque K :\n", np.array2string(K, precision=2, suppress_small=True))
print(f"fx = {fx:.2f} ± {std_int[0]:.2f} px   fy = {fy:.2f} ± {std_int[1]:.2f} px")
print(f"cx = {cx:.2f} ± {std_int[2]:.2f} px   cy = {cy:.2f} ± {std_int[3]:.2f} px")
print("Distorsion (k1 k2 p1 p2 k3) :", np.round(dist.ravel(), 5))
print(f"Taille pixel (capteur) : {pixel_w_um:.3f} x {pixel_h_um:.3f} µm")
print(f"Champ de vision        : {fovx:.2f}° (horiz.) x {fovy:.2f}° (vert.)")
print(f"Focale                 : {focal_mm:.3f} mm = {focal_mm / 1000:.6f} m")
print(f"Point principal        : ({principal_pt_mm[0]:.3f}, {principal_pt_mm[1]:.3f}) mm")
print(f"Rapport d'aspect fy/fx : {aspect:.4f}")
print("\nErreur RMS par image (px) :")
for f, e in zip(used, per_view):
    flag = "  <-- suspecte" if e > max(2 * np.median(per_view), 1.0) else ""
    print(f"  {os.path.basename(f):30s} {e:.3f}{flag}")

models = [
    ("Sans distorsion", cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 | cv.CALIB_ZERO_TANGENT_DIST),
    ("k1 seul", cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 | cv.CALIB_ZERO_TANGENT_DIST),
    ("k1, k2", cv.CALIB_FIX_K3 | cv.CALIB_ZERO_TANGENT_DIST),
    ("k1, k2, p1, p2", cv.CALIB_FIX_K3),
    ("k1, k2, p1, p2, k3 (défaut)", 0),
]
print("\n===== Comparaison des modèles de distorsion =====")
print(f"{'Modèle':30s} {'RMS (px)':>9s} {'fx (px)':>9s} {'focale (mm)':>12s}  coefficients")
comparaison = []
for name, flags in models:
    r, Km, dm, _, _ = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None, flags=flags)
    f_mm = cv.calibrationMatrixValues(Km, img_size, SENSOR_W_MM, SENSOR_H_MM)[2]
    print(f"{name:30s} {r:9.4f} {Km[0, 0]:9.1f} {f_mm:12.3f}  {np.round(dm.ravel(), 4)}")
    comparaison.append({"modele": name, "rms_px": float(r), "fx_px": float(Km[0, 0]),
                        "focale_mm": float(f_mm), "dist": dm.ravel().tolist()})
    if name == "k1, k2":
        K_k1k2, dist_k1k2 = Km, dm

img0 = read_native(used[0])
h0, w0 = img0.shape[:2]
panels = [img0]
for Km, dm, label in [(K, dist, "k1,k2,p1,p2,k3"), (K_k1k2, dist_k1k2, "k1,k2")]:
    nK, _ = cv.getOptimalNewCameraMatrix(Km, dm, (w0, h0), 1, (w0, h0))
    u = cv.undistort(img0, Km, dm, None, nK)
    cv.putText(u, label, (60, 160), cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 12)
    panels.append(u)
cv.putText(panels[0], "original", (60, 160), cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 12)
cv.imwrite(os.path.join(OUT_DIR, "comparaison_undistort.jpg"), np.hstack(panels))

np.savez(os.path.join(OUT_DIR, "calibration.npz"),
         mtx=K, dist=dist, rvecs=np.array(rvecs), tvecs=np.array(tvecs),
         square_size_mm=SQUARE_SIZE_MM, pattern_size=np.array(PATTERN_SIZE))

with open(os.path.join(OUT_DIR, "calibration.json"), "w", encoding="utf-8") as f:
    json.dump({
        "nb_images": len(used),
        "resolution_px": list(img_size),
        "rms_px": float(rms),
        "K": K.tolist(),
        "std_fx_fy_cx_cy": std_int[:4].tolist(),
        "distorsion_k1_k2_p1_p2_k3": dist.ravel().tolist(),
        "std_distorsion": std_int[4:9].tolist(),
        "capteur_mm": [SENSOR_W_MM, SENSOR_H_MM],
        "taille_pixel_um": [pixel_w_um, pixel_h_um],
        "fov_deg": [fovx, fovy],
        "focale_mm": focal_mm,
        "focale_m": focal_mm / 1000,
        "point_principal_mm": list(principal_pt_mm),
        "aspect_ratio": aspect,
        "rms_par_image": {os.path.basename(f): float(e) for f, e in zip(used, per_view)},
        "comparaison_modeles_distorsion": comparaison,
    }, f, indent=2, ensure_ascii=False)

img = read_native(used[0])
h, w = img.shape[:2]
newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
und = cv.undistort(img, K, dist, None, newK)
cv.imwrite(os.path.join(OUT_DIR, "undistort_" + os.path.basename(used[0])), np.hstack([img, und]))

print(f"\nRésultats enregistrés dans {OUT_DIR}/")