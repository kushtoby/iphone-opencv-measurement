import glob
import cv2
import numpy as np

# --- YOUR BOARD ---
pattern_size = (9, 6)     # inner corners (cols, rows)
square_size_mm = 23.0     # measured on your print

images = sorted(glob.glob("calib_images/*.jpg") + glob.glob("calib_images/*.jpeg"))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size_mm

objpoints, imgpoints = [], []
img_size = None

print("Found images:", len(images))

for fn in images:
    img = cv2.imread(fn)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]  # (W,H)

    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        print("NO corners:", fn)
        continue

    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    objpoints.append(objp)
    imgpoints.append(corners2)
    print("OK corners:", fn)

print("Usable images:", len(objpoints))
if len(objpoints) < 10:
    raise RuntimeError("Too few usable images. Take more photos with varied tilt/position.")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("\n=== RESULTS ===")
print("Image size (W,H):", img_size)
print("RMS reprojection error:", ret)
print("K:\n", K)
print("dist (k1,k2,p1,p2,k3):", dist.ravel())