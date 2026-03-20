---
name: dart-calibration
description: Guidance for implementing camera calibration modules (ChArUco, intrinsics, ROI, board-fitting, stereo)
user_invocable: true
---

# Dart Calibration Skill

## Wann verwenden
Bei jeder Arbeit an Kamera-Kalibrierung: Intrinsics, ROI-Auswahl, Board-Fitting, Stereo-Kalibrierung.

## Kalibrierungsablauf

### 1. Intrinsic-Kalibrierung (ChArUco)
```python
import cv2
from cv2 import aruco

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((7, 5), 0.04, 0.02, dictionary)

# Pro Frame: Marker erkennen
corners, ids, rejected = aruco.detectMarkers(gray, dictionary)
if ids is not None:
    retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    all_corners.append(charuco_corners)
    all_ids.append(charuco_ids)

# Nach genug Frames (>15): Kalibrieren
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    all_corners, all_ids, gray.shape[::-1], None, None
)
# Akzeptieren wenn reprojection_error < 0.5
```

### 2. ROI-Auswahl
- User wählt Rechteck (x, y, w, h) auf undistorted Frame
- Zielgröße: ~480x480 px
- Alle folgenden Frames werden auf ROI gecroppt → CPU-Ersparnis

### 3. Board-Fitting (Homographie)
```python
# 4-Punkt oder Ellipsen-Fitting
# Dartboard-Mittelpunkt + 4 Referenzpunkte identifizieren
# Homographie berechnen:
H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
# H speichern im CameraProfile
```

### 4. Fine-Tuning
- Zoom auf Triple-20 und Bullseye
- Manuelles Nachjustieren der Homographie-Offsets
- Validierung: gemessene Drahtabstände vs. Soll-Werte

### 5. Stereo-Kalibrierung
```python
ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_left, img_points_right,
    camera_matrix_left, dist_left, camera_matrix_right, dist_right,
    image_size, flags=cv2.CALIB_FIX_INTRINSIC
)
# CALIB_FIX_INTRINSIC: Intrinsics aus Single-Kalib beibehalten
```

## Datenmodelle
```python
class CameraProfile(BaseModel):
    id: str
    role: Literal["left", "right", "top"]
    resolution: tuple[int, int]
    intrinsics: dict  # fx, fy, cx, cy
    dist_coeffs: list[float]
    roi: dict  # x, y, w, h
    homography: list[list[float]]  # 3x3
    timestamp: datetime

class StereoProfile(BaseModel):
    camera_left_id: str
    camera_right_id: str
    rotation_matrix: list[list[float]]  # 3x3
    translation_vector: list[float]  # 3x1
    reprojection_error: float
```

## Dateien
- `backend/vision/calibration.py` — Kalibrierungslogik
- `backend/vision/camera.py` — Kamera-Capture
- `backend/data/profiles/` — JSON-Profile
- `backend/tests/test_calibration.py`

## Qualitätskriterien
- Reprojection Error < 0.5 px für Intrinsics
- Marker-Coverage > 60% des Frames
- Homographie-Validierung über bekannte Boardmaße
