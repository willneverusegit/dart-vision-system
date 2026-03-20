---
name: dart-vision-pipeline
description: Guidance for image processing pipeline (ROI, grayscale, diff image, Canny, contour detection, triangulation)
user_invocable: true
---

# Dart Vision Pipeline Skill

## Wann verwenden
Bei Arbeit an Bildverarbeitung, Pfeilerkennung, Processing-Worker, Shared-Memory IPC.

## Pipeline-Stufen

```
Frame → ROI-Crop → Graustufen → GaussianBlur → Differenzbild → Canny → Konturen → Tip-Extraktion → Score
```

### 1. ROI-Crop
```python
cropped = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
```

### 2. Vorverarbeitung
```python
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Optional: Helligkeitsnormalisierung
normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
```

### 3. Bewegungsdetektion (Differenzbild)
```python
# Hintergrundbild (leeres Board) speichern
background = gray_empty_board.copy()
# Differenz berechnen
diff = cv2.absdiff(background, current_gray)
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
# Signifikante Änderung = Pfeil eingetroffen
change_ratio = np.count_nonzero(thresh) / thresh.size
if change_ratio > 0.005:  # Schwellwert
    # Warten bis Board ruhig (Vibration abklingen lassen)
    time.sleep(0.3)
    # Einschlagframe einfrieren
    impact_frame = capture_frame()
```

### 4. Kanten- und Konturerkennung
```python
edges = cv2.Canny(blurred, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Pfeil-Konturen filtern (Aspektverhältnis, Mindestgröße)
dart_contours = [c for c in contours if is_dart_contour(c)]
# Tip = unterster/vorderster Punkt der Kontur
tip = extract_tip(dart_contours[0])
```

### 5. Triangulation (Multi-Cam)
```python
# 2D-Punkte aus beiden Kameras
pts_left = np.array([[tip_left.x, tip_left.y]], dtype=np.float64).T
pts_right = np.array([[tip_right.x, tip_right.y]], dtype=np.float64).T
# Projektionsmatrizen
P1 = camera_matrix_left @ np.hstack([np.eye(3), np.zeros((3,1))])
P2 = camera_matrix_right @ np.hstack([R, T])
# Triangulieren
points_4d = cv2.triangulatePoints(P1, P2, pts_left, pts_right)
point_3d = (points_4d[:3] / points_4d[3]).flatten()
```

## Shared-Memory IPC
```python
from multiprocessing import shared_memory
# Capture-Prozess schreibt Frame in Shared Memory
shm = shared_memory.SharedMemory(create=True, size=frame.nbytes, name="cam_frame")
buf = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
np.copyto(buf, frame)
# Processing-Worker liest
shm_read = shared_memory.SharedMemory(name="cam_frame")
frame = np.ndarray(shape, dtype=np.uint8, buffer=shm_read.buf)
```

## Modi-Anpassung
- **Eco**: Nur jedes 3. Frame verarbeiten, kein Canny, nur Differenzbild
- **Normal**: Volle Pipeline
- **Debug**: Alle Zwischenstufen als Thumbnails ausgeben (gray, diff, edges, contours)

## Dateien
- `backend/vision/pipeline.py` — Hauptpipeline
- `backend/vision/detection.py` — Pfeil-/Tip-Erkennung
- `backend/vision/camera.py` — Capture + Shared Memory
- `backend/tests/test_pipeline.py`

## Performance-Ziele
- 30 FPS bei 640x480 mit 1-2 Kameras auf < 8 GB RAM
- 20 FPS bei 3 Kameras (Eco-Modus automatisch)
