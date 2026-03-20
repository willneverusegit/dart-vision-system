# Phase 7 — Stereo-Kalibrierung & Triangulation

**Impact**: 4 | **Aufwand**: 5 | **Score**: 0.80
**Abhängigkeit**: Phase 2 (Single-Cam Kalibrierung)
**Skill**: `/dart-calibration`, `/dart-vision-pipeline`
**Agent**: `backend-coder` (T1-T3), `frontend-coder` (T4)

## Tasks

### P7-T1: Multi-Camera Capture
- **Status**: TODO
- **Beschreibung**: 2-3 Kameras gleichzeitig verwalten
- **Dateien**:
  - `backend/vision/camera.py` — Erweiterung: MultiCameraManager
- **Akzeptanzkriterien**:
  - 2-3 Kameras gleichzeitig öffnen und streamen
  - Frame-Synchronisation (Zeitstempel-basiert)
  - Graceful Degradation wenn Kamera ausfällt

### P7-T2: Stereo-Kalibrierung
- **Status**: TODO
- **Beschreibung**: Extrinsic-Kalibrierung via cv2.stereoCalibrate
- **Dateien**:
  - `backend/vision/calibration.py` — calibrate_stereo()
  - `backend/tests/test_stereo.py`
- **Akzeptanzkriterien**:
  - Gemeinsame Marker-Aufnahmen aus beiden Kameras
  - CALIB_FIX_INTRINSIC Flag gesetzt
  - StereoProfile mit R, T, reprojection_error gespeichert

### P7-T3: Triangulation
- **Status**: TODO
- **Beschreibung**: 2D-Punkte aus 2 Kameras → 3D-Punkt
- **Dateien**:
  - `backend/vision/triangulation.py` — triangulate_point()
- **Akzeptanzkriterien**:
  - cv2.triangulatePoints korrekt angewendet
  - 3D-Punkt auf Board-Ebene projiziert
  - Genauigkeit < 2mm bei korrekt kalibriertem Setup

### P7-T4: Multi-Cam UI
- **Status**: TODO
- **Beschreibung**: Multi-Cam Seite mit Diagnostik
- **Dateien**:
  - `frontend/js/multicam-ui.js`
- **Akzeptanzkriterien**:
  - Alle Kameras als Thumbnails sichtbar
  - Extrinsic-Wizard startbar
  - DiagnosticsPanel: Reprojection Error, FPS pro Kamera
