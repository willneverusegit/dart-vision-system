# Phase 2 — Single-Cam Kalibrierung & ROI

**Impact**: 5 | **Aufwand**: 3 | **Score**: 1.67
**Abhängigkeit**: Phase 1 abgeschlossen
**Skill**: `/dart-calibration`
**Agent**: `backend-coder` (T1-T2, T4), `frontend-coder` (T3)

## Tasks

### P2-T1: Camera Capture Module
- **Status**: TODO
- **Beschreibung**: OpenCV VideoCapture Wrapper mit Frame-Streaming
- **Dateien**:
  - `backend/vision/camera.py` — CameraManager, capture_frame(), list_devices()
  - `backend/tests/test_camera.py`
- **Akzeptanzkriterien**:
  - Verfügbare Kameras auflisten
  - Frames als JPEG kodieren und über WebSocket senden
  - Kamera öffnen/schließen ohne Ressource-Leaks

### P2-T2: Intrinsic-Kalibrierung
- **Status**: TODO
- **Beschreibung**: ChArUco-basierte Kalibrierung mit Live-Feedback
- **Dateien**:
  - `backend/vision/calibration.py` — detect_markers(), calibrate_intrinsics(), save_profile()
  - `backend/api/calibration.py` — REST-Endpoints
  - `backend/tests/test_calibration.py`
- **Akzeptanzkriterien**:
  - ChArUco-Marker werden im Live-Stream erkannt
  - Nach >=15 Frames: Intrinsics berechnet, Reprojection Error < 0.5
  - Profil als JSON gespeichert in `backend/data/profiles/`

### P2-T3: ROI-Auswahl UI
- **Status**: TODO
- **Beschreibung**: Canvas-Overlay zum Zeichnen des ROI-Rechtecks
- **Dateien**:
  - `frontend/js/calibration-ui.js` — ROI Cropper Komponente
- **Akzeptanzkriterien**:
  - Draggable/resizable Rechteck auf Kamerabild
  - ROI an Backend senden (POST)
  - Vorschau des gecropten Bereichs

### P2-T4: Board-Fitting
- **Status**: TODO
- **Beschreibung**: Homographie-Berechnung aus Board-Erkennung
- **Dateien**:
  - `backend/vision/board_fitting.py` — detect_board(), compute_homography()
  - `backend/tests/test_board_fitting.py`
- **Akzeptanzkriterien**:
  - Board-Mittelpunkt + Orientierung erkannt
  - Homographie-Matrix berechnet und im Profil gespeichert
  - Visuelles Overlay zeigt erkanntes Board-Modell
