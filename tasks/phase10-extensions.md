# Phase 10 — Dritte Kamera & Erweiterungen

**Impact**: 2 | **Aufwand**: 5 | **Score**: 0.40
**Abhängigkeit**: Phase 7 (Stereo)
**Skill**: `/dart-calibration`, `/dart-vision-pipeline`
**Agent**: `backend-coder`

## Tasks

### P10-T1: Drei-Kamera Triangulation
- **Status**: TODO
- **Beschreibung**: Triangulation mit 3 Kameras für höhere Genauigkeit
- **Dateien**:
  - `backend/vision/triangulation.py` — Erweiterung: multi_triangulate()
- **Akzeptanzkriterien**:
  - 3 Kamera-Paare triangulieren, Median als Ergebnis
  - Outlier-Detection bei widersprüchlichen Messungen
  - Genauigkeit < 1mm

### P10-T2: Kamera Auto-Discovery & Hot-Plug
- **Status**: TODO
- **Beschreibung**: Automatische Erkennung neuer/entfernter Kameras
- **Dateien**:
  - `backend/vision/camera.py` — Erweiterung: watch_devices()
- **Akzeptanzkriterien**:
  - Neue Kamera wird automatisch erkannt und gelistet
  - Entfernte Kamera wird graceful behandelt (kein Crash)
  - UI aktualisiert sich automatisch
