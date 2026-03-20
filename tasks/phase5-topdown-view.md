# Phase 5 — Top-Down View & Score-Overlay

**Impact**: 5 | **Aufwand**: 3 | **Score**: 1.67
**Abhängigkeit**: Phase 2 (Kalibrierung mit Homographie)
**Skill**: `/dart-frontend`, `/dart-calibration`
**Agent**: `backend-coder` (T1), `frontend-coder` (T2-T3)

## Tasks

### P5-T1: Homography Warp
- **Status**: TODO
- **Beschreibung**: Top-Down Board-Bild via cv2.warpPerspective erzeugen
- **Dateien**:
  - `backend/vision/warp.py` — warp_to_topdown()
- **Akzeptanzkriterien**:
  - Kamerabild wird zu entzerrter Draufsicht transformiert
  - Ausgabegröße konfigurierbar (z.B. 500x500)
  - Funktioniert mit gespeicherter Homographie aus Profil

### P5-T2: Canvas Overlay
- **Status**: TODO
- **Beschreibung**: Erkannte Darts auf Top-Down-Ansicht im Frontend zeichnen
- **Dateien**:
  - `frontend/js/board-canvas.js` — Erweiterung: drawHits()
- **Akzeptanzkriterien**:
  - Treffer-Marker an korrekter Position auf Canvas
  - Farbkodierung nach Confidence (grün/gelb/rot)
  - Letzte 3 Würfe sichtbar, ältere verblassen

### P5-T3: Score Overlay
- **Status**: TODO
- **Beschreibung**: Echtzeit Score-Anzeige mit Animation
- **Dateien**:
  - `frontend/js/game-ui.js` — Erweiterung: animateScore()
- **Akzeptanzkriterien**:
  - Score-Popup bei neuem Treffer (z.B. "+60 T20")
  - Gesamtpunkte animiert aktualisiert
  - Wurf-Historie scrollbar
