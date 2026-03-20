# Phase 4 — Single-Cam Treffererkennung

**Impact**: 4 | **Aufwand**: 4 | **Score**: 1.00
**Abhängigkeit**: Phase 2 (Kalibrierung) abgeschlossen
**Skill**: `/dart-vision-pipeline`, `/dart-scoring`
**Agent**: `backend-coder`

## Tasks

### P4-T1: Background Subtraction
- **Status**: TODO
- **Beschreibung**: Differenzbild zwischen leerem Board und aktuellem Frame
- **Dateien**:
  - `backend/vision/detection.py` — capture_background(), detect_motion()
  - `backend/tests/test_detection.py`
- **Akzeptanzkriterien**:
  - Hintergrundbild speichern (leeres Board)
  - Bewegung erkennen wenn Pfeil eintrifft
  - Vibrations-Delay (300ms warten nach Einschlag)

### P4-T2: Contour Pipeline
- **Status**: TODO
- **Beschreibung**: Canny + Konturen + Dart-Tip Extraktion
- **Dateien**:
  - `backend/vision/pipeline.py` — process_frame(), extract_tip()
- **Akzeptanzkriterien**:
  - Pfeil-Konturen von Board-Konturen unterscheiden
  - Tip-Position als (x, y) Pixel-Koordinate extrahiert
  - Debug-Modus zeigt alle Pipeline-Stufen

### P4-T3: Field Mapping
- **Status**: TODO
- **Beschreibung**: Tip-Koordinaten via Homographie → Board-Koordinaten → Feld
- **Dateien**:
  - `backend/scoring/board_model.py` — pixel_to_board(), get_score()
- **Akzeptanzkriterien**:
  - Pixel-Koordinaten korrekt auf Board-Koordinaten transformiert
  - Score-Berechnung stimmt (Spot-Checks: Bull, T20, D16)
  - Unit-Tests mit bekannten Koordinaten

### P4-T4: Auto-Score Integration
- **Status**: TODO
- **Beschreibung**: Pipeline → HitEvent → GameState → WebSocket Push
- **Dateien**:
  - `backend/vision/pipeline.py` — Erweiterung: HitEvent erzeugen
  - `backend/api/websocket.py` — HitEvents an Frontend senden
- **Akzeptanzkriterien**:
  - Erkannter Treffer erzeugt automatisch HitEvent
  - GameState wird aktualisiert
  - Frontend zeigt Treffer in Echtzeit
