# Phase 8 — Eco/Debug-Modi & Performance

**Impact**: 3 | **Aufwand**: 4 | **Score**: 0.75
**Abhängigkeit**: Phase 4 (Treffererkennung) + Phase 7 (Stereo)
**Skill**: `/dart-vision-pipeline`
**Agent**: `backend-coder`

## Tasks

### P8-T1: Shared-Memory IPC
- **Status**: TODO
- **Beschreibung**: Frame-Übergabe via multiprocessing.shared_memory
- **Dateien**:
  - `backend/vision/shared_buffer.py` — FrameBuffer Klasse
  - `backend/tests/test_shared_buffer.py`
- **Akzeptanzkriterien**:
  - Capture-Prozess schreibt Frames in Shared Memory
  - Processing-Worker liest ohne Kopie
  - Kein Memory-Leak, sauberes Cleanup

### P8-T2: Eco Mode
- **Status**: TODO
- **Beschreibung**: Reduzierte Frame-Rate, minimale Verarbeitung
- **Dateien**:
  - `backend/vision/pipeline.py` — Erweiterung: mode-abhängige Pipeline
- **Akzeptanzkriterien**:
  - Eco: nur jedes 3. Frame, kein Canny, nur Differenzbild
  - Automatisch aktiviert bei 3 Kameras
  - CPU-Verbrauch messbar geringer

### P8-T3: Debug Mode
- **Status**: TODO
- **Beschreibung**: Alle Pipeline-Stufen visuell als Thumbnails
- **Dateien**:
  - `backend/vision/pipeline.py` — debug_output()
  - `frontend/js/game-ui.js` — Debug-Thumbnails
- **Akzeptanzkriterien**:
  - Grauwert, Differenzbild, Canny, Konturen als separate Thumbnails
  - Toggle via ModeToggle im Frontend
  - Kein Performance-Impact wenn Debug aus
