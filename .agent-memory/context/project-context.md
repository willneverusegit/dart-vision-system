# Projekt-Kontext — Dart Vision System

## Tech Stack
- **Backend**: Python 3.11+, FastAPI, OpenCV 4.x, numpy, Pydantic
- **Frontend**: Vanilla JS (ES Modules), HTML5 Canvas, CSS Dark Theme
- **Kommunikation**: REST + WebSocket
- **Persistenz**: JSON-Dateien, optional SQLite

## Architektur
- `backend/api/` — FastAPI Routers (cameras, calibration, detection, game, profiles, warp, websocket)
- `backend/vision/` — Camera, Calibration, Board-Fitting, Detection, Pipeline
- `backend/scoring/` — Board-Model (Field-Mapping, Score-Berechnung)
- `backend/models/` — Pydantic Datenmodelle (BoardModel, GameState, HitEvent, CameraProfile)
- `backend/services/` — Profile Store (JSON Persistenz)
- `frontend/js/` — Module (app, board-canvas, game-ui, calibration-ui, ws-client)

## Abgeschlossene Phasen
1. Phase 1: MVP (FastAPI + Frontend Grundgeruest)
2. Phase 2: Kalibrierung (ChArUco, Intrinsics, ROI, Board-Fitting)
3. Phase 3: Persistenz (Profile CRUD, JSON Store, Auto-Load)
4. Phase 4: Treffererkennung (BackgroundModel, DetectionLoop, Contour-Pipeline, Scoring)
5. Phase 5: Top-Down View (WarpEngine, Score-Overlay, Animationen)

## Konventionen
- ruff + black (Python), type hints, Google docstrings
- Conventional commits (feat:, fix:, refactor:)
- Ein Phase-Branch pro Entwicklungsphase
- Supervisor-Agent orchestriert via TASK-INDEX
- N806 ignoriert (Mathe-Konvention: H, R, T)

## Qualitaet
- Tests: 106 bestanden (pytest), alle muessen bestehen vor Commit
- Lint: ruff check, muss sauber sein
