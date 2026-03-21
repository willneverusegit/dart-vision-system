# Projekt-Kontext — Dart Vision System

## Tech Stack
- **Backend**: Python 3.11+, FastAPI, OpenCV 4.x, numpy, Pydantic
- **Frontend**: Vanilla JS (ES Modules), HTML5 Canvas, CSS Dark Theme
- **Kommunikation**: REST + WebSocket
- **Persistenz**: JSON-Dateien, optional SQLite

## Architektur
- `backend/api/` — FastAPI Routers
- `backend/vision/` — Camera, Calibration, Pipeline, Board-Fitting
- `backend/scoring/` — Board-Modell, Spiellogik
- `backend/models/` — Pydantic Datenmodelle
- `frontend/js/` — Module (app, board-canvas, game-ui, calibration-ui, ws-client)

## Konventionen
- ruff + black (Python), type hints, Google docstrings
- Conventional commits (feat:, fix:, refactor:)
- Ein Phase-Branch pro Entwicklungsphase
- Supervisor-Agent orchestriert via TASK-INDEX

## Qualität
- Tests: pytest, alle müssen bestehen vor Commit
- Lint: ruff check, muss sauber sein
- N806 ignoriert (Mathe-Konvention: H, R, T)
