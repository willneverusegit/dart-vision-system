# Phase 1 — MVP: Basic UI + Manual Scoring

**Impact**: 5 | **Aufwand**: 2 | **Score**: 2.50
**Skill**: `/dart-api`, `/dart-frontend`, `/dart-scoring`
**Agent**: `backend-coder` (T1-T3), `frontend-coder` (T4-T5), beide (T6)

## Tasks

### P1-T1: Projekt-Init
- **Status**: TODO
- **Agent**: `backend-coder`
- **Beschreibung**: Python-Projekt aufsetzen
- **Dateien erstellen**:
  - `backend/pyproject.toml` — ruff, pytest config
  - `backend/requirements.txt` — fastapi, uvicorn, opencv-python, numpy, pydantic
  - `backend/__init__.py`
  - `backend/main.py` — FastAPI Hello World mit `/api/health`
- **Akzeptanzkriterien**:
  - `uvicorn backend.main:app` startet ohne Fehler
  - `GET /api/health` gibt `{"status": "ok"}` zurück
  - `ruff check backend/` ohne Fehler

### P1-T2: Datenmodelle
- **Status**: TODO
- **Agent**: `backend-coder`
- **Skill**: `/dart-scoring`, `/dart-api`
- **Beschreibung**: Alle Pydantic-Modelle aus der Spezifikation implementieren
- **Dateien erstellen**:
  - `backend/models/camera.py` — CameraProfile, CameraInfo, CameraRole
  - `backend/models/board.py` — BoardModel
  - `backend/models/game.py` — GameState, HitEvent, GameConfig, GameResult
  - `backend/models/stereo.py` — StereoProfile
  - `backend/models/__init__.py` — Re-exports
- **Akzeptanzkriterien**:
  - Alle Modelle aus Spezifikation abgedeckt
  - JSON-Serialisierung/Deserialisierung funktioniert
  - `backend/tests/test_models.py` besteht

### P1-T3: Basic API
- **Status**: TODO
- **Agent**: `backend-coder`
- **Skill**: `/dart-api`
- **Beschreibung**: REST-Endpoints für Spiel-CRUD und WebSocket-Stub
- **Dateien erstellen**:
  - `backend/api/__init__.py`
  - `backend/api/game.py` — POST /game/start, POST /game/stop, GET /game/state
  - `backend/api/websocket.py` — WebSocket /ws/stream (Echo-Stub)
- **Akzeptanzkriterien**:
  - Spiel starten/stoppen über API möglich
  - WebSocket-Verbindung kann aufgebaut werden
  - `backend/tests/test_api.py` besteht

### P1-T4: Frontend-Scaffold
- **Status**: TODO
- **Agent**: `frontend-coder`
- **Skill**: `/dart-frontend`
- **Beschreibung**: HTML/CSS/JS Grundgerüst mit Hash-Routing
- **Dateien erstellen**:
  - `frontend/index.html` — SPA Shell, Navigation
  - `frontend/css/style.css` — Dark Theme, Layout
  - `frontend/js/app.js` — AppState, Hash-Router
  - `frontend/js/ws-client.js` — WebSocket-Client (Stub)
- **Akzeptanzkriterien**:
  - Seiten wechseln über Navigation (#setup, #calibration, #game)
  - Dark Theme sichtbar
  - Responsive auf Desktop + Tablet

### P1-T5: Game Page UI
- **Status**: TODO
- **Agent**: `frontend-coder`
- **Skill**: `/dart-frontend`, `/dart-scoring`
- **Beschreibung**: Dartboard-Canvas mit manuellem Click-to-Score
- **Dateien erstellen**:
  - `frontend/js/board-canvas.js` — Dartboard zeichnen (Sektoren, Ringe, Farben)
  - `frontend/js/game-ui.js` — Scoreboard, Wurf-Historie, Spieler-Anzeige
- **Akzeptanzkriterien**:
  - Dartboard wird korrekt gezeichnet (20 Sektoren, Double/Triple/Bull Ringe)
  - Klick auf Board → korrekter Score berechnet und angezeigt
  - Scoreboard zeigt Gesamtpunkte und Wurf-Historie

### P1-T6: Integration
- **Status**: TODO
- **Agent**: `backend-coder` + `frontend-coder`
- **Beschreibung**: Frontend mit Backend verbinden, End-to-End manuelles Scoring
- **Akzeptanzkriterien**:
  - Frontend ruft API auf (Spiel starten/stoppen)
  - Manueller Score wird über API gespeichert
  - Spielstand bleibt über Page-Reload bestehen (via API)
  - WebSocket-Verbindung aktiv
