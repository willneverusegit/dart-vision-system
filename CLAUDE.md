# Dart Vision Scoring System

## Projekt

Webbasiertes automatisches Dart-Scoring-System mit 1-3 Kameras. Erkennt Pfeile via Computer Vision und berechnet Scores in Echtzeit.

- **Backend**: Python 3.11+, FastAPI, OpenCV 4.x, numpy
- **Frontend**: HTML/CSS/JavaScript (Vanilla, kein Framework im MVP)
- **Kommunikation**: REST (FastAPI) + WebSocket für Bild-/Event-Streaming
- **Persistenz**: JSON-Dateien + optional SQLite

## Architektur

```
backend/
  api/          # FastAPI Routers (REST + WebSocket)
  vision/       # Kamera-Capture, Kalibrierung, Bildverarbeitung
  scoring/      # Board-Geometrie, Feld-Mapping, Spiellogik
  models/       # Pydantic Datenmodelle
  tests/        # pytest Tests
  data/profiles/  # Gespeicherte Kalibrierprofile (JSON)
frontend/
  index.html    # SPA Entry Point
  css/          # Styles
  js/           # Modules (Setup, Calibration, Game, WebSocket)
tasks/          # Aufgaben-Breakdown nach Phasen
```

## Konventionen

### Python
- Linting: `ruff check backend/`
- Formatting: `ruff format backend/`
- ruff config: N806 ignoriert (Mathe-Konvention für H, R, T Matrizen) — siehe `backend/pyproject.toml`
- Quick fix: `ruff check backend/ --fix` (safe fixes only)
- Type Hints: überall, Pydantic BaseModel für Datenstrukturen
- Docstrings: Google Style
- Tests: `python -m pytest backend/tests/ --tb=short -q`
- Use FastAPI `TestClient` (sync), nicht `httpx.AsyncClient` — vermeidet async fixture Probleme

### JavaScript
- Vanilla JS, ES Modules
- Kein Build-Tool im MVP
- ESLint wenn vorhanden
- Kein innerHTML — verwende `createElement`/`replaceChildren` (XSS-Prävention)

### Git
- Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
- Ein Feature pro Branch
- Branch-Schema: `feature/phase{N}-{name}` (z.B. `feature/phase2-calibration`)
- PRs gegen `master` (nicht `main`)

## Skill-Routing

| Aufgabe | Skill |
|---------|-------|
| Kamera-Kalibrierung (ChArUco, Intrinsics, Stereo) | `/dart-calibration` |
| Bildverarbeitung (ROI, Graustufen, Canny, Differenz) | `/dart-vision-pipeline` |
| Web-UI (Seiten, Canvas, WebSocket-Client) | `/dart-frontend` |
| API-Endpoints (REST, WebSocket, Pydantic) | `/dart-api` |
| Scoring (Board-Modell, Feldberechnung, Spielmodi) | `/dart-scoring` |

## Agent-Orchestrierung

| Agent | Einsatz |
|-------|---------|
| `research-agent` | Recherche zu OpenCV, FastAPI, Dart-Regeln — schreibt keinen Produktionscode |
| `backend-coder` | Implementiert Python-Module, muss ruff + pytest bestehen |
| `frontend-coder` | Implementiert Web-UI, testet visuell |
| `supervisor` | Liest `tasks/TASK-INDEX.md`, dispatcht passenden Agent, trackt Fortschritt |

**Workflow**: Supervisor liest TASK-INDEX → identifiziert nächsten Task → dispatcht Agent → Agent liest CLAUDE.md + passenden Skill → implementiert → Hooks validieren.

## Task-Referenz

Siehe `tasks/TASK-INDEX.md` für die priorisierte Aufgabenliste (10 Phasen, 34 Sub-Tasks).

## Wichtige Datenmodelle

- `CameraProfile`: id, role, resolution, intrinsics, distortion, roi, homography
- `StereoProfile`: camera_left_id, camera_right_id, R, T, reprojection_error
- `BoardModel`: homography (3x3), board_radius, ring_radii, sector_angles
- `HitEvent`: timestamp, camera_ids, image_points, world_point, score, confidence
- `GameState`: mode, players, current_player, remaining_points, history

## API-Endpoints

- `GET /api/cameras` — verfügbare Geräte
- `POST /api/calibrate/start` — Kalibrierung starten
- `POST /api/calibrate/finish` — Kalibrierung speichern
- `POST /api/game/start` — Spiel starten
- `POST /api/game/stop` — Spiel beenden
- `WebSocket /ws/stream` — Preview-Frames + Events

## Session-Hooks

- `SessionStart` → `.claude/hooks/session-bootstrap.sh` (Branch, Phase, Tasks, Tests)
- `PostToolUse:Bash` → `.claude/hooks/phase-switch.sh` (bei `git checkout`/`switch`)
- `.agent-memory/` enthält Session-Kontext für agentic-os Integration
