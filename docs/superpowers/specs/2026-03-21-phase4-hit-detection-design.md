# Phase 4 — Single-Cam Treffererkennung (Design Spec)

**Date**: 2026-03-21
**Status**: Approved
**Dependencies**: Phase 2 (Kalibrierung), Phase 3 (Persistenz)

## Overview

Automatische Treffererkennung mit einer einzelnen Kamera. Kontinuierlicher Detection-Loop erkennt via Background Subtraction wann ein Pfeil einschlägt, extrahiert die Tip-Position über eine Contour Pipeline, mappt die Pixel-Koordinaten auf das Board und berechnet den Score. Debug-Visualisierung streamt annotierte Frames per WebSocket.

## Architecture: Synchrone Pipeline (Thread-basiert)

Ein dedizierter Background-Thread läuft in einer Loop:
1. Frame grabben → mit Background vergleichen
2. Motion detected → 300ms warten (Vibration) → stabilen Frame grabben
3. Contour Pipeline → Canny → Konturen filtern → Tip extrahieren
4. Field Mapping → Homographie → Polar → Sektor/Ring → Score
5. HitEvent erzeugen → GameState updaten → WebSocket broadcast

## Modules

### 1. `backend/vision/detection.py` — Background Subtraction + Motion Detection

**Classes/Functions:**
- `BackgroundModel`: Stores reference frame (empty board) as grayscale
  - `capture_background(camera_id: str)` — Grab frame via CameraManager, convert to grayscale, store as reference
  - `get_background() -> np.ndarray | None`
- `detect_motion(frame: np.ndarray, background: np.ndarray, threshold: int = 25, min_area: int = 500) -> bool` — `cv2.absdiff`, threshold, check if contour area > min_area
- `wait_for_stable_frame(camera_id: str, delay_ms: int = 300, max_retries: int = 3) -> np.ndarray | None` — After motion detected, wait until consecutive frames stabilize (diff < threshold)

### 2. `backend/vision/pipeline.py` — Contour Pipeline + Detection Loop

**Data:**
- `PipelineResult(BaseModel)`: diff_image (excluded from serialization), edges, contours list, tip_point tuple or None, debug_frame (excluded), field str, score int, multiplier int

**Functions:**
- `process_frame(frame: np.ndarray, background: np.ndarray) -> PipelineResult` — Full pipeline: diff → Canny → contour filter → tip extraction → annotated debug frame
- `extract_tip(contours: list) -> tuple[int, int] | None` — Find dart tip from filtered contours (lowest/sharpest point via convex hull extremum analysis)
- Filter dart contours from board contours via: aspect ratio (elongated), min/max area, position relative to board center

**Class:**
- `DetectionLoop`:
  - `__init__(camera_id, background_model, board_model, on_hit_callback, debug=False)`
  - `start()` / `stop()` — Start/stop background thread
  - `is_running: bool`
  - Thread loop: grab frame → detect_motion → wait_for_stable → process_frame → if tip found: map to field → callback(HitEvent)

### 3. `backend/scoring/board_model.py` — Field Mapping + Score Calculation

**Functions:**
- `pixel_to_board(pixel_point: tuple, homography: np.ndarray) -> tuple[float, float]` — Uses `cv2.perspectiveTransform` (reuses logic from board_fitting.py)
- `board_to_polar(board_point: tuple[float, float]) -> tuple[float, float]` — Cartesian board-mm → (radius_mm, angle_deg)
- `polar_to_field(radius: float, angle: float, board: BoardModel) -> str` — Map to sector (20 sectors at 18° each) + ring → field name (e.g., "T20", "D16", "BULL", "MISS")
- `get_score(field: str) -> tuple[int, int]` — Field name → (base_score, multiplier), e.g., "T20" → (20, 3), "BULL" → (50, 1)

**Ring boundaries (from BoardModel.ring_radii):**
- Bull: 0 – 6.35mm
- Outer Bull: 6.35 – 15.9mm
- Inner Single: 15.9 – 99.0mm
- Triple: 99.0 – 107.0mm
- Outer Single: 107.0 – 162.0mm
- Double: 162.0 – 170.0mm
- Miss: > 170.0mm

**Sector layout** (clockwise from top, standard dartboard):
[20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

### 4. `backend/api/websocket.py` — WebSocket Broadcasting

**Changes:**
- Replace echo stub with `ConnectionManager` class
  - `connect(ws)` / `disconnect(ws)` / `broadcast(message: dict)`
- Message types:
  - `{"type": "hit_event", "data": HitEvent.model_dump()}`
  - `{"type": "debug_frame", "data": "<base64 JPEG>"}`
  - `{"type": "detection_status", "data": {"running": bool, "camera_id": str}}`
  - `{"type": "game_update", "data": GameState.model_dump()}`
- Debug frames sent as base64-encoded JPEG in JSON

**New endpoints (REST, in game.py or new detection router):**
- `POST /api/detection/start` — Start DetectionLoop for given camera
- `POST /api/detection/stop` — Stop DetectionLoop
- `POST /api/detection/background` — Capture background frame
- `GET /api/detection/status` — Check if detection is running

## Data Flow

```
Camera Frame
    ↓
detect_motion(frame, background) → bool
    ↓ true
wait_for_stable_frame(camera_id, 300ms)
    ↓
process_frame(stable_frame, background) → PipelineResult
    ↓
pixel_to_board(tip_point, homography) → board_point
    ↓
board_to_polar(board_point) → (radius, angle)
    ↓
polar_to_field(radius, angle, board_model) → field
    ↓
get_score(field) → (score, multiplier)
    ↓
HitEvent(field, score, multiplier, board_point, ...)
    ↓
GameState update (via game API internal call)
    ↓
WebSocket broadcast(hit_event + debug_frame if debug=True)
```

## Tests

- `test_detection.py` — Background capture, motion detection with synthetic frames (white board + black rectangle as "dart")
- `test_pipeline.py` — Contour filtering, tip extraction with test images
- `test_board_model.py` — Known coordinates → expected fields:
  - (0, 0) → BULL (50)
  - Board center + 103mm at 20-sector angle → T20 (60)
  - Board center + 166mm at 16-sector angle → D16 (32)
  - 200mm from center → MISS (0)
- `test_websocket.py` — Connection manager, HitEvent broadcast

## Non-Goals (MVP)

- Multi-camera triangulation (Phase 7)
- Confidence scoring (Phase 9)
- Multiple simultaneous dart detection (one dart at a time)
- Persistent detection state across server restarts
