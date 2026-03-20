---
name: dart-api
description: Guidance for FastAPI backend (REST endpoints, WebSocket streaming, Pydantic models, CORS)
user_invocable: true
---

# Dart API Skill

## Wann verwenden
Bei Arbeit an FastAPI-Endpoints, WebSocket-Streaming, Pydantic-Modellen, Backend-Struktur.

## App-Struktur
```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import cameras, calibration, game, websocket

app = FastAPI(title="Dart Vision API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(cameras.router, prefix="/api")
app.include_router(calibration.router, prefix="/api")
app.include_router(game.router, prefix="/api")
app.include_router(websocket.router)
```

## REST-Endpoints

### Kameras
```python
# backend/api/cameras.py
@router.get("/cameras")
async def list_cameras() -> list[CameraInfo]: ...

@router.post("/cameras/{camera_id}/role")
async def set_camera_role(camera_id: str, role: CameraRole): ...
```

### Kalibrierung
```python
# backend/api/calibration.py
@router.post("/calibrate/start")
async def start_calibration(camera_id: str) -> CalibrationStatus: ...

@router.post("/calibrate/finish")
async def finish_calibration(camera_id: str) -> CameraProfile: ...

@router.post("/stereo/start")
async def start_stereo(left_id: str, right_id: str) -> StereoStatus: ...
```

### Spiel
```python
# backend/api/game.py
@router.post("/game/start")
async def start_game(config: GameConfig) -> GameState: ...

@router.post("/game/stop")
async def stop_game() -> GameResult: ...

@router.get("/game/state")
async def get_game_state() -> GameState: ...
```

### Profile
```python
@router.get("/profiles")
async def list_profiles() -> list[ProfileSummary]: ...

@router.post("/profiles")
async def save_profile(profile: CameraProfile): ...

@router.get("/profiles/{name}")
async def load_profile(name: str) -> CameraProfile: ...
```

## WebSocket
```python
# backend/api/websocket.py
@router.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    while True:
        # JPEG-Frame senden (binary)
        frame_bytes = get_current_frame_jpeg()
        await websocket.send_bytes(frame_bytes)
        # Events senden (JSON)
        if event := get_pending_event():
            await websocket.send_json(event.model_dump())
        await asyncio.sleep(1/30)  # 30 FPS
```

## Event-Typen über WebSocket
```python
class WSEvent(BaseModel):
    type: Literal["frame", "hit", "score_update", "calibration_status", "error"]
    data: dict
```

## Dateien
- `backend/main.py` — FastAPI App
- `backend/api/cameras.py` — Kamera-Endpoints
- `backend/api/calibration.py` — Kalibrierungs-Endpoints
- `backend/api/game.py` — Spiel-Endpoints
- `backend/api/websocket.py` — WebSocket-Streaming
- `backend/models/` — Pydantic Modelle
- `backend/tests/test_api.py`

## Starten
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
