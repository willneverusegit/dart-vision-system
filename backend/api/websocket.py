import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Echo stub: receive and echo back
            data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            await websocket.send_json({"type": "echo", "data": data})
    except (WebSocketDisconnect, TimeoutError):
        pass
