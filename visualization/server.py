"""
FastAPI & WebSocket Real-Time Visualization Server
Authoritative implementation matching Section 19 & SOFTWARE_ARCHITECTURE.md §5 (M4).
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Set, Optional
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from simulation.kitti_replay import VirtualLidarReplay
from visualization.dashboard_bridge import DashboardBridge


app = FastAPI(title="Adaptive LiDAR Perception Dashboard")
bridge = DashboardBridge(subsample_ratio=3)

# Connected clients
active_websockets: Set[WebSocket] = set()

# Server state
replay_engine: Optional[VirtualLidarReplay] = None
is_paused = False
playback_speed = 1.0
current_frame_idx = 0
sequence_name = "00"


@app.get("/")
async def get_index():
    static_index = Path(__file__).parent / "static" / "index.html"
    return FileResponse(str(static_index))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            cmd = msg.get("cmd")
            
            global is_paused, playback_speed, current_frame_idx
            if cmd == "pause":
                is_paused = True
            elif cmd == "play":
                is_paused = False
            elif cmd == "set_speed":
                playback_speed = float(msg.get("speed", 1.0))
            elif cmd == "seek":
                current_frame_idx = int(msg.get("frame", 0))
            elif cmd == "reset":
                current_frame_idx = 0
                is_paused = False
    except WebSocketDisconnect:
        active_websockets.remove(websocket)


async def broadcast_frame(payload: dict):
    if not active_websockets:
        return
    msg_str = json.dumps(payload)
    disconnected = set()
    for ws in active_websockets:
        try:
            await ws.send_text(msg_str)
        except Exception:
            disconnected.add(ws)
    for ws in disconnected:
        active_websockets.remove(ws)


async def replay_loop():
    global current_frame_idx, is_paused, playback_speed
    
    replay = VirtualLidarReplay(sequence=sequence_name, mode="ground_truth")
    dataset = replay.dataset
    total_frames = len(dataset)
    
    print(f"[Dashboard Server] Replay streaming ready for sequence {sequence_name} ({total_frames} frames)...")

    while True:
        if not is_paused and total_frames > 0:
            current_frame_idx = current_frame_idx % total_frames
            
            frame_gen = replay.stream_frames()
            # Advance to current_frame_idx
            frame_item = None
            for idx, item in enumerate(frame_gen):
                if idx == current_frame_idx:
                    frame_item = item
                    break
            
            if frame_item:
                payload = bridge.create_frame_payload(
                    frame=frame_item["frame"],
                    pose=frame_item["pose"],
                    pred_labels=frame_item["pred_labels"],
                    map_snapshot=frame_item["map_snapshot"],
                    track_set=frame_item["track_set"],
                    metrics=frame_item["metrics"],
                    trajectory=frame_item["trajectory"],
                    classes_cfg=dataset.classes_cfg
                )
                await broadcast_frame(payload)

            current_frame_idx = (current_frame_idx + 1) % total_frames

        interval = 0.10 / max(playback_speed, 0.1)
        await asyncio.sleep(interval)


def start_server(host: str = "0.0.0.0", port: int = 8080, sequence: str = "00"):
    global sequence_name
    sequence_name = sequence
    
    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(replay_loop())

    print(f"\n==================================================================")
    print(f" ADAPTIVE LIDAR DASHBOARD RUNNING AT: http://localhost:{port}")
    print(f"==================================================================\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
