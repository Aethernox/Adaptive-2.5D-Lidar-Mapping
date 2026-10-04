"""
FastAPI & WebSocket High-FPS Real-Time Simulation Server
Zero-latency streaming server powered by pre-processed simulation packages.
Eliminates on-the-fly inference and mapping bottlenecks, delivering 60+ FPS localhost simulation.
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Set, Optional, Dict, Any, Union
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from simulation.preprocessor import SimulationPreprocessor, PreprocessedSimulation


app = FastAPI(title="Adaptive LiDAR Perception & Simulation Server")

# Connected WebSocket clients
active_websockets: Set[WebSocket] = set()

# Server state
simulation_data: Optional[PreprocessedSimulation] = None
is_paused: bool = False
playback_speed: float = 1.0
target_fps: float = 30.0
current_frame_idx: int = 0
sequence_name: str = "00"
loop_playback: bool = True


@app.get("/")
async def get_index():
    static_index = Path(__file__).parent / "static" / "index.html"
    return FileResponse(str(static_index))


@app.get("/api/metadata")
async def get_metadata():
    """Returns sequence metadata, frame counts, and performance metrics."""
    global simulation_data, sequence_name
    if simulation_data is not None:
        return {
            "status": "ready",
            "sequence": sequence_name,
            "total_frames": len(simulation_data),
            "current_frame": current_frame_idx,
            "metadata": simulation_data.metadata
        }
    return {"status": "loading", "sequence": sequence_name, "total_frames": 0}


async def broadcast_current_frame():
    """Immediately sends the current frame to all connected clients."""
    global simulation_data, current_frame_idx, active_websockets
    if not active_websockets or simulation_data is None or len(simulation_data) == 0:
        return

    current_frame_idx = current_frame_idx % len(simulation_data)
    json_str = simulation_data.get_json(current_frame_idx)
    
    disconnected = set()
    for ws in list(active_websockets):
        try:
            await ws.send_text(json_str)
        except Exception:
            disconnected.add(ws)
            
    for ws in disconnected:
        active_websockets.discard(ws)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global is_paused, playback_speed, current_frame_idx, target_fps, loop_playback
    await websocket.accept()
    active_websockets.add(websocket)

    # Immediately send initial frame and metadata
    if simulation_data and len(simulation_data) > 0:
        init_payload = {
            "type": "init_metadata",
            "sequence": sequence_name,
            "total_frames": len(simulation_data),
            "start_frame": simulation_data.metadata.get("start_frame", 0),
            "end_frame": simulation_data.metadata.get("end_frame", len(simulation_data)),
            "fps": target_fps,
            "metadata": simulation_data.metadata
        }
        await websocket.send_text(json.dumps(init_payload))
        await websocket.send_text(simulation_data.get_json(current_frame_idx))

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            cmd = msg.get("cmd")
            
            if cmd == "pause":
                is_paused = True
            elif cmd == "play":
                is_paused = False
            elif cmd == "toggle_play":
                is_paused = not is_paused
            elif cmd == "set_speed":
                playback_speed = max(0.05, float(msg.get("speed", 1.0)))
            elif cmd == "set_fps":
                target_fps = max(1.0, float(msg.get("fps", 30.0)))
            elif cmd == "seek":
                if simulation_data and len(simulation_data) > 0:
                    current_frame_idx = int(msg.get("frame", 0)) % len(simulation_data)
                    await broadcast_current_frame()
            elif cmd == "step":
                delta = int(msg.get("delta", 1))
                if simulation_data and len(simulation_data) > 0:
                    current_frame_idx = (current_frame_idx + delta) % len(simulation_data)
                    is_paused = True
                    await broadcast_current_frame()
            elif cmd == "reset":
                current_frame_idx = 0
                is_paused = False
                await broadcast_current_frame()
            elif cmd == "toggle_loop":
                loop_playback = not loop_playback

    except WebSocketDisconnect:
        active_websockets.discard(websocket)
    except Exception:
        active_websockets.discard(websocket)


async def high_fps_simulation_loop():
    """
    High-frequency non-blocking simulation broadcast loop.
    Reads pre-serialized packets from memory in O(1) time.
    """
    global current_frame_idx, is_paused, playback_speed, target_fps, simulation_data, loop_playback

    while True:
        if simulation_data and len(simulation_data) > 0:
            total_frames = len(simulation_data)
            
            if not is_paused and active_websockets:
                current_frame_idx = current_frame_idx % total_frames
                json_str = simulation_data.get_json(current_frame_idx)
                
                # Fast broadcast
                disconnected = set()
                for ws in list(active_websockets):
                    try:
                        await ws.send_text(json_str)
                    except Exception:
                        disconnected.add(ws)
                for ws in disconnected:
                    active_websockets.discard(ws)

                # Advance frame
                if current_frame_idx + 1 >= total_frames:
                    if loop_playback:
                        current_frame_idx = 0
                    else:
                        is_paused = True
                else:
                    current_frame_idx += 1

            # Precision pacing based on target FPS and speed multiplier
            effective_fps = max(1.0, target_fps * playback_speed)
            interval = 1.0 / effective_fps
            await asyncio.sleep(interval)
        else:
            await asyncio.sleep(0.05)


def start_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    sequence: str = "00",
    start_frame: int = 0,
    end_frame: Optional[int] = 200,
    fps: float = 30.0,
    mode: str = "ground_truth",
    subsample_ratio: int = 3,
    force_preprocess: bool = False,
    preprocessed_sim: Optional[PreprocessedSimulation] = None,
    dataset_root: str = "kitti_dataset",
    cache_dir: str = "data_cache"
):
    """
    Starts the high-FPS localhost simulation server.
    Ensures dataset sequence is pre-processed in terminal ahead of time.
    """
    global sequence_name, simulation_data, target_fps
    sequence_name = sequence
    target_fps = fps

    # 1. Prepare Simulation Package
    if preprocessed_sim is not None:
        simulation_data = preprocessed_sim
    else:
        preprocessor = SimulationPreprocessor(
            dataset_root=dataset_root,
            cache_dir=cache_dir,
            subsample_ratio=subsample_ratio
        )
        simulation_data = preprocessor.get_or_create_simulation(
            sequence=sequence,
            start_frame=start_frame,
            end_frame=end_frame,
            mode=mode,
            subsample_ratio=subsample_ratio,
            force_reprocess=force_preprocess
        )

    print(f"\n==================================================================")
    print(f" HIGH-FPS SIMULATION SERVER RUNNING AT: http://localhost:{port}")
    print(f" Sequence: {sequence} | Ready Frames: {len(simulation_data)} | Target FPS: {target_fps}")
    print(f" Status  : Zero pipeline latency active (O(1) memory streaming)")
    print(f"==================================================================\n")

    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(high_fps_simulation_loop())

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
