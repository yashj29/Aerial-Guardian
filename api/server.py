"""
FastAPI backend for the Aerial Guardian demo UI.

Endpoints:
  GET  /api/sequences              - list all sequences with status
  GET  /api/videos/{name}          - stream output MP4
  GET  /api/frames/original/{name}/{idx} - raw JPEG from sequence folder
  GET  /api/frames/tracked/{name}/{idx}  - JPEG extracted from output video
  GET  /api/metrics/{name}         - per-frame metrics JSON
  POST /api/run/{name}             - trigger full pipeline on a sequence
  GET  /api/run/{name}/status      - check if a run is in progress
  WS   /ws/live/{name}             - stream live inference as base64 JPEGs
"""

import asyncio
import base64
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

# Allow importing from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import AerialGuardianPipeline

DATASET_ROOT = ROOT.parent / "VisDrone2019-MOT-val"
OUTPUT_ROOT = ROOT / "outputs"
CONFIG_PATH = ROOT / "configs" / "config.yaml"

app = FastAPI(title="Aerial Guardian API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── shared state ──────────────────────────────────────────────────────────────
_running: dict[str, bool] = {}   # seq_name → is currently processing
_pipeline: Optional[AerialGuardianPipeline] = None


def _get_pipeline() -> AerialGuardianPipeline:
    global _pipeline
    if _pipeline is None:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        cfg["detector"]["use_sahi"] = False   # faster for live demo
        _pipeline = AerialGuardianPipeline(
            det_cfg=cfg["detector"],
            trk_cfg=cfg["tracker"],
            vis_cfg=cfg["visualizer"],
            out_cfg=cfg["output"],
        )
    return _pipeline


def _sequence_names() -> list[str]:
    seq_root = DATASET_ROOT / "sequences"
    if not seq_root.exists():
        return []
    return sorted(p.name for p in seq_root.iterdir() if p.is_dir())


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/api/sequences")
def list_sequences():
    results = []
    for name in _sequence_names():
        video = OUTPUT_ROOT / f"{name}.mp4"
        metrics_file = OUTPUT_ROOT / f"{name}_metrics.json"
        avg_fps = None
        if metrics_file.exists():
            with open(metrics_file) as f:
                m = json.load(f)
            avg_fps = m.get("avg_fps")
        results.append({
            "name": name,
            "has_video": video.exists(),
            "is_running": _running.get(name, False),
            "avg_fps": avg_fps,
        })
    return results


@app.get("/api/videos/{name}")
def serve_video(name: str, request: Request):
    """Serve MP4 with range-request support so browsers can seek."""
    video = OUTPUT_ROOT / f"{name}.mp4"
    if not video.exists():
        return JSONResponse({"error": "video not found"}, status_code=404)

    file_size = video.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        start, end = range_header.replace("bytes=", "").split("-")
        start = int(start)
        end = int(end) if end else file_size - 1
        chunk = end - start + 1

        def iter_chunk():
            with open(video, "rb") as f:
                f.seek(start)
                remaining = chunk
                while remaining:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_chunk(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk),
            },
        )

    return StreamingResponse(
        open(video, "rb"),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )


@app.get("/api/frames/original/{name}/{idx}")
def original_frame(name: str, idx: int):
    """Serve a raw frame from the sequence folder as JPEG."""
    seq_dir = DATASET_ROOT / "sequences" / name
    frames = sorted(list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png")))
    if not frames or idx >= len(frames):
        return JSONResponse({"error": "frame not found"}, status_code=404)
    img = cv2.imread(str(frames[idx]))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/frames/tracked/{name}/{idx}")
def tracked_frame(name: str, idx: int):
    """Extract frame idx from the processed output video."""
    video = OUTPUT_ROOT / f"{name}.mp4"
    if not video.exists():
        return JSONResponse({"error": "video not found"}, status_code=404)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return JSONResponse({"error": "frame read failed"}, status_code=404)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/metrics/{name}")
def get_metrics(name: str):
    metrics_file = OUTPUT_ROOT / f"{name}_metrics.json"
    if not metrics_file.exists():
        return JSONResponse({"error": "metrics not found"}, status_code=404)
    with open(metrics_file) as f:
        return json.load(f)


@app.post("/api/run/{name}")
def run_sequence(name: str):
    """Trigger full pipeline processing in a background thread."""
    if _running.get(name):
        return {"status": "already_running"}

    def _worker():
        _running[name] = True
        try:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            pipeline = AerialGuardianPipeline(
                det_cfg=cfg["detector"],
                trk_cfg=cfg["tracker"],
                vis_cfg=cfg["visualizer"],
                out_cfg=cfg["output"],
            )
            seq_dir = DATASET_ROOT / "sequences" / name
            out_path = OUTPUT_ROOT / f"{name}.mp4"
            pipeline.run_sequence(seq_dir, out_path)
        finally:
            _running[name] = False

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return {"status": "started"}


@app.get("/api/run/{name}/status")
def run_status(name: str):
    video = OUTPUT_ROOT / f"{name}.mp4"
    return {
        "is_running": _running.get(name, False),
        "has_video": video.exists(),
    }


# ── WebSocket: live inference stream ─────────────────────────────────────────

@app.websocket("/ws/live/{name}")
async def live_inference(websocket: WebSocket, name: str):
    """
    Streams live inference over a sequence as base64 JPEGs.
    Client receives JSON: {frame_idx, total, fps, n_dets, n_tracks, image}
    Client can send "stop" to cancel early.
    """
    await websocket.accept()
    seq_dir = DATASET_ROOT / "sequences" / name
    frames = sorted(list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png")))
    if not frames:
        await websocket.send_json({"error": "sequence not found"})
        await websocket.close()
        return

    pipeline = _get_pipeline()
    pipeline.visualizer.reset()
    pipeline.tracker.reset()

    stop_flag = {"stop": False}

    async def listen_for_stop():
        try:
            while True:
                msg = await websocket.receive_text()
                if msg == "stop":
                    stop_flag["stop"] = True
                    break
        except WebSocketDisconnect:
            stop_flag["stop"] = True

    asyncio.create_task(listen_for_stop())

    loop = asyncio.get_event_loop()

    for i, img_path in enumerate(frames):
        if stop_flag["stop"]:
            break

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # Run inference in thread pool so we don't block the event loop
        annotated, meta = await loop.run_in_executor(
            None, pipeline.process_frame_only, frame
        )

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf).decode()

        try:
            await websocket.send_json({
                "frame_idx": i,
                "total": len(frames),
                "fps": meta["fps"],
                "n_dets": meta["n_dets"],
                "n_tracks": meta["n_tracks"],
                "image": b64,
            })
        except WebSocketDisconnect:
            break

        # Yield control so the stop listener can run
        await asyncio.sleep(0)

    try:
        await websocket.send_json({"done": True})
        await websocket.close()
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
