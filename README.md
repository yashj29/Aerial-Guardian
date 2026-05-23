# Aerial Guardian — Drone Person Detection & Tracking

**Assignment:** The Aerial Guardian · BotLab Dynamics Research Engineer (CV & Deep Learning)  
**Dataset:** VisDrone2019-MOT-val (Task 4 — Multi-Object Tracking, Person classes)  
**Model size:** ~6 MB · **Hardware tested:** Intel Core i7 CPU (no GPU) · **Avg FPS:** ~10 FPS (CPU), ~25 FPS (GPU TensorRT)

---

## Quickest way to see results (no setup needed)

**Pre-built output videos (all 7 sequences):**  
📁 [Google Drive — Output Videos](https://drive.google.com/PLACEHOLDER) ← download and play directly

Each video shows:
- **Colored bounding boxes** around each tracked person
- **Unique persistent IDs** (ID:N labels) that remain consistent across frames
- **Trajectory tails** — 40-frame path history per person
- **FPS + active track count** overlay

> Videos are not included in the repo (largest is 187 MB). Download from the Drive link above.

---

## Architecture

```
Input Frame
    │
    ▼
YOLOv8n + SAHI Sliced Inference       ← small-object adaptation
(slice 1920×1080 into 640×640 patches, detect on each, merge)
    │
    ▼
BoT-SORT Tracker + ECC Camera Motion Compensation  ← drone ego-motion fix
(estimate homography between frames, warp track predictions before matching)
    │
    ▼
Annotated Output (boxes · IDs · tails)
```

**Why these choices:** See [REPORT.md](REPORT.md) for full technical justification.

---

## Setup & Run

### Requirements
- Python 3.10+
- (~6 MB model weights, auto-downloaded on first run)

```bash
git clone <repo-url>
cd aerial-guardian
pip install -r requirements.txt
```

### Process all sequences
```bash
python run.py
# Output videos saved to outputs/
```

### Process a single sequence
```bash
python run.py --seq uav0000086_00000_v
```

### Use GPU (if available)
```bash
python run.py --device cuda:0
```

### Flags
| Flag | Effect |
|---|---|
| `--no-sahi` | Disable sliced inference (~18 FPS, lower recall on tiny persons) |
| `--tracker bytetrack` | Use ByteTrack instead of BoT-SORT |
| `--device cpu/cuda:0` | Override compute device |

---

## Interactive Demo (bonus)

A React web app is included for interactive exploration:

```bash
# Terminal 1 — Python API
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# Terminal 2 — React frontend
cd web && npm install && npm run dev
```

Open **http://localhost:5173**

Features:
- **Video tab** — plays output MP4 for any sequence
- **Live tab** — real-time WebSocket inference stream
- **Compare tab** — scrub frame-by-frame, original vs tracked side-by-side  
- **Metrics tab** — FPS chart and track count over time

Or just double-click `start.bat` on Windows.

---

## Dataset layout expected

```
Desktop/botlab/
├── VisDrone2019-MOT-val/
│   ├── sequences/
│   │   ├── uav0000086_00000_v/   ← frames as 0000001.jpg ...
│   │   └── ...
│   └── annotations/
└── aerial-guardian/              ← this repo
```

---

## Project structure

```
aerial-guardian/
├── src/
│   ├── detector.py     # YOLOv8n + SAHI sliced inference
│   ├── tracker.py      # BoT-SORT with ECC camera motion compensation
│   ├── visualizer.py   # Bounding boxes, unique IDs, trajectory tails
│   └── pipeline.py     # End-to-end sequence processor (outputs H.264 MP4)
├── api/
│   └── server.py       # FastAPI backend for the React demo
├── web/                # React + Vite frontend
├── configs/
│   └── config.yaml     # All tunable parameters
├── outputs/            # Generated output videos (MP4, H.264)
├── run.py              # CLI entry point
├── evaluate.py         # MOT metrics (MOTA/MOTP) vs ground truth
├── requirements.txt
├── REPORT.md           # Technical design decisions
└── start.bat           # Windows: launches both servers
```

---

## FPS Benchmark

| Configuration | Hardware | Avg FPS |
|---|---|---|
| YOLOv8n full-frame | Intel Core i7 CPU | ~18 FPS |
| YOLOv8n + SAHI (12 patches) | Intel Core i7 CPU | ~10 FPS |
| YOLOv8n + SAHI + BoT-SORT | Intel Core i7 CPU | ~10 FPS |
| YOLOv8n + SAHI (TensorRT FP16) | Jetson AGX Orin | ~15–20 FPS (estimated) |

---

## Key design decisions

See [REPORT.md](REPORT.md) for:
1. Why SAHI solves the small-object detection problem at drone altitude
2. How ECC-based Camera Motion Compensation reduces ID switching from drone ego-motion
3. How to deploy on NVIDIA Jetson (TensorRT FP16 export, GPU-accelerated optical flow)
4. Engineering trade-offs: speed vs accuracy
