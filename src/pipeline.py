"""
Main pipeline: ties together detection, tracking, and visualization.
Processes VisDrone image-sequence folders.
"""

import json
import subprocess
import time
import cv2
import numpy as np
import imageio_ffmpeg
from pathlib import Path
from tqdm import tqdm

from src.detector import SAHIDetector
from src.tracker import DroneTracker
from src.visualizer import Visualizer


class _VideoWriter:
    """Wrapper that writes MJPG AVI then re-encodes to H.264 MP4 on close."""

    def __init__(self, final_path: Path, w: int, h: int, fps: float):
        self.final_path = final_path
        self.tmp_path = final_path.with_suffix(".tmp.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(str(self.tmp_path), fourcc, fps, (w, h))
        self._fps = fps

    def write(self, frame: np.ndarray):
        self._writer.write(frame)

    def release(self):
        self._writer.release()
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg, "-y", "-i", str(self.tmp_path),
             "-vcodec", "libx264", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(self.final_path)],
            capture_output=True, check=True,
        )
        self.tmp_path.unlink(missing_ok=True)


class AerialGuardianPipeline:
    def __init__(self, det_cfg: dict, trk_cfg: dict, vis_cfg: dict, out_cfg: dict):
        self.detector = SAHIDetector(det_cfg)
        self.tracker = DroneTracker(trk_cfg)
        self.visualizer = Visualizer(vis_cfg)
        self.out_cfg = out_cfg

    def run_sequence(self, seq_dir: str | Path, output_path: str | Path) -> dict:
        """
        Process a VisDrone image sequence (folder of JPG frames).
        Saves a sidecar _metrics.json next to the output video.
        Output is H.264 MP4 for browser compatibility.
        """
        seq_dir = Path(seq_dir)
        output_path = Path(output_path)
        frames = sorted(
            list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png"))
        )
        if not frames:
            raise FileNotFoundError(f"No images found in {seq_dir}")

        sample = cv2.imread(str(frames[0]))
        h, w = sample.shape[:2]
        fps_out = self.out_cfg.get("video_fps", 30)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = _VideoWriter(output_path, w, h, fps_out)

        per_frame = []
        self.visualizer.reset()
        self.tracker.reset()

        for img_path in tqdm(frames, desc=seq_dir.name, unit="frame"):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            t0 = time.perf_counter()
            dets = self.detector.detect(frame)
            tracks = self.tracker.update(dets, frame)
            elapsed = time.perf_counter() - t0

            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            per_frame.append({
                "fps": round(fps, 2),
                "n_dets": int(len(dets)),
                "n_tracks": int(len(tracks)),
            })

            annotated = self.visualizer.draw(frame, tracks, fps)
            writer.write(annotated)

        writer.release()  # also triggers H.264 re-encode

        fps_values = [f["fps"] for f in per_frame]
        avg_fps = float(np.mean(fps_values)) if fps_values else 0.0

        metrics = {
            "sequence": seq_dir.name,
            "frames": len(per_frame),
            "avg_fps": round(avg_fps, 2),
            "width": w,
            "height": h,
            "per_frame": per_frame,
        }
        metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        print(f"  Sequence done | frames={len(per_frame)} | avg FPS={avg_fps:.2f}")
        return metrics

    def process_frame_only(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """Run detect+track on a single frame. Used by the API for live streaming."""
        t0 = time.perf_counter()
        dets = self.detector.detect(frame)
        tracks = self.tracker.update(dets, frame)
        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        annotated = self.visualizer.draw(frame, tracks, fps)
        return annotated, {"fps": round(fps, 2), "n_dets": int(len(dets)), "n_tracks": int(len(tracks))}
