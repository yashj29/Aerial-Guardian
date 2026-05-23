"""
Visualizer: draws bounding boxes, unique-colored IDs, and trajectory tails.
"""

import cv2
import numpy as np
from collections import defaultdict


def _color_for_id(track_id: int) -> tuple:
    """Deterministic distinct color per track ID using golden angle hashing."""
    hue = int((track_id * 137.508) % 180)
    hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class Visualizer:
    def __init__(self, cfg: dict):
        self.tail_length = cfg.get("tail_length", 40)
        self.box_thickness = cfg.get("box_thickness", 2)
        self.text_scale = cfg.get("text_scale", 0.5)
        self.show_fps = cfg.get("show_fps", True)
        # track_id -> deque of (cx, cy) center points
        self.history: dict[int, list] = defaultdict(list)

    def draw(self, frame: np.ndarray, tracks: np.ndarray, fps: float = 0.0) -> np.ndarray:
        """
        Draw tracking results onto frame.

        tracks: (M, 7) [x1, y1, x2, y2, track_id, conf, class_id]
        """
        out = frame.copy()

        for t in tracks:
            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            tid = int(t[4])
            color = _color_for_id(tid)

            # Update trajectory history
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.history[tid].append((cx, cy))
            if len(self.history[tid]) > self.tail_length:
                self.history[tid] = self.history[tid][-self.tail_length:]

            # Draw trajectory tail (fades from translucent to opaque)
            pts = self.history[tid]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thickness = max(1, int(self.box_thickness * alpha))
                faded = tuple(int(c * alpha) for c in color)
                cv2.line(out, pts[i - 1], pts[i], faded, thickness, cv2.LINE_AA)

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.box_thickness)

            # ID label with background for readability
            label = f"ID:{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS overlay
        if self.show_fps and fps > 0:
            cv2.putText(out, f"FPS: {fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Active track count
        n = len(tracks)
        cv2.putText(out, f"Tracks: {n}", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        return out

    def reset(self):
        self.history.clear()
