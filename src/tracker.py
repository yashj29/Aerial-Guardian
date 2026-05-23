"""
Tracker module: BoT-SORT with Camera Motion Compensation (CMC).

Why BoT-SORT over DeepSORT?
- No ReID model needed → lighter weight (with_reid=False)
- Built-in ECC-based CMC: estimates the homography between consecutive frames
  using background pixel correlation, then warps Kalman-filter track predictions
  to account for camera translation/rotation. This directly combats ID switching
  caused by drone ego-motion.

CMC flow per frame:
  1. ECC minimization between current and previous grayscale frame → warp matrix H
  2. All Kalman-predicted track centers are transformed by H before IoU matching
  3. Result: tracks "move with camera" so stationary people stay matched
"""

import numpy as np
import cv2
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.bytetrack.bytetrack import ByteTrack


class DroneTracker:
    def __init__(self, cfg: dict):
        tracker_type = cfg.get("type", "botsort").lower()

        if tracker_type == "botsort":
            # with_reid=False removes the need for a ReID model (~0 extra MB)
            # cmc_method='ecc' enables built-in camera motion compensation
            self.tracker = BotSort(
                with_reid=False,
                cmc_method=cfg.get("cmc_method", "ecc"),
                track_high_thresh=cfg.get("track_high_thresh", 0.5),
                track_low_thresh=cfg.get("track_low_thresh", 0.1),
                new_track_thresh=cfg.get("new_track_thresh", 0.6),
                track_buffer=cfg.get("track_buffer", 30),
                match_thresh=cfg.get("match_thresh", 0.8),
                frame_rate=30,
            )
        else:
            self.tracker = ByteTrack(
                track_thresh=cfg.get("track_high_thresh", 0.45),
                min_conf=cfg.get("track_low_thresh", 0.1),
                track_buffer=cfg.get("track_buffer", 25),
                match_thresh=cfg.get("match_thresh", 0.8),
                frame_rate=30,
            )

        self.tracker_type = tracker_type

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections.

        Args:
            detections: (N, 6) array [x1, y1, x2, y2, conf, class_id]
            frame: BGR frame (used internally for CMC)

        Returns:
            (M, 8) array [x1, y1, x2, y2, track_id, conf, class_id, idx]
            Returns empty (0, 8) array when no active tracks.
        """
        if detections is None or detections.shape[0] == 0:
            empty_dets = np.empty((0, 6), dtype=np.float32)
            result = self.tracker.update(empty_dets, frame)
        else:
            result = self.tracker.update(detections, frame)

        if result is None or len(result) == 0:
            return np.empty((0, 8), dtype=np.float32)
        return np.array(result, dtype=np.float32)

    def reset(self):
        """Reset tracker state between sequences."""
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
        else:
            # Re-initialize with same type
            self.tracker.__init__()
