"""
Detector module: YOLOv8n + SAHI sliced inference.

Why SAHI? Drone footage at altitude makes people appear at 10-30px height.
Standard 640x640 inference misses them. SAHI slices the frame into overlapping
patches, runs detection on each, then merges with NMS — dramatically improving
small-object recall without any model retraining.
"""

import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil


class SAHIDetector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_path = cfg["model"]
        self.confidence = cfg["confidence"]
        self.iou = cfg["iou_threshold"]
        self.device = cfg["device"]
        self.target_class = cfg["target_class"]
        self.use_sahi = cfg.get("use_sahi", True)

        # SAHI wraps the YOLO model — same weights, smarter inference strategy
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=self.confidence,
            device=self.device,
        )

        # Keep a direct YOLO reference for full-frame fallback
        self._yolo = YOLO(self.model_path)

        self.slice_h = cfg.get("slice_height", 640)
        self.slice_w = cfg.get("slice_width", 640)
        self.overlap_h = cfg.get("overlap_height_ratio", 0.2)
        self.overlap_w = cfg.get("overlap_width_ratio", 0.2)
        self.postprocess_thresh = cfg.get("postprocess_match_threshold", 0.5)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection on a single BGR frame.

        Returns:
            np.ndarray of shape (N, 6): [x1, y1, x2, y2, confidence, class_id]
        """
        if self.use_sahi:
            return self._detect_sahi(frame)
        return self._detect_full(frame)

    def _detect_sahi(self, frame: np.ndarray) -> np.ndarray:
        """Sliced inference: detect on patches, merge results."""
        pil_img = read_image_as_pil(frame)

        result = get_sliced_prediction(
            pil_img,
            self.detection_model,
            slice_height=self.slice_h,
            slice_width=self.slice_w,
            overlap_height_ratio=self.overlap_h,
            overlap_width_ratio=self.overlap_w,
            postprocess_match_threshold=self.postprocess_thresh,
            verbose=0,
        )

        detections = []
        for obj in result.object_prediction_list:
            if obj.category.id != self.target_class:
                continue
            bbox = obj.bbox
            detections.append([
                bbox.minx, bbox.miny, bbox.maxx, bbox.maxy,
                obj.score.value, obj.category.id
            ])

        if not detections:
            return np.empty((0, 6), dtype=np.float32)
        return np.array(detections, dtype=np.float32)

    def _detect_full(self, frame: np.ndarray) -> np.ndarray:
        """Full-frame inference fallback (faster but misses small objects)."""
        results = self._yolo.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.target_class],
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return np.empty((0, 6), dtype=np.float32)

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, conf, cls]).astype(np.float32)
