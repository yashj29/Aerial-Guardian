"""
Evaluation against VisDrone MOT ground truth.

VisDrone annotation format per line:
  frame_id, track_id, x, y, w, h, score, category, truncation, occlusion

Person categories: 1 (pedestrian), 2 (people)

Computes per-sequence and overall:
  - MOTA  (Multiple Object Tracking Accuracy)
  - MOTP  (Multiple Object Tracking Precision)
  - IDF1
  - ID switches
  - FP / FN / TP counts

Requires: motmetrics  (pip install motmetrics)
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import motmetrics as mm
except ImportError:
    raise ImportError("pip install motmetrics")

DATASET_ROOT = Path(__file__).parent.parent / "VisDrone2019-MOT-val"
OUTPUT_ROOT = Path(__file__).parent / "outputs"

# VisDrone categories to keep (person classes)
PERSON_CATS = {1, 2}


def load_gt(ann_path: Path) -> dict[int, list]:
    """Load ground truth for one sequence. Returns {frame_id: [[x1,y1,x2,y2,tid], ...]}"""
    gt = defaultdict(list)
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            fid, tid, x, y, w, h = int(parts[0]), int(parts[1]), *[float(p) for p in parts[2:6]]
            category = int(parts[7])
            if category not in PERSON_CATS:
                continue
            if w <= 0 or h <= 0:
                continue
            gt[fid].append([x, y, x + w, y + h, tid])
    return gt


def load_predictions(pred_path: Path) -> dict[int, list]:
    """
    Load tracker predictions from a MOTChallenge-format txt file.
    Format: frame_id, track_id, x, y, w, h, conf, -1, -1, -1
    """
    preds = defaultdict(list)
    if not pred_path.exists():
        return preds
    with open(pred_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid, tid, x, y, w, h = int(parts[0]), int(parts[1]), *[float(p) for p in parts[2:6]]
            preds[fid].append([x, y, x + w, y + h, tid])
    return preds


def evaluate_sequence(seq_name: str, gt: dict, preds: dict) -> mm.MOTAccumulator:
    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt.keys()) | set(preds.keys()))

    for fid in all_frames:
        gt_objs = gt.get(fid, [])
        pr_objs = preds.get(fid, [])

        gt_ids = [int(o[4]) for o in gt_objs]
        pr_ids = [int(o[4]) for o in pr_objs]

        if gt_objs and pr_objs:
            gt_boxes = np.array([[o[0], o[1], o[2] - o[0], o[3] - o[1]] for o in gt_objs])
            pr_boxes = np.array([[o[0], o[1], o[2] - o[0], o[3] - o[1]] for o in pr_objs])
            dist = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=0.5)
        else:
            dist = mm.distances.iou_matrix(
                np.zeros((len(gt_objs), 4)),
                np.zeros((len(pr_objs), 4)),
                max_iou=0.5
            ) if gt_objs or pr_objs else np.empty((0, 0))

        acc.update(gt_ids, pr_ids, dist)

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=str, default=str(OUTPUT_ROOT / "predictions"),
                        help="Directory containing per-sequence prediction .txt files")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    ann_dir = DATASET_ROOT / "annotations"

    mh = mm.metrics.create()
    accs, names = [], []

    for ann_file in sorted(ann_dir.glob("*.txt")):
        seq_name = ann_file.stem
        pred_file = pred_dir / f"{seq_name}.txt"

        gt = load_gt(ann_file)
        preds = load_predictions(pred_file)

        acc = evaluate_sequence(seq_name, gt, preds)
        accs.append(acc)
        names.append(seq_name)
        print(f"  Evaluated: {seq_name}")

    summary = mh.compute_many(
        accs, names=names,
        metrics=["num_frames", "mota", "motp", "idf1", "num_switches", "num_false_positives", "num_misses"],
        generate_overall=True,
    )
    print("\n=== Evaluation Results ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
