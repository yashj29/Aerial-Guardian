"""
Aerial Guardian - Entry point.

Usage:
  # Process all validation sequences
  python run.py

  # Process a single sequence
  python run.py --seq uav0000086_00000_v

  # Disable SAHI (faster, lower recall on small objects)
  python run.py --no-sahi

  # Use CPU explicitly
  python run.py --device cpu
"""

import argparse
import yaml
from pathlib import Path

from src.pipeline import AerialGuardianPipeline

DATASET_ROOT = Path(__file__).resolve().parent.parent / "VisDrone2019-MOT-val"
OUTPUT_ROOT = Path(__file__).parent / "outputs"
CONFIG_PATH = Path(__file__).parent / "configs" / "config.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Aerial Guardian: Drone Person Tracker")
    parser.add_argument("--seq", type=str, default=None,
                        help="Single sequence name (e.g. uav0000086_00000_v). Runs all if omitted.")
    parser.add_argument("--no-sahi", action="store_true",
                        help="Disable SAHI sliced inference (faster, worse on small objects)")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: 'cpu' or 'cuda:0'")
    parser.add_argument("--tracker", type=str, default=None, choices=["botsort", "bytetrack"],
                        help="Override tracker type")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)

    if args.no_sahi:
        cfg["detector"]["use_sahi"] = False
    if args.device:
        cfg["detector"]["device"] = args.device
        cfg["tracker"]["device"] = args.device
    if args.tracker:
        cfg["tracker"]["type"] = args.tracker

    pipeline = AerialGuardianPipeline(
        det_cfg=cfg["detector"],
        trk_cfg=cfg["tracker"],
        vis_cfg=cfg["visualizer"],
        out_cfg=cfg["output"],
    )

    seq_root = DATASET_ROOT / "sequences"
    if args.seq:
        sequences = [seq_root / args.seq]
    else:
        sequences = sorted(seq_root.iterdir())

    all_results = {}
    for seq_dir in sequences:
        if not seq_dir.is_dir():
            continue
        out_path = OUTPUT_ROOT / f"{seq_dir.name}.mp4"
        print(f"\nProcessing: {seq_dir.name}")
        result = pipeline.run_sequence(seq_dir, out_path)
        all_results[seq_dir.name] = result

    print("\n=== Summary ===")
    total_fps = []
    for name, r in all_results.items():
        print(f"  {name}: {r['frames']} frames @ {r['avg_fps']:.1f} FPS → {r['output']}")
        total_fps.append(r["avg_fps"])

    if total_fps:
        import numpy as np
        print(f"\n  Overall avg FPS: {np.mean(total_fps):.2f}")
    print(f"\nOutput videos saved to: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
