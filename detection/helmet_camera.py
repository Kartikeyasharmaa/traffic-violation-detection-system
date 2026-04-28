from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from detection.helmet_detection import HelmetViolationDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run helmet violation detection on a live camera feed.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default is 0.")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="Display the live processed video window.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument(
        "--vehicle-model",
        default=str(settings.vehicle_model_path),
        help="Path to the YOLO vehicle model.",
    )
    parser.add_argument(
        "--helmet-model",
        default=str(settings.helmet_model_path),
        help="Path to the custom helmet detection model.",
    )
    parser.add_argument("--output", default=None, help="Optional path for the annotated output video.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    detector = HelmetViolationDetector(
        vehicle_model_path=args.vehicle_model,
        helmet_model_path=args.helmet_model,
        confidence=args.confidence,
    )
    detector.run(video_path=args.camera, show=args.show, output_path=args.output)
