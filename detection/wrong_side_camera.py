from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from detection.wrong_side_detection import WrongSideViolationDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run wrong-side violation detection on a live camera feed.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default is 0.")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="Display the live processed video window.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument(
        "--allowed-direction",
        choices=["ltr", "rtl", "ttb", "btt"],
        default=settings.wrong_side_default_direction,
        help="Direction that is considered legal traffic flow.",
    )
    parser.add_argument("--min-displacement", type=int, default=settings.wrong_side_default_min_displacement, help="Minimum movement in pixels before a track is evaluated.")
    parser.add_argument("--model", default=str(settings.vehicle_model_path), help="Path to the YOLO vehicle model.")
    parser.add_argument("--output", default=None, help="Optional path for the annotated output video.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    detector = WrongSideViolationDetector(
        model_path=args.model,
        confidence=args.confidence,
        min_displacement=args.min_displacement,
    )
    detector.run(
        video_path=args.camera,
        allowed_direction=args.allowed_direction,
        show=args.show,
        output_path=args.output,
    )
