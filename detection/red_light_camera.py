from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from detection.red_light_detection import RedLightViolationDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run red-light violation detection on a live camera feed.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default is 0.")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="Display the live processed video window.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument("--line-y", type=float, default=settings.red_light_default_line_y, help="Stop-line position as a fraction of frame height.")
    parser.add_argument(
        "--approach-direction",
        choices=["top_to_bottom", "bottom_to_top"],
        default=settings.red_light_default_approach_direction,
        help="Expected vehicle movement direction toward the stop line.",
    )
    parser.add_argument("--red-duration", type=int, default=settings.red_light_default_red_duration, help="Red signal duration in seconds.")
    parser.add_argument("--green-duration", type=int, default=settings.red_light_default_green_duration, help="Green signal duration in seconds.")
    parser.add_argument("--model", default=str(settings.vehicle_model_path), help="Path to the YOLO vehicle model.")
    parser.add_argument("--output", default=None, help="Optional path for the annotated output video.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    detector = RedLightViolationDetector(
        model_path=args.model,
        confidence=args.confidence,
        red_duration=args.red_duration,
        green_duration=args.green_duration,
    )
    detector.run(
        video_path=args.camera,
        line_ratio=args.line_y,
        approach_direction=args.approach_direction,
        show=args.show,
        output_path=args.output,
    )
