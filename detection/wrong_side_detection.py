from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from detection.ocr import OCRProcessor
from detection.utils import (
    CentroidTracker,
    FrameOutputManager,
    ViolationEventGate,
    build_violation_image_path,
    crop_with_padding_and_offset,
    draw_direction_guides,
    draw_label,
    estimate_plate_bbox,
    load_yolo_model,
    open_video_capture,
    prepare_frame_for_inference,
    persist_violation,
    relative_bbox,
    result_to_detections,
    scale_detections,
    setup_logger,
    update_violation_number_plate,
)


class WrongSideViolationDetector:
    def __init__(self, model_path: str, confidence: float, min_displacement: int) -> None:
        self.logger = setup_logger("wrong_side_detection")
        self.model = load_yolo_model(model_path, self.logger, fallback=settings.default_vehicle_model)
        self.confidence = confidence
        self.min_displacement = min_displacement
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=140, history_size=45)
        plate_detector_path = settings.helmet_model_path if Path(settings.helmet_model_path).exists() else None
        self.ocr = OCRProcessor(plate_detector_path=plate_detector_path)
        self.event_gate = ViolationEventGate(cooldown_frames=120)
        self.flagged_tracks: set[int] = set()
        self.flagged_plate_boxes: dict[int, tuple[int, int, int, int]] = {}
        self.save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wrong_side_writer")

    def _plate_anchor_x(self, history: list[tuple[int, int]]) -> float:
        if len(history) < 2:
            return 0.50

        dx = history[-1][0] - history[0][0]
        if abs(dx) < max(18, self.min_displacement // 3):
            return 0.50

        return 0.22 if dx < 0 else 0.78

    def _plate_focus_bbox(
        self,
        vehicle_bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, int, int],
        history: list[tuple[int, int]],
    ) -> tuple[int, int, int, int]:
        anchor_x = self._plate_anchor_x(history)
        if anchor_x < 0.35:
            return relative_bbox(
                vehicle_bbox,
                frame_shape,
                x1_ratio=0.00,
                y1_ratio=0.54,
                x2_ratio=0.38,
                y2_ratio=0.86,
            )
        if anchor_x > 0.65:
            return relative_bbox(
                vehicle_bbox,
                frame_shape,
                x1_ratio=0.62,
                y1_ratio=0.54,
                x2_ratio=1.00,
                y2_ratio=0.86,
            )
        return relative_bbox(
            vehicle_bbox,
            frame_shape,
            x1_ratio=0.18,
            y1_ratio=0.54,
            x2_ratio=0.82,
            y2_ratio=0.86,
        )

    def _preview_plate_bbox(
        self,
        source_frame,
        vehicle_bbox: tuple[int, int, int, int],
        track_id: int,
        history: list[tuple[int, int]],
    ) -> tuple[int, int, int, int]:
        cached_bbox = self.flagged_plate_boxes.get(track_id)
        if cached_bbox is not None:
            return cached_bbox

        focus_bbox = self._plate_focus_bbox(vehicle_bbox, source_frame.shape, history)
        vehicle_crop, (crop_x, crop_y) = crop_with_padding_and_offset(source_frame, focus_bbox, padding=4)
        plate_text, detected_bbox = self.ocr.extract_number_plate_details(vehicle_crop)
        if detected_bbox is not None and plate_text != "UNKNOWN":
            px1, py1, px2, py2 = detected_bbox
            return (crop_x + px1, crop_y + py1, crop_x + px2, crop_y + py2)

        return estimate_plate_bbox(
            vehicle_bbox,
            source_frame.shape,
            width_ratio=0.20,
            height_ratio=0.09,
            x_anchor=self._plate_anchor_x(history),
            y_anchor=0.74,
        )

    def _resolve_plate_details(
        self,
        source_frame,
        vehicle_bbox: tuple[int, int, int, int],
        fallback_plate_bbox: tuple[int, int, int, int],
        history: list[tuple[int, int]],
    ) -> tuple[str, tuple[int, int, int, int]]:
        focus_bbox = self._plate_focus_bbox(vehicle_bbox, source_frame.shape, history)
        vehicle_crop, (crop_x, crop_y) = crop_with_padding_and_offset(source_frame, focus_bbox, padding=4)
        detected_bbox = self.ocr.find_number_plate_bbox(vehicle_crop)
        plate, plate_bbox = self.ocr.extract_number_plate_details(vehicle_crop)
        final_plate_bbox = fallback_plate_bbox
        if detected_bbox is not None:
            px1, py1, px2, py2 = detected_bbox
            final_plate_bbox = (crop_x + px1, crop_y + py1, crop_x + px2, crop_y + py2)
        if plate_bbox is not None:
            px1, py1, px2, py2 = plate_bbox
            final_plate_bbox = (crop_x + px1, crop_y + py1, crop_x + px2, crop_y + py2)
        return plate, final_plate_bbox

    def _is_wrong_direction(self, history: list[tuple[int, int]], allowed_direction: str) -> bool:
        start_x, start_y = history[0]
        end_x, end_y = history[-1]
        dx = end_x - start_x
        dy = end_y - start_y

        if allowed_direction in {"ltr", "rtl"}:
            if abs(dx) < self.min_displacement or abs(dx) < abs(dy):
                return False
            return dx < -self.min_displacement if allowed_direction == "ltr" else dx > self.min_displacement

        if abs(dy) < self.min_displacement or abs(dy) < abs(dx):
            return False
        return dy < -self.min_displacement if allowed_direction == "ttb" else dy > self.min_displacement

    def run(self, video_path: Union[str, int], allowed_direction: str, show: bool, output_path: str | None = None) -> None:
        capture = open_video_capture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_manager = FrameOutputManager(
            video_path=str(video_path),
            violation_type="wrong_side",
            fps=fps,
            frame_size=(frame_width, frame_height),
            show=show,
            logger=self.logger,
            output_path=output_path,
        )
        frame_index = 0
        violations = 0
        is_camera_input = isinstance(video_path, int)
        fast_preview_mode = show and not is_camera_input
        inference_max_width = 352 if show else 960
        inference_imgsz = 288 if show else 736
        source_frame_skip = 3 if fast_preview_mode else (2 if is_camera_input and show else 0)

        while True:
            success, frame = capture.read()
            if not success:
                break

            original_frame = frame.copy()
            inference_frame, scale = prepare_frame_for_inference(original_frame, max_width=inference_max_width)
            result = self.model(
                inference_frame,
                verbose=False,
                imgsz=inference_imgsz,
                classes=settings.vehicle_classes,
            )[0]
            frame = original_frame.copy()
            draw_direction_guides(frame, allowed_direction)
            detections = result_to_detections(result, settings.vehicle_classes, self.confidence)
            detections = scale_detections(detections, scale)
            tracks = self.tracker.update(detections)

            for track in tracks.values():
                if track.get("missing_frames", 0) > 0:
                    continue

                x1, y1, x2, y2 = track["bbox"]
                history = list(track["history"])
                is_flagged = track["track_id"] in self.flagged_tracks
                if len(history) < 2:
                    box_color = (0, 180, 255)
                    wrong_direction = False
                else:
                    wrong_direction = self._is_wrong_direction(history, allowed_direction) if len(history) >= 4 else False
                    box_color = (0, 0, 255) if (wrong_direction or is_flagged) else (0, 180, 255)

                plate_display_bbox = None

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                draw_label(
                    frame,
                    "WRONG SIDE" if (wrong_direction or is_flagged) else f"Vehicle ID {track['track_id']}",
                    (x1, max(30, y1)),
                    box_color,
                )
                if len(history) >= 2:
                    cv2.arrowedLine(frame, history[0], history[-1], box_color, 2, tipLength=0.25)
                if wrong_direction or is_flagged:
                    plate_display_bbox = self._preview_plate_bbox(
                        original_frame,
                        track["bbox"],
                        track["track_id"],
                        history,
                    )
                    px1, py1, px2, py2 = plate_display_bbox
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (50, 220, 255), 2)
                    draw_label(frame, "Number Plate", (px1, max(30, py1)), (50, 220, 255))

                if not wrong_direction or is_flagged:
                    continue

                if self.event_gate.should_skip(
                    frame_index=frame_index,
                    bbox=track["bbox"],
                    number_plate=None,
                    track_id=track["track_id"],
                ):
                    continue

                if plate_display_bbox is None:
                    plate_display_bbox = self._preview_plate_bbox(
                        original_frame,
                        track["bbox"],
                        track["track_id"],
                        history,
                    )
                violation_frame = frame.copy()
                image_path, absolute_image_path = build_violation_image_path("wrong_side")
                record_id = persist_violation("wrong_side", "UNKNOWN", image_path)
                self.flagged_tracks.add(track["track_id"])
                self.flagged_plate_boxes[track["track_id"]] = plate_display_bbox
                self.event_gate.record(
                    frame_index=frame_index,
                    bbox=track["bbox"],
                    number_plate=None,
                    track_id=track["track_id"],
                )
                self.save_executor.submit(
                    self._save_violation_record,
                    record_id,
                    absolute_image_path,
                    violation_frame,
                    original_frame.copy(),
                    track["bbox"],
                    plate_display_bbox,
                    history,
                    track["track_id"],
                )
                violations += 1
                self.logger.info("Wrong-side violation saved | record_id=%s | track_id=%s", record_id, track["track_id"])

            cv2.putText(
                frame,
                f"Wrong-Side Violations: {violations}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )

            if output_manager.handle_frame(frame):
                break

            frame_index += 1
            for _ in range(source_frame_skip):
                if not capture.grab():
                    break
                frame_index += 1

        capture.release()
        self.save_executor.shutdown(wait=True)
        output_manager.close()
        self.logger.info("Processing complete. Total wrong-side violations: %s", violations)

    def _save_violation_record(
        self,
        record_id: int,
        absolute_image_path: Path,
        frame_snapshot,
        source_frame,
        vehicle_bbox: tuple[int, int, int, int],
        fallback_plate_bbox: tuple[int, int, int, int],
        history: list[tuple[int, int]],
        track_id: int,
    ) -> None:
        plate, final_plate_bbox = self._resolve_plate_details(source_frame, vehicle_bbox, fallback_plate_bbox, history)
        px1, py1, px2, py2 = final_plate_bbox
        cv2.rectangle(frame_snapshot, (px1, py1), (px2, py2), (50, 220, 255), 2)
        draw_label(frame_snapshot, f"Number Plate | {plate}", (px1, max(30, py1)), (50, 220, 255))
        cv2.imwrite(str(absolute_image_path), frame_snapshot)
        update_violation_number_plate(record_id, plate)
        self.logger.info(
            "Wrong-side violation enriched | record_id=%s | track_id=%s | plate=%s",
            record_id,
            track_id,
            plate,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect wrong-side driving from a video.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--model",
        default=str(settings.vehicle_model_path),
        help="Path to the YOLO vehicle model. Defaults to models/yolov8n.pt if present.",
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument(
        "--allowed-direction",
        choices=["ltr", "rtl", "ttb", "btt"],
        default=settings.wrong_side_default_direction,
        help="Direction that is considered legal traffic flow.",
    )
    parser.add_argument(
        "--min-displacement",
        type=int,
        default=settings.wrong_side_default_min_displacement,
        help="Minimum movement in pixels before a track is evaluated.",
    )
    parser.add_argument("--show", action="store_true", help="Display the live processed video window.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the annotated output video. Defaults to outputs/wrong_side/.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    detector = WrongSideViolationDetector(
        model_path=arguments.model,
        confidence=arguments.confidence,
        min_displacement=arguments.min_displacement,
    )
    detector.run(
        video_path=arguments.video,
        allowed_direction=arguments.allowed_direction,
        show=arguments.show,
        output_path=arguments.output,
    )
