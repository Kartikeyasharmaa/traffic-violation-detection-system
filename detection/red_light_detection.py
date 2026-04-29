from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

import cv2
import numpy as np

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


class RedLightViolationDetector:
    def __init__(self, model_path: str, confidence: float, red_duration: int, green_duration: int) -> None:
        self.logger = setup_logger("red_light_detection")
        self.model = load_yolo_model(model_path, self.logger, fallback=settings.default_vehicle_model)
        self.confidence = confidence
        self.red_duration = red_duration
        self.green_duration = green_duration
        self.tracker = CentroidTracker(max_disappeared=60, max_distance=190, history_size=80)
        plate_detector_path = settings.helmet_model_path if Path(settings.helmet_model_path).exists() else None
        self.ocr = OCRProcessor(plate_detector_path=plate_detector_path)
        self.event_gate = ViolationEventGate(cooldown_frames=45)
        self.flagged_tracks: set[int] = set()
        self.flagged_plate_boxes: dict[int, tuple[int, int, int, int]] = {}
        self.signal_state_cache: str | None = None
        self.save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="red_light_writer")

    def _plate_focus_bbox(
        self,
        vehicle_bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, int, int],
    ) -> tuple[int, int, int, int]:
        return relative_bbox(
            vehicle_bbox,
            frame_shape,
            x1_ratio=0.05,
            y1_ratio=0.56,
            x2_ratio=0.95,
            y2_ratio=0.90,
        )

    def _fallback_plate_anchor_x(
        self,
        vehicle_bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, int, int],
    ) -> float:
        frame_width = frame_shape[1]
        x1, _, x2, _ = vehicle_bbox
        if x1 <= 5:
            return 0.28
        if x2 >= frame_width - 5:
            return 0.72
        return 0.50

    def _preview_plate_bbox(
        self,
        source_frame,
        vehicle_bbox: tuple[int, int, int, int],
        track_id: int,
    ) -> tuple[int, int, int, int]:
        cached_bbox = self.flagged_plate_boxes.get(track_id)
        if cached_bbox is not None:
            return cached_bbox

        focus_bbox = self._plate_focus_bbox(vehicle_bbox, source_frame.shape)
        vehicle_crop, (crop_x, crop_y) = crop_with_padding_and_offset(source_frame, focus_bbox, padding=4)
        plate_text, detected_bbox = self.ocr.extract_number_plate_details(vehicle_crop)
        if detected_bbox is not None and plate_text != "UNKNOWN":
            px1, py1, px2, py2 = detected_bbox
            return (crop_x + px1, crop_y + py1, crop_x + px2, crop_y + py2)

        return estimate_plate_bbox(
            vehicle_bbox,
            source_frame.shape,
            width_ratio=0.16,
            height_ratio=0.10,
            x_anchor=self._fallback_plate_anchor_x(vehicle_bbox, source_frame.shape),
            y_anchor=0.78,
        )

    def _resolve_plate_details(
        self,
        source_frame,
        vehicle_bbox: tuple[int, int, int, int],
        fallback_plate_bbox: tuple[int, int, int, int],
    ) -> tuple[str, tuple[int, int, int, int]]:
        focus_bbox = self._plate_focus_bbox(vehicle_bbox, source_frame.shape)
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

    def _signal_state(self, frame_index: int, fps: float) -> str:
        elapsed_seconds = frame_index / fps if fps else 0
        cycle = self.red_duration + self.green_duration
        return "RED" if elapsed_seconds % cycle < self.red_duration else "GREEN"

    def _draw_signal(self, frame, state: str) -> None:
        if state == "RED":
            color = (0, 0, 255)
        elif state == "YELLOW":
            color = (0, 215, 255)
        else:
            color = (0, 200, 0)
        cv2.rectangle(frame, (frame.shape[1] - 170, 20), (frame.shape[1] - 20, 80), color, -1)
        cv2.putText(frame, f"Signal: {state}", (frame.shape[1] - 156, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    def _detect_signal_state_from_frame(self, frame) -> str | None:
        height, width = frame.shape[:2]
        roi_width = max(120, int(width * 0.08))
        roi_height = max(180, int(height * 0.24))
        x2 = max(roi_width + 20, width - 20)
        x1 = max(0, x2 - roi_width)
        y1 = 20
        y2 = min(height, y1 + roi_height)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        top_red_score = 0.0
        middle_yellow_score = 0.0
        bottom_green_score = 0.0
        roi_height, roi_width = roi.shape[:2]
        center_x = roi_width // 2
        radius = max(10, min(roi_width // 4, roi_height // 8))
        centers_y = [int(roi_height * 0.18), int(roi_height * 0.50), int(roi_height * 0.82)]

        for index, center_y in enumerate(centers_y):
            mask = np.zeros((roi_height, roi_width), dtype="uint8")
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            b, g, r, _ = cv2.mean(roi, mask=mask)
            brightness = max(r, g, b)
            if index == 0:
                top_red_score = (r - max(g, b)) if brightness > 70 else 0.0
            elif index == 1:
                if brightness > 70:
                    middle_yellow_score = ((r + g) / 2.0) - b - (abs(r - g) * 0.35)
                else:
                    middle_yellow_score = 0.0
            elif index == 2:
                bottom_green_score = (g - max(r, b)) if brightness > 70 else 0.0

        if top_red_score >= 32 and top_red_score >= middle_yellow_score + 2 and top_red_score >= bottom_green_score + 6:
            return "RED"
        if middle_yellow_score >= 28 and middle_yellow_score >= bottom_green_score + 2:
            return "YELLOW"
        if bottom_green_score >= 35 and bottom_green_score >= top_red_score + 8:
            return "GREEN"
        return None

    def _save_violation_record(
        self,
        record_id: int,
        absolute_image_path: Path,
        frame_snapshot,
        source_frame,
        vehicle_bbox: tuple[int, int, int, int],
        fallback_plate_bbox: tuple[int, int, int, int],
        track_id: int,
    ) -> None:
        plate, final_plate_bbox = self._resolve_plate_details(source_frame, vehicle_bbox, fallback_plate_bbox)
        px1, py1, px2, py2 = final_plate_bbox
        cv2.rectangle(frame_snapshot, (px1, py1), (px2, py2), (50, 220, 255), 2)
        draw_label(frame_snapshot, f"Number Plate | {plate}", (px1, max(30, py1)), (50, 220, 255))
        cv2.imwrite(str(absolute_image_path), frame_snapshot)
        update_violation_number_plate(record_id, plate)
        self.logger.info(
            "Red light violation enriched | record_id=%s | track_id=%s | plate=%s",
            record_id,
            track_id,
            plate,
        )

    def _is_red_light_crossing(
        self,
        bbox_history: list[tuple[int, int, int, int]],
        *,
        stop_line_y: int,
        frame_height: int,
        approach_direction: str,
    ) -> bool:
        if len(bbox_history) < 2:
            return False

        line_margin = max(6, int(frame_height * 0.01))
        zone_margin = max(12, int(frame_height * 0.05))

        if approach_direction == "top_to_bottom":
            bottom_edges = [box[3] for box in bbox_history]
            centers = [int((box[1] + box[3]) / 2) for box in bbox_history]
            current_edge = bottom_edges[-1]
            current_center = centers[-1]
            previous_boxes = bbox_history[:-1]
            was_before_line = any(edge <= stop_line_y - line_margin for edge in bottom_edges[:-1])
            crossed_line_visibly = any(box[1] <= stop_line_y <= box[3] + line_margin for box in previous_boxes)
            entered_from_nearby = min(bottom_edges[:-1]) <= stop_line_y + zone_margin if previous_boxes else False
            moving_forward = current_edge - min(bottom_edges[:-1]) > line_margin if previous_boxes else False
            beyond_line_now = current_edge >= stop_line_y + line_margin or current_center >= stop_line_y
            return beyond_line_now and moving_forward and (was_before_line or crossed_line_visibly or entered_from_nearby)

        top_edges = [box[1] for box in bbox_history]
        centers = [int((box[1] + box[3]) / 2) for box in bbox_history]
        current_edge = top_edges[-1]
        current_center = centers[-1]
        previous_boxes = bbox_history[:-1]
        was_before_line = any(edge >= stop_line_y + line_margin for edge in top_edges[:-1])
        crossed_line_visibly = any(box[1] - line_margin <= stop_line_y <= box[3] for box in previous_boxes)
        entered_from_nearby = max(top_edges[:-1]) >= stop_line_y - zone_margin if previous_boxes else False
        moving_forward = max(top_edges[:-1]) - current_edge > line_margin if previous_boxes else False
        beyond_line_now = current_edge <= stop_line_y - line_margin or current_center <= stop_line_y
        return beyond_line_now and moving_forward and (was_before_line or crossed_line_visibly or entered_from_nearby)

    def run(
        self,
        video_path: Union[str, int],
        line_ratio: float,
        approach_direction: str,
        show: bool,
        output_path: str | None = None,
    ) -> None:
        capture = open_video_capture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_manager = FrameOutputManager(
            video_path=str(video_path),
            violation_type="red_light",
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
        inference_max_width = 384 if show else 1180
        inference_imgsz = 288 if show else 896
        source_frame_skip = 2 if fast_preview_mode else (1 if show else 0)

        while True:
            success, frame = capture.read()
            if not success:
                break

            original_frame = frame.copy()
            frame_height, frame_width = frame.shape[:2]
            stop_line_y = int(frame_height * line_ratio)
            should_refresh_signal = not show or frame_index % 2 == 0
            detected_signal_state = self._detect_signal_state_from_frame(original_frame) if should_refresh_signal else None
            if detected_signal_state is not None:
                self.signal_state_cache = detected_signal_state
            signal_state = detected_signal_state or self.signal_state_cache or self._signal_state(frame_index, fps)

            inference_frame, scale = prepare_frame_for_inference(frame, max_width=inference_max_width)
            result = self.model(
                inference_frame,
                verbose=False,
                imgsz=inference_imgsz,
                classes=settings.vehicle_classes,
            )[0]
            detections = result_to_detections(result, settings.vehicle_classes, self.confidence)
            detections = scale_detections(detections, scale)
            tracks = self.tracker.update(detections)

            if signal_state == "RED":
                line_color = (0, 0, 255)
            elif signal_state == "YELLOW":
                line_color = (0, 215, 255)
            else:
                line_color = (0, 200, 0)
            cv2.line(frame, (40, stop_line_y), (frame_width - 40, stop_line_y), line_color, 3)
            self._draw_signal(frame, signal_state)

            for track in tracks.values():
                if track.get("missing_frames", 0) > 0:
                    continue

                x1, y1, x2, y2 = track["bbox"]
                bbox_history = list(track.get("bbox_history", []))

                crossed_line = self._is_red_light_crossing(
                    bbox_history,
                    stop_line_y=stop_line_y,
                    frame_height=frame_height,
                    approach_direction=approach_direction,
                )
                is_flagged = track["track_id"] in self.flagged_tracks
                plate_display_bbox = None

                color = (0, 0, 255) if is_flagged else (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(
                    frame,
                    "RED LIGHT" if is_flagged else f"Vehicle ID {track['track_id']}",
                    (x1, max(30, y1)),
                    color,
                )
                if is_flagged:
                    plate_display_bbox = self._preview_plate_bbox(
                        original_frame,
                        track["bbox"],
                        track["track_id"],
                    )
                    px1, py1, px2, py2 = plate_display_bbox
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (50, 220, 255), 2)
                    draw_label(frame, "Number Plate", (px1, max(30, py1)), (50, 220, 255))

                if len(bbox_history) < 2:
                    continue

                if signal_state != "RED" or not crossed_line or is_flagged:
                    continue

                if self.event_gate.should_skip(
                    frame_index=frame_index,
                    bbox=track["bbox"],
                    number_plate=None,
                    track_id=track["track_id"],
                    spatial_matching=False,
                ):
                    continue

                if plate_display_bbox is None:
                    plate_display_bbox = self._preview_plate_bbox(
                        original_frame,
                        track["bbox"],
                        track["track_id"],
                    )
                px1, py1, px2, py2 = plate_display_bbox
                cv2.rectangle(frame, (px1, py1), (px2, py2), (50, 220, 255), 2)
                draw_label(frame, "Number Plate", (px1, max(30, py1)), (50, 220, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                draw_label(frame, "RED LIGHT", (x1, max(30, y1)), (0, 0, 255))
                self.flagged_tracks.add(track["track_id"])
                self.flagged_plate_boxes[track["track_id"]] = plate_display_bbox
                violation_frame = frame.copy()
                image_path, absolute_image_path = build_violation_image_path("red_light")
                record_id = persist_violation("red_light", "UNKNOWN", image_path)
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
                    track["track_id"],
                )
                violations += 1
                self.logger.info("Red light violation saved | record_id=%s | track_id=%s", record_id, track["track_id"])

            cv2.putText(
                frame,
                f"Red Light Violations: {violations}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
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
        self.logger.info("Processing complete. Total red-light violations: %s", violations)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect red light jumping vehicles from a video.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--model",
        default=str(settings.vehicle_model_path),
        help="Path to the YOLO vehicle model. Defaults to models/yolov8n.pt if present.",
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument(
        "--line-y",
        type=float,
        default=settings.red_light_default_line_y,
        help="Stop-line position as a fraction of frame height.",
    )
    parser.add_argument(
        "--approach-direction",
        choices=["top_to_bottom", "bottom_to_top"],
        default=settings.red_light_default_approach_direction,
        help="Expected vehicle movement direction toward the stop line.",
    )
    parser.add_argument(
        "--red-duration",
        type=int,
        default=settings.red_light_default_red_duration,
        help="Duration of the simulated red signal in seconds.",
    )
    parser.add_argument(
        "--green-duration",
        type=int,
        default=settings.red_light_default_green_duration,
        help="Duration of the simulated green signal in seconds.",
    )
    parser.add_argument("--show", action="store_true", help="Display the live processed video window.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the annotated output video. Defaults to outputs/red_light/.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    detector = RedLightViolationDetector(
        model_path=arguments.model,
        confidence=arguments.confidence,
        red_duration=arguments.red_duration,
        green_duration=arguments.green_duration,
    )
    detector.run(
        video_path=arguments.video,
        line_ratio=arguments.line_y,
        approach_direction=arguments.approach_direction,
        show=arguments.show,
        output_path=arguments.output,
    )
