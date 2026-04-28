from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union

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
    bbox_center,
    crop_with_padding,
    crop_with_padding_and_offset,
    draw_label,
    estimate_plate_bbox,
    load_yolo_model,
    open_video_capture,
    overlap_ratio,
    prepare_frame_for_inference,
    persist_violation,
    point_in_bbox,
    result_to_detections,
    save_violation_image,
    scale_detections,
    setup_logger,
    update_violation_number_plate,
)


class HelmetViolationDetector:
    def __init__(
        self,
        vehicle_model_path: str,
        helmet_model_path: Optional[str],
        confidence: float,
    ) -> None:
        self.logger = setup_logger("helmet_detection")
        self.vehicle_model = load_yolo_model(vehicle_model_path, self.logger, fallback=settings.default_vehicle_model)
        self.helmet_model = self._load_helmet_model(helmet_model_path)
        self.confidence = confidence
        self.ocr = OCRProcessor(plate_detector=self.helmet_model)
        self.tracker = CentroidTracker(max_disappeared=15, max_distance=100, history_size=20)
        self.event_gate = ViolationEventGate(cooldown_frames=120)
        self.flagged_tracks: set[int] = set()
        self.save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="helmet_writer")
        self.helmet_class_ids, self.no_helmet_class_ids = self._resolve_helmet_classes()
        self.plate_class_ids, self.rider_class_ids = self._resolve_auxiliary_classes()
        self.face_cascade = self._load_face_cascade()

    def _load_helmet_model(self, model_path: Optional[str]):
        if model_path and Path(model_path).exists():
            return load_yolo_model(model_path, self.logger)
        self.logger.warning("Helmet model not found. Falling back to heuristic head-shape checks.")
        return None

    def _resolve_helmet_classes(self) -> tuple[set[int], set[int]]:
        if self.helmet_model is None:
            return set(), set()

        names = self.helmet_model.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        helmet_classes: set[int] = set()
        no_helmet_classes: set[int] = set()

        for class_id, label in items:
            normalized = str(label).lower().replace("-", "_").replace(" ", "_")
            if "helmet" not in normalized:
                continue
            if any(token in normalized for token in ("no_helmet", "without_helmet", "nohelmet", "withouthelmet")):
                no_helmet_classes.add(int(class_id))
            else:
                helmet_classes.add(int(class_id))

        return helmet_classes, no_helmet_classes

    def _resolve_auxiliary_classes(self) -> tuple[set[int], set[int]]:
        if self.helmet_model is None:
            return set(), set()

        names = self.helmet_model.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        plate_classes: set[int] = set()
        rider_classes: set[int] = set()
        for class_id, label in items:
            normalized = str(label).lower().replace("-", "_").replace(" ", "_")
            if "plate" in normalized:
                plate_classes.add(int(class_id))
            if "rider" in normalized:
                rider_classes.add(int(class_id))
        return plate_classes, rider_classes

    def _load_face_cascade(self):
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return None
        cascade = cv2.CascadeClassifier(str(cascade_path))
        return cascade if not cascade.empty() else None

    def _head_region(self, rider_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = rider_bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        inset_x = int(width * 0.08)
        head_height = max(18, int(height * 0.28))
        return (x1 + inset_x, y1, x2 - inset_x, y1 + head_height)

    def _find_detection_in_region(
        self,
        region_bbox: tuple[int, int, int, int],
        detections: list[dict],
    ) -> Optional[tuple[int, int, int, int]]:
        best_match: Optional[tuple[float, tuple[int, int, int, int]]] = None

        for detection in detections:
            detection_bbox = detection["bbox"]
            center = bbox_center(detection_bbox)
            overlap_score = overlap_ratio(detection_bbox, region_bbox)
            if not point_in_bbox(center, region_bbox) and overlap_score < 0.10:
                continue

            score = float(detection.get("confidence", 0.0)) + overlap_score
            if best_match is None or score > best_match[0]:
                best_match = (score, detection_bbox)

        return best_match[1] if best_match is not None else None

    def _heuristic_helmet_present(self, frame, rider_bbox: tuple[int, int, int, int]) -> bool:
        head_crop = crop_with_padding(frame, self._head_region(rider_bbox), padding=4)
        if head_crop.size == 0:
            return False

        gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        min_radius = max(5, min(gray.shape[:2]) // 8)
        max_radius = max(min_radius + 4, min(gray.shape[:2]) // 2)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(10, gray.shape[1] // 3),
            param1=60,
            param2=18,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        return circles is not None

    def _rider_has_helmet(self, frame, rider_bbox: tuple[int, int, int, int], helmet_detections: list[dict]) -> bool:
        head_region = self._head_region(rider_bbox)
        if self._find_detection_in_region(head_region, helmet_detections) is not None:
            return True

        if self.helmet_model is None:
            return self._heuristic_helmet_present(frame, rider_bbox)

        return False

    def _match_riders_to_bike(
        self,
        riders: list[dict],
        bike_bbox: tuple[int, int, int, int],
    ) -> list[dict]:
        bx1, by1, bx2, by2 = bike_bbox
        matched: list[dict] = []
        for rider in riders:
            rx1, ry1, rx2, ry2 = rider["bbox"]
            rider_center = bbox_center(rider["bbox"])
            horizontal_overlap = max(0, min(bx2, rx2) - max(bx1, rx1))
            rider_width = max(1, rx2 - rx1)
            if horizontal_overlap / rider_width < 0.30:
                continue
            if ry2 < by1 - 20 or ry1 > by2 + 20:
                continue
            if not (bx1 - 40 <= rider_center[0] <= bx2 + 40):
                continue
            matched.append(rider)

        matched.sort(key=lambda item: item["bbox"][0])
        return matched

    def _match_plate_to_bike(
        self,
        plates: list[dict],
        bike_bbox: tuple[int, int, int, int],
    ) -> Optional[dict]:
        bx1, by1, bx2, by2 = bike_bbox
        best_match: Optional[tuple[float, dict]] = None

        for plate in plates:
            px1, py1, px2, py2 = plate["bbox"]
            center = bbox_center(plate["bbox"])
            inside_expanded = bx1 - 30 <= center[0] <= bx2 + 30 and by1 - 20 <= center[1] <= by2 + 40
            if not inside_expanded:
                continue

            score = overlap_ratio(plate["bbox"], bike_bbox) + (center[1] / max(by2 + 1, 1))
            if best_match is None or score > best_match[0]:
                best_match = (score, plate)

        return best_match[1] if best_match is not None else None

    def _detect_face_heads(
        self,
        frame,
        bike_bbox: tuple[int, int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        if self.face_cascade is None:
            return []

        height, width = frame.shape[:2]
        bx1, by1, bx2, by2 = bike_bbox
        bike_width = max(1, bx2 - bx1)
        bike_height = max(1, by2 - by1)

        region_x1 = max(0, bx1 - int(bike_width * 0.10))
        region_y1 = max(0, by1 - int(bike_height * 0.45))
        region_x2 = min(width, bx2 + int(bike_width * 0.10))
        region_y2 = min(height, by1 + int(bike_height * 0.22))
        region = frame[region_y1:region_y2, region_x1:region_x2]
        if region.size == 0:
            return []

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(16, 16),
        )

        face_boxes: list[tuple[int, int, int, int]] = []
        for x, y, w, h in faces:
            face_boxes.append(
                (
                    region_x1 + x,
                    region_y1 + y,
                    region_x1 + x + w,
                    region_y1 + y + h,
                )
            )

        face_boxes.sort(key=lambda bbox: bbox_center(bbox)[0])
        return self._dedupe_bboxes(face_boxes)

    def _dedupe_bboxes(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        unique: list[tuple[int, int, int, int]] = []
        for bbox in boxes:
            if any(overlap_ratio(bbox, existing) >= 0.45 or overlap_ratio(existing, bbox) >= 0.45 for existing in unique):
                continue
            unique.append(bbox)
        return unique

    def _match_head_detections_to_bike(
        self,
        head_detections: list[dict],
        bike_bbox: tuple[int, int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        bx1, by1, bx2, by2 = bike_bbox
        bike_width = max(1, bx2 - bx1)
        bike_height = max(1, by2 - by1)

        matched: list[tuple[int, int, int, int]] = []
        for detection in head_detections:
            head_bbox = detection["bbox"]
            center_x, center_y = bbox_center(head_bbox)
            within_x = bx1 - int(bike_width * 0.18) <= center_x <= bx2 + int(bike_width * 0.18)
            within_y = by1 - int(bike_height * 0.65) <= center_y <= by1 + int(bike_height * 0.30)
            if within_x and within_y:
                matched.append(head_bbox)

        matched.sort(key=lambda bbox: bbox_center(bbox)[0])
        return self._dedupe_bboxes(matched)

    def _merge_rider_candidates(self, rider_detections: list[dict], person_detections: list[dict]) -> list[dict]:
        merged: list[dict] = []
        for detection in rider_detections + person_detections:
            bbox = detection["bbox"]
            if any(overlap_ratio(bbox, existing["bbox"]) >= 0.55 or overlap_ratio(existing["bbox"], bbox) >= 0.55 for existing in merged):
                continue
            merged.append(detection)

        merged.sort(key=lambda item: item["bbox"][0])
        return merged

    def _score_plate_text(self, plate: str) -> float:
        if not plate or plate == "UNKNOWN":
            return -1.0
        cleaned = "".join(char for char in plate.upper() if char.isalnum())
        if len(cleaned) < 5:
            return -1.0

        letters = sum(char.isalpha() for char in cleaned)
        digits = sum(char.isdigit() for char in cleaned)
        score = float(len(cleaned))
        if letters >= 2:
            score += 2.0
        if digits >= 2:
            score += 2.0
        score -= abs(len(cleaned) - 8) * 0.25
        return score

    def _choose_best_plate(self, candidates: list[str]) -> str:
        best_plate = "UNKNOWN"
        best_score = -1.0

        for candidate in candidates:
            score = self._score_plate_text(candidate)
            if score > best_score:
                best_score = score
                best_plate = candidate

        return best_plate

    def _expand_bbox(
        self,
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, int, int],
        padding_x: int = 6,
        padding_y: int = 4,
    ) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        return (
            max(0, x1 - padding_x),
            max(0, y1 - padding_y),
            min(width, x2 + padding_x),
            min(height, y2 + padding_y),
        )

    def _combine_bboxes(
        self,
        boxes: list[tuple[int, int, int, int]],
        frame_shape: tuple[int, int, int],
        padding_x: int = 14,
        padding_y: int = 14,
    ) -> Optional[tuple[int, int, int, int]]:
        valid_boxes = [bbox for bbox in boxes if bbox is not None]
        if not valid_boxes:
            return None

        x1 = min(bbox[0] for bbox in valid_boxes)
        y1 = min(bbox[1] for bbox in valid_boxes)
        x2 = max(bbox[2] for bbox in valid_boxes)
        y2 = max(bbox[3] for bbox in valid_boxes)
        return self._expand_bbox((x1, y1, x2, y2), frame_shape, padding_x=padding_x, padding_y=padding_y)

    def _resolve_plate_for_bike(
        self,
        frame,
        bike_bbox: tuple[int, int, int, int],
        matched_plate_bbox: Optional[tuple[int, int, int, int]],
    ) -> tuple[str, Optional[tuple[int, int, int, int]]]:
        plate_candidates: list[str] = []
        absolute_bbox: Optional[tuple[int, int, int, int]] = None

        if matched_plate_bbox is not None:
            absolute_bbox = matched_plate_bbox
            for candidate_bbox in (
                absolute_bbox,
                self._expand_bbox(absolute_bbox, frame.shape, padding_x=10, padding_y=6),
            ):
                x1, y1, x2, y2 = candidate_bbox
                crop = frame[y1:y2, x1:x2]
                plate_text, _ = self.ocr.extract_number_plate_details(crop)
                plate_candidates.append(plate_text)

        bike_crop, (crop_x, crop_y) = crop_with_padding_and_offset(frame, bike_bbox, padding=16)
        plate_text, plate_bbox = self.ocr.extract_number_plate_details(bike_crop)
        plate_candidates.append(plate_text)
        if absolute_bbox is None and plate_bbox is not None:
            px1, py1, px2, py2 = plate_bbox
            absolute_bbox = (crop_x + px1, crop_y + py1, crop_x + px2, crop_y + py2)

        return self._choose_best_plate(plate_candidates), absolute_bbox

    def run(self, video_path: Union[str, int], show: bool, output_path: Optional[str] = None) -> None:
        capture = open_video_capture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_manager = FrameOutputManager(
            video_path=str(video_path),
            violation_type="helmet",
            fps=fps,
            frame_size=(frame_width, frame_height),
            show=show,
            logger=self.logger,
            output_path=output_path,
        )

        frame_index = 0
        violations = 0
        is_camera_input = isinstance(video_path, int)
        preview_mode = show
        fast_preview_mode = preview_mode and not is_camera_input
        use_person_detector = not (self.helmet_model is not None and self.rider_class_ids)
        vehicle_classes = [settings.motorcycle_class] if not use_person_detector else [settings.person_class, settings.motorcycle_class]
        inference_max_width = 400 if fast_preview_mode else (480 if preview_mode else 880)
        inference_imgsz = 384 if fast_preview_mode else (448 if preview_mode else 768)
        source_frame_skip = 5 if fast_preview_mode else (1 if is_camera_input and preview_mode else 0)

        while True:
            success, frame = capture.read()
            if not success:
                break

            original_frame = frame.copy()
            inference_frame, scale = prepare_frame_for_inference(frame, max_width=inference_max_width)
            result = self.vehicle_model(
                inference_frame,
                verbose=False,
                imgsz=inference_imgsz,
                classes=vehicle_classes,
            )[0]
            detections = result_to_detections(result, min_confidence=self.confidence)
            detections = scale_detections(detections, scale)
            persons = [item for item in detections if item["class_id"] == settings.person_class] if use_person_detector else []
            motorcycles = [item for item in detections if item["class_id"] == settings.motorcycle_class]

            tracked_motorcycles = self.tracker.update(motorcycles)
            bike_tracks = [track for track in tracked_motorcycles.values() if track.get("missing_frames", 0) == 0]

            helmet_detections: list[dict] = []
            no_helmet_detections: list[dict] = []
            plate_detections: list[dict] = []
            rider_detections: list[dict] = []
            if self.helmet_model is not None:
                helmet_result = self.helmet_model(inference_frame, verbose=False, imgsz=inference_imgsz)[0]
                raw_helmet_detections = result_to_detections(helmet_result, min_confidence=self.confidence)
                raw_helmet_detections = scale_detections(raw_helmet_detections, scale)
                if self.no_helmet_class_ids:
                    no_helmet_detections = [
                        item for item in raw_helmet_detections if item["class_id"] in self.no_helmet_class_ids
                    ]
                if self.helmet_class_ids:
                    helmet_detections = [
                        item for item in raw_helmet_detections if item["class_id"] in self.helmet_class_ids
                    ]
                else:
                    helmet_detections = raw_helmet_detections
                plate_detections = [item for item in raw_helmet_detections if item["class_id"] in self.plate_class_ids]
                rider_detections = [item for item in raw_helmet_detections if item["class_id"] in self.rider_class_ids]

            rider_candidates = self._merge_rider_candidates(rider_detections, persons)

            for bike in bike_tracks:
                track_id = bike["track_id"]
                bike_bbox = bike["bbox"]
                matched_riders = self._match_riders_to_bike(rider_candidates, bike_bbox)
                explicit_no_helmet_heads = self._match_head_detections_to_bike(no_helmet_detections, bike_bbox)
                explicit_helmet_heads = self._match_head_detections_to_bike(helmet_detections, bike_bbox)
                face_heads = (
                    self._detect_face_heads(original_frame, bike_bbox)
                    if not explicit_no_helmet_heads and not matched_riders
                    else []
                )

                if not matched_riders and not explicit_no_helmet_heads and not face_heads:
                    continue

                violating_head_boxes: list[tuple[int, int, int, int]] = list(explicit_no_helmet_heads)
                for face_head in face_heads:
                    if any(
                        overlap_ratio(face_head, helmet_head) >= 0.35 or overlap_ratio(helmet_head, face_head) >= 0.35
                        for helmet_head in explicit_helmet_heads
                    ):
                        continue
                    violating_head_boxes.append(face_head)

                for rider in matched_riders:
                    rider_bbox = rider["bbox"]
                    head_bbox = self._head_region(rider_bbox)
                    no_helmet_head_bbox = self._find_detection_in_region(head_bbox, no_helmet_detections)
                    helmet_head_bbox = self._find_detection_in_region(head_bbox, helmet_detections)
                    explicit_no_helmet = no_helmet_head_bbox is not None
                    has_helmet = self._rider_has_helmet(frame, rider_bbox, helmet_detections)
                    head_display_bbox = no_helmet_head_bbox or helmet_head_bbox or head_bbox

                    if any(
                        overlap_ratio(head_display_bbox, existing) >= 0.45 or overlap_ratio(existing, head_display_bbox) >= 0.45
                        for existing in explicit_helmet_heads
                    ):
                        continue

                    if explicit_no_helmet or not has_helmet:
                        violating_head_boxes.append(head_display_bbox)

                violating_head_boxes = self._dedupe_bboxes(violating_head_boxes)
                for rider_index, head_bbox in enumerate(violating_head_boxes, start=1):
                    cv2.rectangle(frame, head_bbox[:2], head_bbox[2:], (0, 0, 255), 2)
                    draw_label(frame, f"No Helmet Rider {rider_index}", (head_bbox[0], max(30, head_bbox[1])), (0, 0, 255))

                matched_plate = self._match_plate_to_bike(plate_detections, bike_bbox)
                matched_plate_bbox = matched_plate["bbox"] if matched_plate is not None else None
                group_bbox = self._combine_bboxes(
                    [bike_bbox] + [rider["bbox"] for rider in matched_riders] + violating_head_boxes,
                    frame.shape,
                    padding_x=20,
                    padding_y=18,
                ) or bike_bbox

                if not violating_head_boxes:
                    continue

                cv2.rectangle(frame, group_bbox[:2], group_bbox[2:], (0, 0, 255), 2)
                plate_display_bbox = matched_plate_bbox or estimate_plate_bbox(
                    group_bbox,
                    frame.shape,
                    width_ratio=0.34,
                    height_ratio=0.11,
                    y_anchor=0.74,
                )
                px1, py1, px2, py2 = plate_display_bbox
                cv2.rectangle(frame, (px1, py1), (px2, py2), (50, 220, 255), 2)
                draw_label(frame, "Number Plate", (px1, max(30, py1)), (50, 220, 255))

                if track_id in self.flagged_tracks:
                    continue

                if self.event_gate.should_skip(
                    frame_index=frame_index,
                    bbox=group_bbox,
                    number_plate=None,
                    track_id=track_id,
                ):
                    continue

                draw_label(
                    frame,
                    f"Bike {track_id} | No Helmet Riders: {len(violating_head_boxes)}",
                    (group_bbox[0], max(30, group_bbox[1])),
                    (0, 0, 255),
                )
                violation_frame = frame.copy()
                image_path, absolute_image_path = save_violation_image(violation_frame, "helmet")
                record_id = persist_violation("helmet", "UNKNOWN", image_path)
                self.flagged_tracks.add(track_id)
                self.event_gate.record(
                    frame_index=frame_index,
                    bbox=group_bbox,
                    number_plate=None,
                    track_id=track_id,
                )
                self.save_executor.submit(
                    self._save_violation_record,
                    record_id,
                    absolute_image_path,
                    violation_frame,
                    original_frame.copy(),
                    bike_bbox,
                    matched_plate_bbox,
                    plate_display_bbox,
                    track_id,
                )
                violations += 1
                self.logger.info("Helmet violation saved | record_id=%s | track_id=%s", record_id, track_id)

            cv2.putText(
                frame,
                f"Helmet Violations: {violations}",
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
        self.logger.info("Processing complete. Total helmet violations: %s", violations)

    def _save_violation_record(
        self,
        record_id: int,
        absolute_image_path: Path,
        frame_snapshot,
        source_frame,
        bike_bbox: tuple[int, int, int, int],
        matched_plate_bbox: Optional[tuple[int, int, int, int]],
        fallback_plate_bbox: tuple[int, int, int, int],
        track_id: int,
    ) -> None:
        plate, absolute_plate_bbox = self._resolve_plate_for_bike(source_frame, bike_bbox, matched_plate_bbox)
        final_plate_bbox = absolute_plate_bbox or fallback_plate_bbox
        if final_plate_bbox is not None:
            px1, py1, px2, py2 = final_plate_bbox
            cv2.rectangle(frame_snapshot, (px1, py1), (px2, py2), (50, 220, 255), 2)
            draw_label(frame_snapshot, f"Number Plate | {plate}", (px1, max(30, py1)), (50, 220, 255))
        cv2.imwrite(str(absolute_image_path), frame_snapshot)
        update_violation_number_plate(record_id, plate)
        self.logger.info(
            "Helmet violation enriched | record_id=%s | track_id=%s | plate=%s",
            record_id,
            track_id,
            plate,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect bike riders without helmets from a video.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--vehicle-model",
        default=str(settings.vehicle_model_path),
        help="Path to the YOLO vehicle model. Defaults to models/yolov8n.pt if present.",
    )
    parser.add_argument(
        "--helmet-model",
        default=str(settings.helmet_model_path),
        help="Optional path to a custom helmet detection model.",
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument("--show", action="store_true", help="Display the live processed video window.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the annotated output video. Defaults to outputs/helmet/.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    detector = HelmetViolationDetector(
        vehicle_model_path=arguments.vehicle_model,
        helmet_model_path=arguments.helmet_model,
        confidence=arguments.confidence,
    )
    detector.run(video_path=arguments.video, show=arguments.show, output_path=arguments.output)
