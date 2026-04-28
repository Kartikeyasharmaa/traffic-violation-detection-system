from __future__ import annotations

import logging
import ctypes
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from backend.database import SessionLocal, create_violation, init_db, update_violation_number_plate as update_violation_number_plate_db
from config import settings


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(settings.logs_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_yolo_model(model_path: str | Path | None, logger: logging.Logger, fallback: Optional[str] = None) -> YOLO:
    if model_path and Path(model_path).exists():
        model_reference = str(model_path)
    elif fallback:
        model_reference = fallback
        logger.info("Model %s not found locally. Falling back to %s.", model_path, fallback)
    elif model_path:
        model_reference = str(model_path)
    else:
        model_reference = settings.default_vehicle_model

    logger.info("Loading YOLO model from %s", model_reference)
    return YOLO(model_reference)


def result_to_detections(result, allowed_classes: Optional[Iterable[int]] = None, min_confidence: float = 0.25) -> list[dict]:
    detections: list[dict] = []
    allowed = set(allowed_classes) if allowed_classes is not None else None

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        if confidence < min_confidence:
            continue
        if allowed is not None and class_id not in allowed:
            continue

        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "class_id": class_id,
                "confidence": confidence,
            }
        )
    return detections


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def crop_with_padding(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: int = 8,
) -> np.ndarray:
    crop, _ = crop_with_padding_and_offset(frame, bbox, padding)
    return crop


def crop_with_padding_and_offset(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: int = 8,
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return frame[y1:y2, x1:x2].copy(), (x1, y1)


def estimate_plate_bbox(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int] | tuple[int, int],
    *,
    width_ratio: float = 0.44,
    height_ratio: float = 0.14,
    y_anchor: float = 0.70,
) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

    plate_width = max(28, int(box_width * width_ratio))
    plate_height = max(14, int(box_height * height_ratio))
    center_x = int((x1 + x2) / 2)
    center_y = int(y1 + box_height * y_anchor)

    px1 = max(0, center_x - plate_width // 2)
    py1 = max(0, center_y - plate_height // 2)
    px2 = min(frame_width, px1 + plate_width)
    py2 = min(frame_height, py1 + plate_height)
    return (px1, py1, px2, py2)


def overlap_ratio(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_w * intersection_h
    if intersection <= 0:
        return 0.0

    box_a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return intersection / box_a_area


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection_area = intersection_w * intersection_h
    if intersection_area <= 0:
        return 0.0

    box_a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    box_b_area = max(1, (bx2 - bx1) * (by2 - by1))
    union_area = box_a_area + box_b_area - intersection_area
    return intersection_area / max(union_area, 1)


def point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def save_violation_image(frame: np.ndarray, violation_type: str) -> tuple[str, Path]:
    if violation_type not in settings.valid_violation_types:
        raise ValueError(f"Unsupported violation type: {violation_type}")

    filename = f"{violation_type}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    relative_path = Path(violation_type) / filename
    absolute_path = settings.image_dir / relative_path
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(absolute_path), frame)
    return relative_path.as_posix(), absolute_path


def build_output_video_path(video_path: str | Path, violation_type: str) -> Path:
    source_name = Path(video_path).stem
    filename = f"{source_name}_{violation_type}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    return settings.outputs_dir / violation_type / filename


def open_video_capture(video_path: str | int) -> cv2.VideoCapture:
    if isinstance(video_path, int):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            capture = cv2.VideoCapture(video_path, backend)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                capture.set(cv2.CAP_PROP_FPS, 24)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                try:
                    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:
                    pass
                return capture
            capture.release()
    return cv2.VideoCapture(video_path)


def prepare_frame_for_inference(
    frame: np.ndarray,
    max_width: int = 1280,
) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0

    scale = max_width / float(max(width, 1))
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_bbox(
    bbox: tuple[int, int, int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    if scale == 1.0:
        return bbox

    inverse = 1.0 / scale
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * inverse),
        int(y1 * inverse),
        int(x2 * inverse),
        int(y2 * inverse),
    )


def scale_detections(detections: list[dict], scale: float) -> list[dict]:
    if scale == 1.0:
        return detections

    scaled: list[dict] = []
    for detection in detections:
        updated = dict(detection)
        updated["bbox"] = scale_bbox(detection["bbox"], scale)
        scaled.append(updated)
    return scaled


def get_display_size() -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1366, 768


def fit_frame_for_display(frame: np.ndarray, max_size: tuple[int, int]) -> np.ndarray:
    max_width = max(640, int(max_size[0] * 0.80))
    max_height = max(480, int(max_size[1] * 0.80))
    height, width = frame.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    if scale >= 1.0:
        return frame

    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


class FrameOutputManager:
    def __init__(
        self,
        *,
        video_path: str,
        violation_type: str,
        fps: float,
        frame_size: tuple[int, int],
        show: bool,
        logger: logging.Logger,
        output_path: str | None = None,
    ) -> None:
        self.logger = logger
        self.window_name = f"{violation_type.title().replace('_', ' ')} Detection"
        self.show_enabled = show
        self.display_size = get_display_size()
        self.window_ready = False
        self.output_path = None
        self.writer = None
        save_video = output_path is not None or not show

        if save_video:
            target_path = Path(output_path) if output_path else build_output_video_path(video_path, violation_type)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            resolved_fps = fps if fps and fps > 0 else 20.0
            self.writer = cv2.VideoWriter(str(target_path), fourcc, resolved_fps, frame_size)
            self.output_path = target_path

            if self.writer.isOpened():
                self.logger.info("Annotated output video will be saved to %s", self.output_path)
            else:
                self.logger.warning("Could not open output video writer for %s", self.output_path)
                self.writer = None
        else:
            self.logger.info("Live preview mode enabled. Output video saving is disabled unless --output is provided.")

    def handle_frame(self, frame: np.ndarray) -> bool:
        if self.writer is not None:
            self.writer.write(frame)

        if not self.show_enabled:
            return False

        try:
            display_frame = fit_frame_for_display(frame, self.display_size)
            if not self.window_ready:
                try:
                    cv2.startWindowThread()
                except cv2.error:
                    pass
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                try:
                    cv2.resizeWindow(self.window_name, display_frame.shape[1], display_frame.shape[0])
                    cv2.moveWindow(self.window_name, 40, 40)
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
                except cv2.error:
                    pass
                self.window_ready = True
            cv2.imshow(self.window_name, display_frame)
            return (cv2.waitKey(1) & 0xFF) == ord("q")
        except cv2.error as exc:
            self.logger.warning(
                "OpenCV GUI display is unavailable on this system. Continuing without a live window. %s",
                exc,
            )
            self.show_enabled = False
            return False

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


class ViolationEventGate:
    def __init__(self, cooldown_frames: int = 90) -> None:
        self.cooldown_frames = cooldown_frames
        self.recorded_track_ids: set[int] = set()
        self.recent_events: deque[dict] = deque(maxlen=300)

    def _prune(self, frame_index: int) -> None:
        while self.recent_events and frame_index - self.recent_events[0]["frame_index"] > self.cooldown_frames:
            self.recent_events.popleft()

    def should_skip(
        self,
        *,
        frame_index: int,
        bbox: tuple[int, int, int, int],
        number_plate: Optional[str],
        track_id: Optional[int] = None,
        spatial_matching: bool = True,
    ) -> bool:
        if track_id is not None and track_id in self.recorded_track_ids:
            return True

        self._prune(frame_index)
        current_center = bbox_center(bbox)
        x1, y1, x2, y2 = bbox
        current_area = max(1, (x2 - x1) * (y2 - y1))
        current_diagonal = max(1.0, float(np.hypot(x2 - x1, y2 - y1)))
        normalized_plate = normalize_plate(number_plate)

        for event in self.recent_events:
            if normalized_plate and event["plate"] and normalized_plate == event["plate"]:
                return True

            if spatial_matching:
                if bbox_iou(event["bbox"], bbox) >= 0.55:
                    return True

                area_ratio = current_area / max(event["area"], 1)
                center_distance = float(np.hypot(current_center[0] - event["center"][0], current_center[1] - event["center"][1]))
                if 0.55 <= area_ratio <= 1.8 and center_distance <= min(current_diagonal, event["diagonal"]) * 0.35:
                    return True

        return False

    def record(
        self,
        *,
        frame_index: int,
        bbox: tuple[int, int, int, int],
        number_plate: Optional[str],
        track_id: Optional[int] = None,
    ) -> None:
        if track_id is not None:
            self.recorded_track_ids.add(track_id)

        x1, y1, x2, y2 = bbox
        self.recent_events.append(
            {
                "frame_index": frame_index,
                "bbox": bbox,
                "center": bbox_center(bbox),
                "area": max(1, (x2 - x1) * (y2 - y1)),
                "diagonal": max(1.0, float(np.hypot(x2 - x1, y2 - y1))),
                "plate": normalize_plate(number_plate),
            }
        )


def persist_violation(
    violation_type: str,
    number_plate: Optional[str],
    image_path: str,
) -> int:
    init_db()
    db = SessionLocal()
    try:
        record = create_violation(
            db=db,
            violation_type=violation_type,
            number_plate=number_plate,
            image_path=image_path,
        )
        return record.id
    finally:
        db.close()


def update_violation_number_plate(violation_id: int, number_plate: Optional[str]) -> None:
    init_db()
    db = SessionLocal()
    try:
        update_violation_number_plate_db(
            db=db,
            violation_id=violation_id,
            number_plate=number_plate,
        )
    finally:
        db.close()


def normalize_plate(number_plate: Optional[str]) -> Optional[str]:
    if not number_plate:
        return None

    normalized = "".join(char for char in str(number_plate).upper() if char.isalnum())
    if not normalized or normalized == "UNKNOWN":
        return None
    return normalized


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    cv2.rectangle(frame, (x, y - 24), (x + max(140, len(text) * 9), y), color, -1)
    cv2.putText(frame, text, (x + 6, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_direction_guides(frame: np.ndarray, allowed_direction: str) -> None:
    height, width = frame.shape[:2]
    guide_color = (255, 220, 0)

    if allowed_direction in {"ltr", "rtl"}:
        guide_y = int(height * 0.88)
        cv2.line(frame, (50, guide_y), (width - 50, guide_y), guide_color, 3)
        if allowed_direction == "ltr":
            cv2.arrowedLine(frame, (80, guide_y - 28), (width - 80, guide_y - 28), guide_color, 3, tipLength=0.05)
            label = "Expected flow: Left to Right"
        else:
            cv2.arrowedLine(frame, (width - 80, guide_y - 28), (80, guide_y - 28), guide_color, 3, tipLength=0.05)
            label = "Expected flow: Right to Left"
    else:
        guide_x = int(width * 0.10)
        cv2.line(frame, (guide_x, 50), (guide_x, height - 50), guide_color, 3)
        if allowed_direction == "ttb":
            cv2.arrowedLine(frame, (guide_x + 30, 80), (guide_x + 30, height - 80), guide_color, 3, tipLength=0.05)
            label = "Expected flow: Top to Bottom"
        else:
            cv2.arrowedLine(frame, (guide_x + 30, height - 80), (guide_x + 30, 80), guide_color, 3, tipLength=0.05)
            label = "Expected flow: Bottom to Top"

    draw_label(frame, label, (20, 110), guide_color)


def pair_riders_with_bikes(persons: list[dict], bikes: list[dict]) -> list[tuple[dict, dict]]:
    pairs: list[tuple[dict, dict]] = []
    used_person_indices: set[int] = set()

    for bike in bikes:
        bx1, by1, bx2, by2 = bike["bbox"]
        best_match: Optional[tuple[float, int, dict]] = None

        for index, person in enumerate(persons):
            if index in used_person_indices:
                continue

            px1, py1, px2, py2 = person["bbox"]
            overlap_w = max(0, min(bx2, px2) - max(bx1, px1))
            person_width = max(1, px2 - px1)
            if overlap_w / person_width < 0.30:
                continue

            if py2 < by1 or py1 > by2:
                continue

            bike_center_x, bike_center_y = bbox_center(bike["bbox"])
            person_center_x, person_center_y = bbox_center(person["bbox"])
            distance = abs(bike_center_x - person_center_x) + abs(bike_center_y - person_center_y)

            if best_match is None or distance < best_match[0]:
                best_match = (distance, index, person)

        if best_match is not None:
            used_person_indices.add(best_match[1])
            pairs.append((bike, best_match[2]))

    return pairs


def line_side(
    point: tuple[int, int],
    line_start: tuple[int, int],
    line_end: tuple[int, int],
) -> float:
    return (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - (line_end[1] - line_start[1]) * (
        point[0] - line_start[0]
    )


class CentroidTracker:
    def __init__(self, max_disappeared: int = 20, max_distance: int = 80, history_size: int = 30) -> None:
        self.next_object_id = 0
        self.objects: OrderedDict[int, dict] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.history_size = history_size

    def register(self, detection: dict) -> None:
        bbox = detection["bbox"]
        centroid = bbox_center(bbox)
        self.objects[self.next_object_id] = {
            "track_id": self.next_object_id,
            "bbox": bbox,
            "class_id": detection.get("class_id"),
            "confidence": detection.get("confidence"),
            "centroid": centroid,
            "history": deque([centroid], maxlen=self.history_size),
            "bbox_history": deque([bbox], maxlen=self.history_size),
            "missing_frames": 0,
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections: list[dict]) -> OrderedDict[int, dict]:
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                self.objects[object_id]["missing_frames"] = self.disappeared[object_id]
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array([bbox_center(detection["bbox"]) for detection in detections], dtype="int")

        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array([data["centroid"] for data in self.objects.values()], dtype="int")

        distances = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows: set[int] = set()
        used_cols: set[int] = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if distances[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            detection = detections[col]
            centroid = tuple(input_centroids[col])
            self.objects[object_id]["bbox"] = detection["bbox"]
            self.objects[object_id]["class_id"] = detection.get("class_id")
            self.objects[object_id]["confidence"] = detection.get("confidence")
            self.objects[object_id]["centroid"] = centroid
            self.objects[object_id]["history"].append(centroid)
            self.objects[object_id]["bbox_history"].append(detection["bbox"])
            self.objects[object_id]["missing_frames"] = 0
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distances.shape[0])) - used_rows
        unused_cols = set(range(distances.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            self.objects[object_id]["missing_frames"] = self.disappeared[object_id]
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        for col in unused_cols:
            self.register(detections[col])

        return self.objects
