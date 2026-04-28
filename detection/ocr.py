from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config import settings


PLATE_PATTERN = re.compile(r"[A-Z0-9]{6,12}")


class OCRProcessor:
    def __init__(
        self,
        prefer: str = "easyocr",
        plate_detector: YOLO | None = None,
        plate_detector_path: str | Path | None = None,
    ) -> None:
        self.logger = logging.getLogger("ocr")
        self.prefer = prefer
        self.engine = None
        self.engine_name: Optional[str] = None
        self.plate_detector = plate_detector
        self.plate_detector_path = Path(plate_detector_path) if plate_detector_path else None
        self.plate_class_ids: set[int] = set()
        self._initialise_engine()
        self._initialise_plate_detector()

    def _initialise_engine(self) -> None:
        if self.prefer == "easyocr":
            try:
                import easyocr

                self.engine = easyocr.Reader(settings.ocr_languages, gpu=False)
                self.engine_name = "easyocr"
                return
            except Exception as exc:  # pragma: no cover - depends on local install/runtime
                self.logger.warning("EasyOCR unavailable, falling back to pytesseract: %s", exc)

        try:
            import pytesseract

            self.engine = pytesseract
            self.engine_name = "pytesseract"
        except Exception as exc:  # pragma: no cover - depends on local install/runtime
            self.logger.warning("OCR engine unavailable: %s", exc)
            self.engine = None
            self.engine_name = None

    def _initialise_plate_detector(self) -> None:
        if self.plate_detector is None and self.plate_detector_path and self.plate_detector_path.exists():
            try:
                self.plate_detector = YOLO(str(self.plate_detector_path))
            except Exception as exc:  # pragma: no cover - depends on local runtime
                self.logger.warning("Plate detector model could not be loaded: %s", exc)
                self.plate_detector = None

        if self.plate_detector is None:
            return

        names = self.plate_detector.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        for class_id, label in items:
            normalized = str(label).lower().replace("-", " ").replace("_", " ")
            if "plate" in normalized:
                self.plate_class_ids.add(int(class_id))

    def extract_number_plate(self, vehicle_crop: np.ndarray) -> str:
        plate, _ = self.extract_number_plate_details(vehicle_crop)
        return plate

    def extract_number_plate_details(
        self,
        vehicle_crop: np.ndarray,
    ) -> tuple[str, Optional[tuple[int, int, int, int]]]:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "UNKNOWN", None

        best_bbox: Optional[tuple[int, int, int, int]] = None
        best_plate = ""
        best_plate_score = -1.0
        best_soft_text = ""
        best_soft_score = -1.0

        for candidate in self._plate_candidates(vehicle_crop):
            if best_bbox is None and candidate.get("bbox") is not None:
                best_bbox = candidate.get("bbox")
            for prepared in self._prepare_images(candidate["image"]):
                for raw_text in self._run_ocr(prepared):
                    plate = self._clean_plate_text(raw_text)
                    if plate:
                        plate_score = self._plate_text_score(plate)
                        if plate_score > best_plate_score:
                            best_plate_score = plate_score
                            best_plate = plate
                            best_bbox = candidate.get("bbox")
                    soft_plate = self._soft_plate_text(raw_text)
                    soft_score = self._plate_text_score(soft_plate)
                    if soft_score > best_soft_score:
                        best_soft_score = soft_score
                        best_soft_text = soft_plate
                        if candidate.get("bbox") is not None:
                            best_bbox = candidate.get("bbox")

        if best_plate:
            return best_plate, best_bbox

        if 6 <= len(best_soft_text) <= 12:
            return best_soft_text, best_bbox

        return "UNKNOWN", best_bbox

    def _run_ocr(self, image: np.ndarray) -> list[str]:
        if self.engine_name == "easyocr":
            try:
                return self.engine.readtext(image, detail=0, paragraph=False, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            except Exception as exc:  # pragma: no cover - depends on local runtime
                self.logger.warning("EasyOCR failed during plate extraction: %s", exc)
                return []

        if self.engine_name == "pytesseract":
            try:
                configs = [
                    "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                ]
                texts = []
                for config in configs:
                    texts.append(self.engine.image_to_string(image, config=config))
                return texts
            except Exception as exc:  # pragma: no cover - depends on local runtime
                self.logger.warning("pytesseract failed during plate extraction: %s", exc)
                return []

        return []

    def _plate_candidates(self, image: np.ndarray) -> list[dict]:
        candidates: list[dict] = []
        candidates.extend(self._model_plate_candidates(image))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(filtered, 80, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        image_area = image.shape[0] * image.shape[1]

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
            x, y, w, h = cv2.boundingRect(contour)
            contour_area = w * h
            aspect_ratio = w / float(max(h, 1))
            area_ratio = contour_area / float(max(image_area, 1))

            bbox = (x, y, x + w, y + h)
            if self._is_reasonable_plate_bbox(image.shape, bbox, area_ratio=area_ratio, aspect_ratio=aspect_ratio):
                candidates.append({"image": image[y : y + h, x : x + w], "bbox": bbox})

        bottom_half = image[image.shape[0] // 2 :, :]
        candidates.append(
            {
                "image": bottom_half,
                "bbox": None,
            }
        )
        candidates.append({"image": image, "bbox": None})
        return candidates

    def _model_plate_candidates(self, image: np.ndarray) -> list[dict]:
        if self.plate_detector is None or not self.plate_class_ids:
            return []

        try:
            result = self.plate_detector(image, verbose=False)[0]
        except Exception as exc:  # pragma: no cover - depends on local runtime
            self.logger.warning("Plate detector inference failed: %s", exc)
            return []

        candidates: list[dict] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return candidates

        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            if class_id not in self.plate_class_ids or confidence < 0.20:
                continue
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            x1 = max(0, x1 - 4)
            y1 = max(0, y1 - 4)
            x2 = min(image.shape[1], x2 + 4)
            y2 = min(image.shape[0], y2 + 4)
            bbox = (x1, y1, x2, y2)
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            aspect_ratio = width / float(height)
            area_ratio = (width * height) / float(max(image.shape[0] * image.shape[1], 1))
            if not self._is_reasonable_plate_bbox(image.shape, bbox, area_ratio=area_ratio, aspect_ratio=aspect_ratio):
                continue
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                candidates.append({"image": crop, "bbox": bbox, "confidence": confidence})

        return candidates

    def _prepare_images(self, image: np.ndarray) -> list[np.ndarray]:
        if image.ndim == 3:
            color_enlarged = cv2.resize(image, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            color_enlarged = cv2.cvtColor(
                cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_GRAY2BGR,
            )

        enlarged = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(enlarged, (3, 3), 0)
        sharpened = cv2.addWeighted(enlarged, 1.5, blurred, -0.5, 0)
        equalized = cv2.equalizeHist(sharpened)
        _, threshold = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
        inverted = cv2.bitwise_not(threshold)
        return [color_enlarged, enlarged, sharpened, equalized, threshold, adaptive, inverted]

    def _clean_plate_text(self, text: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        if not normalized:
            return ""

        match = PLATE_PATTERN.search(normalized)
        if match:
            return match.group(0)

        if 5 <= len(normalized) <= 12:
            return normalized
        return ""

    def _soft_plate_text(self, text: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        if 5 <= len(normalized) <= 12:
            return normalized
        return ""

    def _plate_text_score(self, text: str) -> float:
        if not text:
            return -1.0

        normalized = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        if not normalized:
            return -1.0

        letters = sum(char.isalpha() for char in normalized)
        digits = sum(char.isdigit() for char in normalized)
        score = float(len(normalized))
        if 7 <= len(normalized) <= 10:
            score += 2.0
        if letters >= 2:
            score += 1.5
        if digits >= 2:
            score += 1.5
        if normalized[:2].isalpha():
            score += 0.8
        if normalized[-2:].isdigit():
            score += 0.8
        return score

    def _is_reasonable_plate_bbox(
        self,
        image_shape: tuple[int, ...],
        bbox: tuple[int, int, int, int],
        *,
        area_ratio: float | None = None,
        aspect_ratio: float | None = None,
    ) -> bool:
        height, width = image_shape[:2]
        x1, y1, x2, y2 = bbox
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        aspect = aspect_ratio if aspect_ratio is not None else box_width / float(box_height)
        area = area_ratio if area_ratio is not None else (box_width * box_height) / float(max(height * width, 1))
        center_y = (y1 + y2) / 2.0

        if not (1.8 <= aspect <= 7.5):
            return False
        if not (0.004 <= area <= 0.18):
            return False
        if center_y < height * 0.35:
            return False
        if box_height > height * 0.40 or box_width > width * 0.92:
            return False
        return True
