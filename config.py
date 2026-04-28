from __future__ import annotations

from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.project_name = "Traffic Violation Detection System"
        self.base_dir = Path(__file__).resolve().parent
        self.backend_dir = self.base_dir / "backend"
        self.frontend_dir = self.base_dir / "frontend"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.detector_logs_dir = self.logs_dir / "detectors"
        self.outputs_dir = self.base_dir / "outputs"
        self.videos_dir = self.base_dir / "videos"
        self.static_dir = self.backend_dir / "static"
        self.image_dir = self.static_dir / "images"
        self.database_path = self.backend_dir / "traffic_violations.db"
        self.database_url = f"sqlite:///{self.database_path.as_posix()}"

        self.default_vehicle_model = "yolov8n.pt"
        self.vehicle_model_path = self.models_dir / "yolov8n.pt"
        self.helmet_model_path = self.models_dir / "helmet.pt"
        self.ocr_languages = ["en"]

        self.vehicle_classes = [1, 2, 3, 5, 7]
        self.motorcycle_class = 3
        self.person_class = 0
        self.valid_violation_types = {"helmet", "red_light", "wrong_side"}

        self.red_light_default_line_y = 0.84
        self.red_light_default_approach_direction = "top_to_bottom"
        self.red_light_default_red_duration = 8
        self.red_light_default_green_duration = 10

        self.wrong_side_default_direction = "ltr"
        self.wrong_side_default_min_displacement = 90

        self.helmet_video_path = self.videos_dir / "helmet_video.mp4"
        self.red_light_video_path = self.videos_dir / "red_light_video.mp4"
        self.wrong_side_video_path = self.videos_dir / "wrong_side_video.mp4"

        self.ensure_directories()

    def ensure_directories(self) -> None:
        for path in (
            self.backend_dir,
            self.frontend_dir,
            self.models_dir,
            self.logs_dir,
            self.detector_logs_dir,
            self.outputs_dir,
            self.videos_dir,
            self.outputs_dir / "helmet",
            self.outputs_dir / "red_light",
            self.outputs_dir / "wrong_side",
            self.static_dir,
            self.image_dir,
            self.image_dir / "helmet",
            self.image_dir / "red_light",
            self.image_dir / "wrong_side",
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
