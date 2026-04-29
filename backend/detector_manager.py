from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import settings
from detection.utils import setup_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_DETECTORS = ("helmet", "red_light", "wrong_side")


@dataclass
class DetectorProcess:
    detector_type: str
    process: subprocess.Popen
    log_handle: object
    log_path: Path
    command: list[str]
    started_at: datetime


class DetectorManager:
    def __init__(self) -> None:
        self.logger = setup_logger("detector_manager")
        self._lock = threading.Lock()
        self._processes: dict[str, DetectorProcess] = {}

    def _build_command(self, detector_type: str) -> list[str]:
        if detector_type == "helmet":
            return [
                sys.executable,
                "-m",
                "detection.helmet_detection",
                "--video",
                str(settings.helmet_video_path),
                "--show",
                "--confidence",
                "0.25",
            ]

        if detector_type == "red_light":
            return [
                sys.executable,
                "-m",
                "detection.red_light_detection",
                "--video",
                str(settings.red_light_video_path),
                "--show",
                "--confidence",
                "0.22",
                "--line-y",
                str(settings.red_light_default_line_y),
                "--approach-direction",
                settings.red_light_default_approach_direction,
                "--red-duration",
                str(settings.red_light_default_red_duration),
                "--green-duration",
                str(settings.red_light_default_green_duration),
            ]

        if detector_type == "wrong_side":
            return [
                sys.executable,
                "-m",
                "detection.wrong_side_detection",
                "--video",
                str(settings.wrong_side_video_path),
                "--show",
                "--confidence",
                "0.25",
                "--allowed-direction",
                settings.wrong_side_default_direction,
                "--min-displacement",
                str(settings.wrong_side_default_min_displacement),
            ]

        raise ValueError(f"Unsupported detector: {detector_type}")

    def _video_path_for(self, detector_type: str) -> Path:
        if detector_type == "helmet":
            return settings.helmet_video_path
        if detector_type == "red_light":
            return settings.red_light_video_path
        if detector_type == "wrong_side":
            return settings.wrong_side_video_path
        raise ValueError(f"Unsupported detector: {detector_type}")

    def _cleanup_finished_locked(self) -> None:
        finished: list[str] = []
        for detector_type, record in self._processes.items():
            return_code = record.process.poll()
            if return_code is None:
                continue

            try:
                record.log_handle.close()
            except Exception:
                pass

            self.logger.info(
                "Detector %s finished with exit code %s. Log: %s",
                detector_type,
                return_code,
                record.log_path,
            )
            finished.append(detector_type)

        for detector_type in finished:
            self._processes.pop(detector_type, None)

    def _log_path_for(self, detector_type: str) -> Path:
        return settings.detector_logs_dir / f"{detector_type}_{datetime.now():%Y%m%d_%H%M%S}.log"

    def start(self, detector_type: str) -> dict[str, object]:
        if detector_type not in VALID_DETECTORS:
            raise ValueError(f"Unsupported detector: {detector_type}")

        with self._lock:
            self._cleanup_finished_locked()
            current = self._processes.get(detector_type)
            if current is not None and current.process.poll() is None:
                return self._status_from_record(current, running=True, already_running=True)

            video_path = self._video_path_for(detector_type)
            if not video_path.exists():
                raise FileNotFoundError(f"Configured video not found: {video_path}")

            command = self._build_command(detector_type)
            log_path = self._log_path_for(detector_type)
            log_handle = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            creationflags = 0
            if sys.platform.startswith("win"):
                creationflags = (
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                )

            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=creationflags,
            )
            record = DetectorProcess(
                detector_type=detector_type,
                process=process,
                log_handle=log_handle,
                log_path=log_path,
                command=command,
                started_at=datetime.utcnow(),
            )
            self._processes[detector_type] = record
            self.logger.info("Started %s detector with pid=%s", detector_type, process.pid)
            return self._status_from_record(record, running=True, already_running=False)

    def stop(self, detector_type: str) -> dict[str, object]:
        if detector_type not in VALID_DETECTORS:
            raise ValueError(f"Unsupported detector: {detector_type}")

        with self._lock:
            self._cleanup_finished_locked()
            record = self._processes.get(detector_type)
            if record is None or record.process.poll() is not None:
                if record is not None:
                    try:
                        record.log_handle.close()
                    except Exception:
                        pass
                    self._processes.pop(detector_type, None)
                return {
                    "detector_type": detector_type,
                    "running": False,
                    "pid": None,
                    "started_at": None,
                    "log_path": None,
                    "already_running": False,
                }

            process = record.process
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("Detector %s did not stop in time; killing pid=%s", detector_type, process.pid)
                process.kill()
                process.wait(timeout=5)
            finally:
                try:
                    record.log_handle.close()
                except Exception:
                    pass
                self._processes.pop(detector_type, None)

            self.logger.info("Stopped %s detector with pid=%s", detector_type, process.pid)
            return {
                "detector_type": detector_type,
                "running": False,
                "pid": None,
                "started_at": None,
                "log_path": str(record.log_path),
                "already_running": False,
            }

    def list_statuses(self) -> list[dict[str, object]]:
        with self._lock:
            self._cleanup_finished_locked()
            statuses = []
            for detector_type in VALID_DETECTORS:
                record = self._processes.get(detector_type)
                if record is None:
                    statuses.append(
                        {
                            "detector_type": detector_type,
                            "running": False,
                            "pid": None,
                            "started_at": None,
                            "log_path": None,
                            "already_running": False,
                        }
                    )
                else:
                    statuses.append(self._status_from_record(record, running=record.process.poll() is None, already_running=False))
            return statuses

    def _status_from_record(
        self,
        record: DetectorProcess,
        *,
        running: bool,
        already_running: bool,
    ) -> dict[str, object]:
        return {
            "detector_type": record.detector_type,
            "running": running,
            "pid": record.process.pid,
            "started_at": record.started_at.isoformat(timespec="seconds"),
            "log_path": str(record.log_path),
            "already_running": already_running,
        }


detector_manager = DetectorManager()
