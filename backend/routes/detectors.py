from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.detector_manager import VALID_DETECTORS, detector_manager


router = APIRouter(tags=["detectors"])


@router.get("/detectors")
def list_detectors() -> dict[str, list[dict[str, object]]]:
    return {"detectors": detector_manager.list_statuses()}


@router.post("/detectors/{detector_type}/start")
def start_detector(detector_type: str) -> dict[str, object]:
    if detector_type not in VALID_DETECTORS:
        raise HTTPException(status_code=404, detail="Detector not found")

    try:
        return detector_manager.start(detector_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not start {detector_type} detector: {exc}") from exc
