from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import StatsRead, Violation, ViolationRead
from config import settings


router = APIRouter(tags=["violations"])


def _to_violation_read(violation: Violation) -> ViolationRead:
    image_url = f"/images/{Path(violation.image_path).as_posix()}" if violation.image_path else None
    return ViolationRead(
        id=violation.id,
        violation_type=violation.violation_type,
        number_plate=violation.number_plate,
        image_path=violation.image_path,
        image_url=image_url,
        timestamp=violation.timestamp,
    )


@router.get("/violations", response_model=list[ViolationRead])
def list_violations(
    violation_type: Optional[Literal["helmet", "red_light", "wrong_side"]] = Query(default=None, alias="type"),
    sort: Literal["asc", "desc"] = Query(default="desc"),
    db: Session = Depends(get_db),
) -> list[ViolationRead]:
    statement = select(Violation)
    if violation_type:
        statement = statement.where(Violation.violation_type == violation_type)

    order_column = Violation.timestamp.asc() if sort == "asc" else Violation.timestamp.desc()
    rows = db.execute(statement.order_by(order_column)).scalars().all()
    return [_to_violation_read(row) for row in rows]


@router.delete("/violations/{violation_id}")
def delete_violation(violation_id: int, db: Session = Depends(get_db)) -> dict[str, str | int]:
    violation = db.get(Violation, violation_id)
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")

    image_file = settings.image_dir / violation.image_path
    if image_file.exists():
        image_file.unlink()

    db.delete(violation)
    db.commit()
    return {"message": "Violation deleted successfully", "id": violation_id}


@router.get("/stats", response_model=StatsRead)
def violation_stats(db: Session = Depends(get_db)) -> StatsRead:
    grouped = db.execute(
        select(Violation.violation_type, func.count(Violation.id)).group_by(Violation.violation_type)
    ).all()
    counts = {violation_type: count for violation_type, count in grouped}
    total = sum(counts.values())

    return StatsRead(
        total_violations=total,
        helmet_violations=counts.get("helmet", 0),
        red_light_violations=counts.get("red_light", 0),
        wrong_side_violations=counts.get("wrong_side", 0),
    )
