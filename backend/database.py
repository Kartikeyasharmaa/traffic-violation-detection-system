from __future__ import annotations

from datetime import datetime
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.models import Base, Violation
from config import settings


engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_violation(
    db: Session,
    violation_type: str,
    number_plate: Optional[str],
    image_path: str,
    occurred_at: Optional[datetime] = None,
) -> Violation:
    violation = Violation(
        violation_type=violation_type,
        number_plate=number_plate,
        image_path=image_path,
        timestamp=occurred_at or datetime.utcnow(),
    )
    db.add(violation)
    db.commit()
    db.refresh(violation)
    return violation


def update_violation_number_plate(
    db: Session,
    violation_id: int,
    number_plate: Optional[str],
) -> Optional[Violation]:
    violation = db.get(Violation, violation_id)
    if violation is None:
        return None

    violation.number_plate = number_plate
    db.commit()
    db.refresh(violation)
    return violation
