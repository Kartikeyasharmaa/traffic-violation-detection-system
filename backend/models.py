from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    violation_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    number_plate: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    image_path: Mapped[str] = mapped_column(String(255), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, index=True)


class ViolationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    violation_type: str
    number_plate: Optional[str]
    image_path: str
    image_url: Optional[str]
    timestamp: datetime


class StatsRead(BaseModel):
    total_violations: int
    helmet_violations: int
    red_light_violations: int
    wrong_side_violations: int
