from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ReportStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    HITL_REQUIRED = "hitl_required"
    COMPLETE = "complete"
    ERROR = "error"


class GenerateReportRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=500)
    brand_kit: dict = {}


class ReportResponse(BaseModel):
    report_id: str
    status: ReportStatus
    prompt: str
    created_at: datetime
    report_urls: dict = {}
    qa_report: dict = {}
    error: str | None = None


class HITLReviewRequest(BaseModel):
    checkpoint: str = Field(
        ...,
        pattern=r"^(cp1_metadata|cp2_schema|cp3_insights|cp4_narrative)$",
    )
    action: str = Field(..., pattern=r"^(approved|corrected|rejected)$")
    corrections: dict = {}
