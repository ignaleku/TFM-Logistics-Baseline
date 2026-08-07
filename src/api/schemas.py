from __future__ import annotations
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    orders_path: str = Field(default="data/uploads/orders_uploaded.csv")
    checkpoint: str = Field(default="data/dqn_rl3_final.pt")
    cost_late_urgent: float = Field(default=20.0, ge=0)
    cost_late_normal: float = Field(default=5.0, ge=0)
    worker_cost_per_hour: float = Field(default=15.0, ge=0)
    hours_per_worker_month: float = Field(default=160.0, ge=0)
    months: Optional[Union[List[str], str]] = Field(
        default=None,
        description="Months to evaluate. List of names/numbers or comma-separated string. "
                    "Null / omitted means all months.",
    )


class RunStartedResponse(BaseModel):
    status: str
    run_id: str
    message: str


class RunResponse(BaseModel):
    run_id: str
    status: str
    elapsed_seconds: float
    output_paths: Dict[str, str]
    stdout_tail: Optional[str] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    status: str
    total_rows: int
    date_range: Optional[str] = None
    detected_months: List[str] = []
    urgent_share: float = 0.0
    message: str


class FilesStatusResponse(BaseModel):
    uploaded_orders: bool
    checkpoint: bool
    latest_capacity_results: bool
    latest_recommendations_summary: bool
    latest_full_results: bool
    paths: Dict[str, str]


# ── Future planning ──────────────────────────────────────────────────────────

class FuturePreviewRequest(BaseModel):
    planning_month: str = Field(description="Month name, abbreviation, or number (1-12)")
    expected_annual_orders: float = Field(gt=0)
    monthly_orders_override: Optional[float] = Field(default=None, gt=0)
    uncertainty_level: str = Field(default="standard", description="low | standard | high")


class FutureRunRequest(BaseModel):
    planning_month: str
    expected_annual_orders: float = Field(gt=0)
    monthly_orders_override: Optional[float] = Field(default=None, gt=0)
    uncertainty_level: str = Field(default="standard")
    checkpoint: str = Field(default="data/dqn_rl3_final.pt")
    cost_late_urgent: float = Field(default=15.0, ge=0)
    cost_late_normal: float = Field(default=10.0, ge=0)
    worker_cost_per_hour: float = Field(default=18.0, ge=0)
    hours_per_worker_month: float = Field(default=160.0, ge=0)
    current_picking_workers: Optional[int] = Field(default=None, ge=0)
    current_packing_workers: Optional[int] = Field(default=None, ge=0)
    current_dispatch_workers: Optional[int] = Field(default=None, ge=0)
    regimes: Optional[List[str]] = Field(default=None, description="Subset of base regime labels to evaluate; omit for all 16.")
