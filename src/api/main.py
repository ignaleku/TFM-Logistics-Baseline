"""
TFM Logistics API — FastAPI backend.

Start with:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.runners import run_monthly_capacity_cost
from src.api.schemas import (
    FilesStatusResponse,
    RunRequest,
    RunResponse,
    UploadResponse,
)
from src.api.utils import validate_orders_csv

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TFM Logistics API",
    description="Simulation + RL-5 capacity planning backend",
    version="1.0.0",
)

ROOT = Path(__file__).resolve().parents[2]

_frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend_origin, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────────────────────

UPLOADS_DIR = ROOT / "data" / "uploads"
API_RUNS_DIR = ROOT / "data" / "api_runs" / "latest"

CHECKPOINT_PRIMARY = ROOT / "data" / "dqn_rl5_v2_final.pt"
CHECKPOINT_FALLBACK = ROOT / "data" / "dqn_rl5_final.pt"

UPLOADED_ORDERS = UPLOADS_DIR / "orders_uploaded.csv"
CAPACITY_CSV = API_RUNS_DIR / "rl5_monthly_capacity_cost_results.csv"
SUMMARY_CSV = API_RUNS_DIR / "rl5_monthly_recommendations_summary.csv"
FULL_CSV = API_RUNS_DIR / "rl5_monthly_capacity_cost_results_app.csv"


def _csv_to_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}. Run a simulation first.")
    df = pd.read_csv(path)
    return df.where(df.notna(), None).to_dict(orient="records")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "tfm-logistics-api"}


@app.post("/upload-orders", response_model=UploadResponse)
async def upload_orders(file: UploadFile = File(...)):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    UPLOADED_ORDERS.write_bytes(content)

    result = validate_orders_csv(UPLOADED_ORDERS)
    if not result["valid"]:
        UPLOADED_ORDERS.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=result["error"])

    return UploadResponse(
        status="ok",
        total_rows=result["total_rows"],
        date_range=result.get("date_range"),
        detected_months=result.get("detected_months", []),
        urgent_share=result.get("urgent_share", 0.0),
        message=f"Uploaded {result['total_rows']:,} orders successfully.",
    )


@app.post("/run/monthly-capacity-cost", response_model=RunResponse)
def run_monthly(req: RunRequest):
    orders_path = ROOT / req.orders_path
    if not orders_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Orders file not found: {req.orders_path}. Upload a CSV first via POST /upload-orders.",
        )

    # Resolve checkpoint (try requested path, then primary, then fallback)
    ckpt_path = ROOT / req.checkpoint
    if not ckpt_path.exists():
        if CHECKPOINT_PRIMARY.exists():
            ckpt_path = CHECKPOINT_PRIMARY
        elif CHECKPOINT_FALLBACK.exists():
            ckpt_path = CHECKPOINT_FALLBACK
        else:
            raise HTTPException(
                status_code=400,
                detail="RL-5 checkpoint not found. Expected data/dqn_rl5_v2_final.pt.",
            )

    result = run_monthly_capacity_cost(
        orders_path=req.orders_path,
        checkpoint=str(ckpt_path.relative_to(ROOT)),
        cost_late_urgent=req.cost_late_urgent,
        cost_late_normal=req.cost_late_normal,
        worker_cost_per_hour=req.worker_cost_per_hour,
        hours_per_worker_month=req.hours_per_worker_month,
    )

    if result["status"] == "error":
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Simulation failed at step: {result.get('step', 'unknown')}",
                "error": result.get("error", ""),
            },
        )

    return RunResponse(**result)


@app.get("/results/latest/recommendations")
def get_latest_recommendations():
    return _csv_to_records(SUMMARY_CSV)


@app.get("/results/latest/full")
def get_latest_full():
    return _csv_to_records(FULL_CSV)


@app.get("/recommend/month/{month_name}")
def recommend_month(month_name: str):
    if not SUMMARY_CSV.exists():
        raise HTTPException(status_code=404, detail="No results available. Run a simulation first.")

    summary_df = pd.read_csv(SUMMARY_CSV)
    row = summary_df[summary_df["month_name"].str.lower() == month_name.lower()]
    if row.empty:
        available = summary_df["month_name"].tolist()
        raise HTTPException(
            status_code=404,
            detail=f"Month '{month_name}' not found. Available: {available}",
        )

    rec: Dict[str, Any] = row.iloc[0].where(row.iloc[0].notna(), None).to_dict()

    # Augment with min-SLA threshold options from full results
    if FULL_CSV.exists():
        full_df = pd.read_csv(FULL_CSV)
        mdf = full_df[full_df["month_name"].str.lower() == month_name.lower()]

        if "urgent_sla" in mdf.columns and not mdf.empty:
            urgent_ok = mdf[mdf["urgent_sla"] >= 0.95]
            if not urgent_ok.empty:
                best = urgent_ok.loc[urgent_ok["total_workers"].idxmin()]
                rec["min_urgent_sla_option"] = best.where(best.notna(), None).to_dict()
            else:
                rec["min_urgent_sla_option"] = None

        if "total_sla" in mdf.columns and not mdf.empty:
            total_ok = mdf[mdf["total_sla"] >= 0.80]
            if not total_ok.empty:
                best = total_ok.loc[total_ok["total_workers"].idxmin()]
                rec["min_total_sla_option"] = best.where(best.notna(), None).to_dict()
            else:
                rec["min_total_sla_option"] = None

    return rec


@app.get("/files/status", response_model=FilesStatusResponse)
def files_status():
    checkpoint_ok = CHECKPOINT_PRIMARY.exists() or CHECKPOINT_FALLBACK.exists()
    checkpoint_path = (
        str(CHECKPOINT_PRIMARY.relative_to(ROOT))
        if CHECKPOINT_PRIMARY.exists()
        else str(CHECKPOINT_FALLBACK.relative_to(ROOT))
        if CHECKPOINT_FALLBACK.exists()
        else "not found"
    )
    return FilesStatusResponse(
        uploaded_orders=UPLOADED_ORDERS.exists(),
        checkpoint=checkpoint_ok,
        latest_capacity_results=CAPACITY_CSV.exists(),
        latest_recommendations_summary=SUMMARY_CSV.exists(),
        latest_full_results=FULL_CSV.exists(),
        paths={
            "uploaded_orders": str(UPLOADED_ORDERS.relative_to(ROOT)) if UPLOADED_ORDERS.exists() else "not found",
            "checkpoint": checkpoint_path,
            "capacity_results": str(CAPACITY_CSV.relative_to(ROOT)) if CAPACITY_CSV.exists() else "not found",
            "recommendations_summary": str(SUMMARY_CSV.relative_to(ROOT)) if SUMMARY_CSV.exists() else "not found",
            "full_results": str(FULL_CSV.relative_to(ROOT)) if FULL_CSV.exists() else "not found",
        },
    )
