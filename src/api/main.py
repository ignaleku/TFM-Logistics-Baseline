"""
TFM Logistics API — FastAPI backend.

Start with:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import datetime
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.runners import run_monthly_capacity_cost
from src.api.schemas import (
    FilesStatusResponse,
    RunRequest,
    RunStartedResponse,
    UploadResponse,
)
from src.api.utils import enrich_orders_df, validate_orders_csv

# ── App setup ─────────────────────────────────────────────────────────────────

_bg_lock = threading.Lock()
_bg_running = False

app = FastAPI(
    title="TFM Logistics API",
    description="Simulation + RL-3 capacity planning backend",
    version="3.0.0",
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

CHECKPOINT_PRIMARY = ROOT / "data" / "dqn_rl3_final.pt"

UPLOADED_ORDERS = UPLOADS_DIR / "orders_uploaded.csv"
CAPACITY_CSV = API_RUNS_DIR / "rl3_monthly_capacity_cost_results.csv"
SUMMARY_CSV = API_RUNS_DIR / "rl3_monthly_recommendations_summary.csv"
FULL_CSV = API_RUNS_DIR / "rl3_monthly_capacity_cost_results_app.csv"
STATUS_JSON = API_RUNS_DIR / "status.json"

ORDER_SUMMARY_CSV = ROOT / "data" / "orders_base_seasonal_summary.csv"


def _csv_to_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}. Run a simulation first.")
    df = pd.read_csv(path)
    return json.loads(df.to_json(orient="records"))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "tfm-logistics-api"}


@app.get("/data/order-summary")
def get_order_summary():
    """Return the monthly order distribution summary (demand + complexity)."""
    if not ORDER_SUMMARY_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail="Order summary not found. Run: python -m src.data.generate_orders_seasonal",
        )
    return _csv_to_records(ORDER_SUMMARY_CSV)


@app.post("/upload-orders", response_model=UploadResponse)
async def upload_orders(file: UploadFile = File(...)):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    UPLOADED_ORDERS.write_bytes(content)

    result = validate_orders_csv(UPLOADED_ORDERS)
    if not result["valid"]:
        UPLOADED_ORDERS.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=result["error"])

    # Enrich: add product_family, complexity_level, workload units if missing
    df = pd.read_csv(UPLOADED_ORDERS, parse_dates=["arrival_time"])
    if not result.get("had_workload_columns", False):
        df = enrich_orders_df(df)

    # Ensure month column is present (required by simulation)
    if "month" not in df.columns:
        df["month"] = df["arrival_time"].dt.month
    if "scenario" not in df.columns:
        df["scenario"] = "uploaded"
    if "sla_minutes" not in df.columns:
        df["sla_minutes"] = df["order_type"].map({"urgent": 240, "normal": 1440}).fillna(1440).astype(int)

    df.to_csv(UPLOADED_ORDERS, index=False)

    return UploadResponse(
        status="ok",
        total_rows=result["total_rows"],
        date_range=result.get("date_range"),
        detected_months=result.get("detected_months", []),
        urgent_share=result.get("urgent_share", 0.0),
        message=(
            f"Uploaded {result['total_rows']:,} orders successfully"
            + (" (workload columns derived automatically)." if not result.get("had_workload_columns") else ".")
        ),
    )


@app.post("/run/monthly-capacity-cost", response_model=RunStartedResponse)
def run_monthly(req: RunRequest):
    global _bg_running

    orders_path = ROOT / req.orders_path
    if not orders_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Orders file not found: {req.orders_path}. Upload a CSV first via POST /upload-orders.",
        )

    ckpt_path = ROOT / req.checkpoint
    if not ckpt_path.exists():
        if CHECKPOINT_PRIMARY.exists():
            ckpt_path = CHECKPOINT_PRIMARY
        else:
            raise HTTPException(
                status_code=400,
                detail="RL-3 checkpoint not found. Expected data/dqn_rl3_final.pt.",
            )

    with _bg_lock:
        if _bg_running:
            raise HTTPException(
                status_code=409,
                detail="A simulation is already running. Wait for it to complete before starting a new one.",
            )
        _bg_running = True

    STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(
        json.dumps({
            "status": "running",
            "step": "capacity_cost",
            "progress_pct": 5,
            "message": "Monthly optimisation started…",
            "started_at": datetime.datetime.utcnow().isoformat(),
            "updated_at": datetime.datetime.utcnow().isoformat(),
        }),
        encoding="utf-8",
    )

    orders_rel = str(orders_path.relative_to(ROOT))
    ckpt_rel = str(ckpt_path.relative_to(ROOT))

    def _background():
        global _bg_running
        try:
            run_monthly_capacity_cost(
                orders_path=orders_rel,
                checkpoint=ckpt_rel,
                cost_late_urgent=req.cost_late_urgent,
                cost_late_normal=req.cost_late_normal,
                worker_cost_per_hour=req.worker_cost_per_hour,
                hours_per_worker_month=req.hours_per_worker_month,
                months=req.months,
            )
        except Exception as exc:
            try:
                STATUS_JSON.write_text(
                    json.dumps({
                        "status": "failed",
                        "step": "unknown",
                        "progress_pct": 0,
                        "message": "Unexpected internal error",
                        "error": str(exc)[:500],
                        "updated_at": datetime.datetime.utcnow().isoformat(),
                    }),
                    encoding="utf-8",
                )
            except Exception:
                pass
        finally:
            with _bg_lock:
                _bg_running = False

    threading.Thread(target=_background, daemon=True).start()

    return RunStartedResponse(
        status="started",
        run_id="latest",
        message="Monthly optimisation started in background. Poll GET /run/status for progress.",
    )


@app.get("/run/status")
def run_status():
    outputs_exist = SUMMARY_CSV.exists() and FULL_CSV.exists()

    if not STATUS_JSON.exists():
        if outputs_exist:
            return {
                "status": "completed",
                "step": "done",
                "progress_pct": 100,
                "message": "Results available from previous run.",
                "error": None,
            }
        return {"status": "idle"}

    try:
        data = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "idle"}

    if data.get("status") in ("failed", "error") and outputs_exist:
        data["status"] = "completed"
        data["step"] = "done"
        data["progress_pct"] = 100
        data["message"] = "Monthly optimisation completed (auto-detected from output files)."
        data.pop("error", None)

    return data


@app.post("/run/sync-status")
def sync_status():
    """Repair status.json based on which output files currently exist."""
    outputs_exist = SUMMARY_CSV.exists() and FULL_CSV.exists()
    capacity_exists = CAPACITY_CSV.exists()

    if outputs_exist:
        payload = {
            "status": "completed",
            "step": "done",
            "progress_pct": 100,
            "message": "Monthly optimisation completed (status synced from output files).",
            "updated_at": datetime.datetime.utcnow().isoformat(),
            "error": None,
        }
        STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
        STATUS_JSON.write_text(json.dumps(payload), encoding="utf-8")
        return {"synced": True, "status": "completed", "outputs_found": True}

    if capacity_exists:
        return {
            "synced": False,
            "status": "partial",
            "message": (
                "Capacity CSV exists but recommendations not exported yet. "
                "Run: python -m src.reporting.export_rl3_monthly_recommendations "
                "--input data/api_runs/latest/rl3_monthly_capacity_cost_results.csv "
                "--output-summary data/api_runs/latest/rl3_monthly_recommendations_summary.csv "
                "--output-full data/api_runs/latest/rl3_monthly_capacity_cost_results_app.csv"
            ),
        }

    return {"synced": False, "status": "idle", "outputs_found": False}


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
    checkpoint_ok = CHECKPOINT_PRIMARY.exists()
    checkpoint_path = (
        str(CHECKPOINT_PRIMARY.relative_to(ROOT))
        if CHECKPOINT_PRIMARY.exists()
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
