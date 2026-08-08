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
from typing import Any, Dict, List, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.runners import (
    FUTURE_DIR,
    HISTORICAL_DIR,
    run_future_planning,
    run_monthly_capacity_cost,
)
from src.api.schemas import (
    FilesStatusResponse,
    FuturePreviewRequest,
    FutureRunRequest,
    RunRequest,
    RunStartedResponse,
    UploadResponse,
)
from src.api.utils import enrich_orders_df, validate_orders_csv
from src.data.future_scenario import build_preview
from src.data.planning_profile import load_planning_profile

Mode = Literal["future", "historical"]

# ── App setup ─────────────────────────────────────────────────────────────────

_bg_lock = threading.Lock()
_bg_running = False

app = FastAPI(
    title="TFM Logistics API",
    description="Simulation + RL-3 capacity planning backend",
    version="4.0.0",
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
STATUS_JSON = API_RUNS_DIR / "status.json"

ORDER_SUMMARY_CSV = ROOT / "data" / "orders_base_seasonal_summary.csv"

_MODE_DIRS: Dict[str, Path] = {"future": FUTURE_DIR, "historical": HISTORICAL_DIR}


def _mode_paths(mode: str) -> Dict[str, Path]:
    if mode not in _MODE_DIRS:
        raise HTTPException(status_code=422, detail="mode must be 'future' or 'historical'")
    d = _MODE_DIRS[mode]
    return {
        "summary": d / "rl3_monthly_recommendations_summary.csv",
        "full": d / "rl3_monthly_capacity_cost_results_app.csv",
        "capacity": d / "rl3_monthly_capacity_cost_results.csv",
        "bottleneck": d / "bottleneck_analysis.json",
        "manifest": d / "run_manifest.json",
    }


def _csv_to_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}. Run a simulation first.")
    df = pd.read_csv(path)
    return json.loads(df.to_json(orient="records"))


def _any_mode_outputs_exist() -> bool:
    return any(_mode_paths(m)["summary"].exists() and _mode_paths(m)["full"].exists() for m in _MODE_DIRS)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "tfm-logistics-api"}


@app.get("/data/order-summary")
def get_order_summary(mode: Mode | None = Query(default=None, description="future | historical — omit for the static annual client-profile baseline")):
    """Demand/complexity summary. With `mode`, returns the CURRENT run's scoped summary
    (embedded in that mode's run_manifest.json — spec §5); without it, the static full-year
    annual client-profile baseline (used by the collapsed 'Annual Client Profile' section)."""
    if mode is not None:
        paths = _mode_paths(mode)
        if not paths["manifest"].exists():
            raise HTTPException(status_code=404, detail=f"No {mode} run available yet.")
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        return manifest.get("order_summary", [])

    if not ORDER_SUMMARY_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail="Order summary not found. Run: python -m src.data.generate_orders_seasonal",
        )
    return _csv_to_records(ORDER_SUMMARY_CSV)


@app.get("/planning/profile")
def get_planning_profile():
    """Read-only client-profile assumptions the Future Planning UI needs: available months,
    SLA targets, uncertainty labels, cost defaults, replications. Never exposes RL reward
    coefficients or other internal training config."""
    profile = load_planning_profile()
    return {
        "version": profile["meta"]["version"],
        "months": [
            {"number": num, "name": mp["name"]}
            for num, mp in sorted(profile["months"].items())
        ],
        "sla_targets": {
            "urgent_target": profile["sla"]["urgent_target"],
            "normal_target": profile["sla"]["normal_target"],
        },
        "uncertainty_levels": [
            {
                "level": level, "demand_cv": v["demand_cv"], "arrival_cv": v["arrival_cv"],
                "description": f"Monthly demand CV {v['demand_cv']:.0%}, arrival-pattern CV {v['arrival_cv']:.0%}",
            }
            for level, v in profile["uncertainty_levels"].items()
        ],
        "cost_defaults": profile["cost_defaults"],
        "default_replications": profile["future_planning"]["default_replications"],
        "regimes": list(profile["regimes"].keys()),
        "hours_per_operating_day": profile["calendar_profile"]["hours_per_operating_day"],
    }


@app.post("/planning/preview")
def post_planning_preview(req: FuturePreviewRequest):
    """Derived planning assumptions for the given inputs — does NOT start a simulation."""
    try:
        return build_preview(
            req.planning_month, req.expected_annual_orders, req.monthly_orders_override, req.uncertainty_level,
            hours_per_worker_month=req.hours_per_worker_month,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


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
            "message": "Historical analysis started…",
            "run_mode": "historical",
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
                current_picking_workers=req.current_picking_workers,
                current_packing_workers=req.current_packing_workers,
                current_dispatch_workers=req.current_dispatch_workers,
            )
        except Exception as exc:
            try:
                STATUS_JSON.write_text(
                    json.dumps({
                        "status": "failed",
                        "step": "unknown",
                        "progress_pct": 0,
                        "run_mode": "historical",
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
        message="Historical analysis started in background. Poll GET /run/status for progress.",
    )


@app.post("/run/future-planning", response_model=RunStartedResponse)
def run_future(req: FutureRunRequest):
    global _bg_running

    ckpt_path = ROOT / req.checkpoint
    if not ckpt_path.exists():
        if CHECKPOINT_PRIMARY.exists():
            ckpt_path = CHECKPOINT_PRIMARY
        else:
            raise HTTPException(status_code=400, detail="RL-3 checkpoint not found. Expected data/dqn_rl3_final.pt.")

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
            "status": "running", "step": "generating_scenarios", "progress_pct": 5,
            "message": "Future planning started…", "run_mode": "future",
            "started_at": datetime.datetime.utcnow().isoformat(),
            "updated_at": datetime.datetime.utcnow().isoformat(),
        }),
        encoding="utf-8",
    )

    ckpt_rel = str(ckpt_path.relative_to(ROOT))

    def _background():
        global _bg_running
        try:
            run_future_planning(
                planning_month=req.planning_month,
                expected_annual_orders=req.expected_annual_orders,
                monthly_orders_override=req.monthly_orders_override,
                uncertainty_level=req.uncertainty_level,
                checkpoint=ckpt_rel,
                cost_late_urgent=req.cost_late_urgent,
                cost_late_normal=req.cost_late_normal,
                worker_cost_per_hour=req.worker_cost_per_hour,
                hours_per_worker_month=req.hours_per_worker_month,
                regimes=req.regimes,
                current_picking_workers=req.current_picking_workers,
                current_packing_workers=req.current_packing_workers,
                current_dispatch_workers=req.current_dispatch_workers,
            )
        except Exception as exc:
            try:
                STATUS_JSON.write_text(
                    json.dumps({
                        "status": "failed", "step": "unknown", "progress_pct": 0, "run_mode": "future",
                        "message": "Unexpected internal error", "error": str(exc)[:500],
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
        status="started", run_id="latest",
        message="Future planning started in background. Poll GET /run/status for progress.",
    )


@app.get("/run/status")
def run_status():
    outputs_exist = _any_mode_outputs_exist()

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
        data["message"] = "Run completed (auto-detected from output files)."
        data.pop("error", None)

    return data


@app.post("/run/sync-status")
def sync_status():
    """Repair status.json based on which output files currently exist (either mode)."""
    outputs_exist = _any_mode_outputs_exist()
    capacity_exists = any(_mode_paths(m)["capacity"].exists() for m in _MODE_DIRS)

    if outputs_exist:
        payload = {
            "status": "completed",
            "step": "done",
            "progress_pct": 100,
            "message": "Run completed (status synced from output files).",
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
            "message": "Capacity results exist but recommendations were not exported — re-run.",
        }

    return {"synced": False, "status": "idle", "outputs_found": False}


@app.get("/results/latest/recommendations")
def get_latest_recommendations(mode: Mode = Query(..., description="future | historical")):
    return _csv_to_records(_mode_paths(mode)["summary"])


@app.get("/results/latest/full")
def get_latest_full(mode: Mode = Query(..., description="future | historical")):
    return _csv_to_records(_mode_paths(mode)["full"])


@app.get("/results/latest/bottlenecks")
def get_latest_bottlenecks(mode: Mode = Query(..., description="future | historical")):
    """Bottleneck ranking, break-even economics, and adaptive-capacity search trail per month
    from the latest run of this mode — see src/analysis/bottleneck_report.py."""
    path = _mode_paths(mode)["bottleneck"]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No {mode} bottleneck analysis available. Run a simulation first.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read bottleneck analysis: {exc}")


@app.get("/results/latest/context")
def get_latest_context(mode: Mode = Query(..., description="future | historical")):
    """Run-scoped context for the latest run of this mode — which months/planning-month it
    actually covered, plus a demand/complexity summary computed from the SAME orders that were
    simulated (never the static annual baseline), current-workforce input if given, and (future
    only) the scenario preview. Used by the persistent context banner (spec §35) and the Demand
    & Complexity tab so both stay correct after a refresh or after running the other mode."""
    path = _mode_paths(mode)["manifest"]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No {mode} run available yet.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read run manifest: {exc}")


@app.get("/files/status", response_model=FilesStatusResponse)
def files_status():
    checkpoint_ok = CHECKPOINT_PRIMARY.exists()
    checkpoint_path = (
        str(CHECKPOINT_PRIMARY.relative_to(ROOT))
        if CHECKPOINT_PRIMARY.exists()
        else "not found"
    )
    any_summary = any(_mode_paths(m)["summary"].exists() for m in _MODE_DIRS)
    any_capacity = any(_mode_paths(m)["capacity"].exists() for m in _MODE_DIRS)
    any_full = any(_mode_paths(m)["full"].exists() for m in _MODE_DIRS)
    return FilesStatusResponse(
        uploaded_orders=UPLOADED_ORDERS.exists(),
        checkpoint=checkpoint_ok,
        latest_capacity_results=any_capacity,
        latest_recommendations_summary=any_summary,
        latest_full_results=any_full,
        paths={
            "uploaded_orders": str(UPLOADED_ORDERS.relative_to(ROOT)) if UPLOADED_ORDERS.exists() else "not found",
            "checkpoint": checkpoint_path,
        },
    )
