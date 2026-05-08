from __future__ import annotations
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]

ALLOWED_MODULES = {
    "src.rl.evaluate_rl5_monthly_capacity_cost",
    "src.reporting.export_rl5_monthly_recommendations",
}


def _run(module: str, args: list[str]) -> tuple[int, str, str]:
    if module not in ALLOWED_MODULES:
        raise ValueError(f"Module '{module}' is not in the allowed list.")
    cmd = [sys.executable, "-m", module, *args]
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    return result.returncode, result.stdout, result.stderr


def run_monthly_capacity_cost(
    orders_path: str,
    checkpoint: str,
    cost_late_urgent: float,
    cost_late_normal: float,
    worker_cost_per_hour: float,
    hours_per_worker_month: float,
) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex[:8]
    out_dir = ROOT / "data" / "api_runs" / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)

    capacity_csv = out_dir / "rl5_monthly_capacity_cost_results.csv"
    summary_csv = out_dir / "rl5_monthly_recommendations_summary.csv"
    full_csv = out_dir / "rl5_monthly_capacity_cost_results_app.csv"

    t0 = time.monotonic()

    # ── Step 1: run simulation ────────────────────────────────────────────────
    rc1, out1, err1 = _run(
        "src.rl.evaluate_rl5_monthly_capacity_cost",
        [
            "--orders", orders_path,
            "--checkpoint", checkpoint,
            "--cost-late-urgent", str(cost_late_urgent),
            "--cost-late-normal", str(cost_late_normal),
            "--worker-cost-per-hour", str(worker_cost_per_hour),
            "--hours-per-worker-month", str(hours_per_worker_month),
            "--output", str(capacity_csv.relative_to(ROOT)),
        ],
    )
    if rc1 != 0:
        return {
            "run_id": run_id,
            "status": "error",
            "step": "evaluate_rl5_monthly_capacity_cost",
            "elapsed_seconds": time.monotonic() - t0,
            "output_paths": {},
            "error": (err1 or out1)[-3000:],
        }

    # ── Step 2: export recommendations ───────────────────────────────────────
    rc2, out2, err2 = _run(
        "src.reporting.export_rl5_monthly_recommendations",
        [
            "--input", str(capacity_csv.relative_to(ROOT)),
            "--output-dir", str(out_dir.relative_to(ROOT)),
        ],
    )
    if rc2 != 0:
        return {
            "run_id": run_id,
            "status": "error",
            "step": "export_rl5_monthly_recommendations",
            "elapsed_seconds": time.monotonic() - t0,
            "output_paths": {"capacity_results": str(capacity_csv)},
            "error": (err2 or out2)[-3000:],
        }

    elapsed = time.monotonic() - t0
    combined_stdout = (out1 + "\n" + out2)[-4000:]

    return {
        "run_id": run_id,
        "status": "ok",
        "elapsed_seconds": round(elapsed, 1),
        "output_paths": {
            "capacity_results": str(capacity_csv.relative_to(ROOT)),
            "recommendations_summary": str(summary_csv.relative_to(ROOT)),
            "full_results": str(full_csv.relative_to(ROOT)),
        },
        "stdout_tail": combined_stdout,
        "error": None,
    }
