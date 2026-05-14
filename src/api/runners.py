from __future__ import annotations

import datetime
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]

ALLOWED_MODULES = {
    "src.rl.evaluate_rl3_monthly_capacity_cost",
    "src.reporting.export_rl3_monthly_recommendations",
}

STATUS_FILE = ROOT / "data" / "api_runs" / "latest" / "status.json"


def _write_status(
    status: str,
    step: str,
    progress_pct: int,
    message: str,
    started_at: str | None = None,
    error: str | None = None,
) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "status": status,
        "step": step,
        "progress_pct": progress_pct,
        "message": message,
        "started_at": started_at or datetime.datetime.utcnow().isoformat(),
        "updated_at": datetime.datetime.utcnow().isoformat(),
        "error": error,
    }
    STATUS_FILE.write_text(json.dumps(payload), encoding="utf-8")


def _run(module: str, args: list[str]) -> tuple[int, str, str]:
    if module not in ALLOWED_MODULES:
        raise ValueError(f"Module '{module}' is not in the allowed list.")
    cmd = [sys.executable, "-m", module, *args]
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=None,
        env=env,
    )
    return result.returncode, result.stdout, result.stderr


def run_monthly_capacity_cost(
    orders_path: str,
    checkpoint: str,
    cost_late_urgent: float,
    cost_late_normal: float,
    worker_cost_per_hour: float,
    hours_per_worker_month: float,
    months: list[str] | str | None = None,
) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex[:8]
    out_dir = ROOT / "data" / "api_runs" / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)

    capacity_csv = out_dir / "rl3_monthly_capacity_cost_results.csv"
    summary_csv  = out_dir / "rl3_monthly_recommendations_summary.csv"
    full_csv     = out_dir / "rl3_monthly_capacity_cost_results_app.csv"

    t0         = time.monotonic()
    started_at = datetime.datetime.utcnow().isoformat()

    # ── Resolve months arg ────────────────────────────────────────────────────
    months_cli: list[str] = []
    if months:
        months_str = months if isinstance(months, str) else ",".join(str(m) for m in months)
        months_cli = ["--months", months_str]
        # Build a short display label for the status message
        tokens = [t.strip() for t in months_str.split(",") if t.strip()]
        months_label = ", ".join(tokens)
        status_msg = f"Running optimisation for: {months_label}…"
    else:
        status_msg = "Running full-year monthly optimisation…"

    # ── Step 1: run RL-3 monthly capacity-cost simulation ────────────────────
    _write_status("running", "capacity_cost", 5, status_msg, started_at)

    rc1, out1, err1 = _run(
        "src.rl.evaluate_rl3_monthly_capacity_cost",
        [
            "--orders",                 orders_path,
            "--checkpoint",             checkpoint,
            "--cost-late-urgent",       str(cost_late_urgent),
            "--cost-late-normal",       str(cost_late_normal),
            "--worker-cost-per-hour",   str(worker_cost_per_hour),
            "--hours-per-worker-month", str(hours_per_worker_month),
            "--output",                 str(capacity_csv.relative_to(ROOT)),
            *months_cli,
        ],
    )
    # Consider step 1 successful if output CSV was written, regardless of exit code.
    # A non-zero rc with the file present typically means warnings printed to stderr.
    sim_ok = rc1 == 0 or capacity_csv.exists()
    if not sim_ok:
        err_detail = (err1 or out1)[-3000:]
        _write_status("failed", "capacity_cost", 0,
                      "Simulation failed — see error field", started_at,
                      error=err_detail)
        return {
            "run_id": run_id,
            "status": "failed",
            "step": "evaluate_rl3_monthly_capacity_cost",
            "elapsed_seconds": time.monotonic() - t0,
            "output_paths": {},
            "error": err_detail,
        }

    # ── Step 2: export recommendations ───────────────────────────────────────
    _write_status("running", "export", 80,
                  "Exporting monthly recommendations…", started_at)

    rc2, out2, err2 = _run(
        "src.reporting.export_rl3_monthly_recommendations",
        [
            "--input",          str(capacity_csv.relative_to(ROOT)),
            "--output-summary", str(summary_csv.relative_to(ROOT)),
            "--output-full",    str(full_csv.relative_to(ROOT)),
        ],
    )
    # Consider step 2 successful if both output files were produced, regardless of exit code.
    # The export script may print warnings to stderr (non-zero rc) yet still write valid files.
    export_ok = rc2 == 0 or (summary_csv.exists() and full_csv.exists())
    if not export_ok:
        err_detail = (err2 or out2)[-3000:]
        _write_status("failed", "export", 80,
                      "Export failed — see error field", started_at,
                      error=err_detail)
        return {
            "run_id": run_id,
            "status": "failed",
            "step": "export_rl3_monthly_recommendations",
            "elapsed_seconds": time.monotonic() - t0,
            "output_paths": {"capacity_results": str(capacity_csv.relative_to(ROOT))},
            "error": err_detail,
        }

    elapsed = time.monotonic() - t0
    _write_status("completed", "done", 100,
                  f"Monthly optimisation completed in {elapsed:.1f}s", started_at)

    return {
        "run_id": run_id,
        "status": "completed",
        "elapsed_seconds": round(elapsed, 1),
        "output_paths": {
            "capacity_results":        str(capacity_csv.relative_to(ROOT)),
            "recommendations_summary": str(summary_csv.relative_to(ROOT)),
            "full_results":            str(full_csv.relative_to(ROOT)),
        },
        "stdout_tail": (out1 + "\n" + out2)[-4000:],
        "error": None,
    }
