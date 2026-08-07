from __future__ import annotations

import datetime
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

ALLOWED_MODULES = {
    "src.rl.evaluate_rl3_monthly_capacity_cost",
    "src.reporting.export_rl3_monthly_recommendations",
}

OUT_DIR = ROOT / "data" / "api_runs" / "latest"
STATUS_FILE = OUT_DIR / "status.json"


def _write_status(
    status: str,
    step: str,
    progress_pct: int,
    message: str,
    started_at: str | None = None,
    error: str | None = None,
    run_mode: str | None = None,
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    """`detail` carries optional fine-grained progress fields (phase, regime X/Y, policy,
    finalist X/Y, replication X/Y, candidate, iteration X/Y) — spec §9. Written no more than
    once per regime/policy/finalist boundary by callers, never per SimPy event."""
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
    if run_mode:
        payload["run_mode"] = run_mode
    if detail:
        payload["detail"] = detail
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


def _write_run_manifest(
    out_dir: Path,
    run_mode: str,
    months: List[int],
    order_summary_df: pd.DataFrame,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a small, run-scoped manifest describing what the latest run actually covered —
    used by GET /results/latest/run-scope so the frontend (Demand & Complexity) can stay
    correct after a browser refresh or 'Load Latest Results', instead of relying on ephemeral
    React state (spec §5.4)."""
    import calendar

    manifest: Dict[str, Any] = {
        "run_mode": run_mode,
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "months": months,
        "month_names": [calendar.month_name[m] for m in months],
        "order_summary": json.loads(order_summary_df.to_json(orient="records")),
    }
    if extra:
        manifest.update(extra)
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _build_bottleneck_analysis(
    capacity_csv: Path,
    orders_path: Path,
    checkpoint_path: Path,
    cost_params: Dict[str, float],
    run_mode: str,
) -> Dict[str, Any]:
    """Build one bottleneck report per month present in capacity_csv (spec §13.4). Cheap
    (bottleneck ranking + break-even) for every month; adaptive search only runs for months
    whose recommended candidate actually needs it (see bottleneck_report.py)."""
    from src.analysis.bottleneck_report import build_bottleneck_report, sanitize_for_json

    results_df = pd.read_csv(capacity_csv)
    orders_all = pd.read_csv(orders_path, parse_dates=["arrival_time"])

    reports = []
    for month_num in sorted(results_df["month"].unique()):
        report = build_bottleneck_report(
            int(month_num), orders_all, results_df, checkpoint_path, cost_params,
            root=ROOT, run_mode=run_mode,
        )
        reports.append(report)
    return sanitize_for_json({"run_mode": run_mode, "months": reports})


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
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    capacity_csv = out_dir / "rl3_monthly_capacity_cost_results.csv"
    summary_csv  = out_dir / "rl3_monthly_recommendations_summary.csv"
    full_csv     = out_dir / "rl3_monthly_capacity_cost_results_app.csv"
    bottleneck_json = out_dir / "bottleneck_analysis.json"

    t0         = time.monotonic()
    started_at = datetime.datetime.utcnow().isoformat()

    # ── Resolve months arg ────────────────────────────────────────────────────
    months_cli: list[str] = []
    if months:
        months_str = months if isinstance(months, str) else ",".join(str(m) for m in months)
        months_cli = ["--months", months_str]
        tokens = [t.strip() for t in months_str.split(",") if t.strip()]
        months_label = ", ".join(tokens)
        status_msg = f"Running optimisation for: {months_label}…"
    else:
        status_msg = "Running full-year monthly optimisation…"

    # ── Step 1: run RL-3 monthly capacity-cost simulation ────────────────────
    _write_status("running", "evaluating_base_regimes", 5, status_msg, started_at, run_mode="historical")

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
    sim_ok = rc1 == 0 or capacity_csv.exists()
    if not sim_ok:
        err_detail = (err1 or out1)[-3000:]
        _write_status("failed", "evaluating_base_regimes", 0,
                      "Simulation failed — see error field", started_at,
                      error=err_detail, run_mode="historical")
        return {
            "run_id": run_id, "status": "failed", "step": "evaluate_rl3_monthly_capacity_cost",
            "elapsed_seconds": time.monotonic() - t0, "output_paths": {}, "error": err_detail,
        }

    # ── Step 2: export recommendations ───────────────────────────────────────
    _write_status("running", "exporting_results", 70, "Exporting monthly recommendations…", started_at, run_mode="historical")

    rc2, out2, err2 = _run(
        "src.reporting.export_rl3_monthly_recommendations",
        [
            "--input",          str(capacity_csv.relative_to(ROOT)),
            "--output-summary", str(summary_csv.relative_to(ROOT)),
            "--output-full",    str(full_csv.relative_to(ROOT)),
        ],
    )
    export_ok = rc2 == 0 or (summary_csv.exists() and full_csv.exists())
    if not export_ok:
        err_detail = (err2 or out2)[-3000:]
        _write_status("failed", "exporting_results", 70,
                      "Export failed — see error field", started_at,
                      error=err_detail, run_mode="historical")
        return {
            "run_id": run_id, "status": "failed", "step": "export_rl3_monthly_recommendations",
            "elapsed_seconds": time.monotonic() - t0,
            "output_paths": {"capacity_results": str(capacity_csv.relative_to(ROOT))}, "error": err_detail,
        }

    # ── Step 2.5: run-scoped manifest (months actually evaluated + demand summary) ───
    try:
        from src.analysis.order_summary import build_order_summary
        from src.rl.evaluate_rl3_monthly_capacity_cost import parse_months

        orders_all = pd.read_csv(ROOT / orders_path, parse_dates=["arrival_time"])
        if "month" not in orders_all.columns:
            orders_all["month"] = orders_all["arrival_time"].dt.month
        available_months = sorted(int(m) for m in orders_all["month"].unique())
        if months:
            months_str = months if isinstance(months, str) else ",".join(str(m) for m in months)
            requested = set(parse_months(months_str))
            resolved_months = [m for m in available_months if m in requested]
        else:
            resolved_months = available_months
        scoped_orders = orders_all[orders_all["month"].isin(resolved_months)]
        order_summary_df = build_order_summary(scoped_orders)
        _write_run_manifest(out_dir, "historical", resolved_months, order_summary_df)
    except Exception:
        # Manifest is supplementary (Demand & Complexity scoping) — must not fail the run.
        pass

    # ── Step 3: bottleneck / adaptive-capacity analysis ──────────────────────
    _write_status("running", "analysing_bottlenecks", 85, "Analysing bottlenecks and capacity…", started_at, run_mode="historical")
    try:
        cost_params = {
            "cost_late_urgent": cost_late_urgent, "cost_late_normal": cost_late_normal,
            "worker_cost_per_hour": worker_cost_per_hour, "hours_per_worker_month": hours_per_worker_month,
        }
        analysis = _build_bottleneck_analysis(capacity_csv, ROOT / orders_path, ROOT / checkpoint, cost_params, "historical")
        bottleneck_json.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    except Exception as exc:
        # Bottleneck analysis is supplementary — a failure here must not fail the whole run,
        # since the core recommendations are already saved.
        bottleneck_json.write_text(json.dumps({"error": str(exc)[:1000]}), encoding="utf-8")

    elapsed = time.monotonic() - t0
    _write_status("completed", "done", 100,
                  f"Monthly optimisation completed in {elapsed:.1f}s", started_at, run_mode="historical")

    return {
        "run_id": run_id,
        "status": "completed",
        "elapsed_seconds": round(elapsed, 1),
        "output_paths": {
            "capacity_results":        str(capacity_csv.relative_to(ROOT)),
            "recommendations_summary": str(summary_csv.relative_to(ROOT)),
            "full_results":            str(full_csv.relative_to(ROOT)),
            "bottleneck_analysis":     str(bottleneck_json.relative_to(ROOT)),
        },
        "stdout_tail": (out1 + "\n" + out2)[-4000:],
        "error": None,
    }


def run_future_planning(
    planning_month: str,
    expected_annual_orders: float,
    monthly_orders_override: Optional[float],
    uncertainty_level: str,
    checkpoint: str,
    cost_late_urgent: float,
    cost_late_normal: float,
    worker_cost_per_hour: float,
    hours_per_worker_month: float,
    regimes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Future-planning background job (spec §7): generate the replication-#1 scenario and
    screen ALL base regimes x 3 policies on it; validate only the top `screening_finalists`
    regimes x 3 policies on the remaining replications; aggregate the finalists' 3
    replications; build the bottleneck/adaptive-capacity report (itself screen-then-validate
    for future planning, spec §8). Runs fully in-process (no subprocess)."""
    from src.analysis.bottleneck_report import build_bottleneck_report, sanitize_for_json
    from src.analysis.future_screening import select_finalists
    from src.analysis.order_summary import build_order_summary
    from src.analysis.replication_aggregation import aggregate_replications
    from src.data.future_scenario import build_preview, generate_future_scenario_orders
    from src.data.planning_profile import load_planning_profile
    from src.reporting.export_rl3_monthly_recommendations import _build_app_results, _build_summary
    from src.rl.evaluate_rl3_monthly_capacity_cost import evaluate_monthly_capacity_cost

    run_id = uuid.uuid4().hex[:8]
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    capacity_csv = out_dir / "rl3_monthly_capacity_cost_results.csv"
    summary_csv  = out_dir / "rl3_monthly_recommendations_summary.csv"
    full_csv     = out_dir / "rl3_monthly_capacity_cost_results_app.csv"
    bottleneck_json = out_dir / "bottleneck_analysis.json"
    future_summary_json = out_dir / "future_planning_summary.json"

    t0 = time.monotonic()
    started_at = datetime.datetime.utcnow().isoformat()
    cost_params = {
        "cost_late_urgent": cost_late_urgent, "cost_late_normal": cost_late_normal,
        "worker_cost_per_hour": worker_cost_per_hour, "hours_per_worker_month": hours_per_worker_month,
    }

    try:
        _write_status("running", "generating_scenarios", 5, "Generating future demand scenario…", started_at,
                      run_mode="future", detail={"phase": "Generating scenarios / preparation"})
        profile = load_planning_profile()
        preview = build_preview(planning_month, expected_annual_orders, monthly_orders_override, uncertainty_level, profile)
        month_num = preview["month"]
        n_reps = preview["replications"]
        checkpoint_path = ROOT / checkpoint
        regimes_all = regimes or list(profile["regimes"].keys())
        finalist_count = int(profile["future_planning"].get("screening_finalists", 4))

        # ── Stage A: screening — every regime x 3 policies on replication #1 only ────────
        rep0_orders = generate_future_scenario_orders(
            planning_month, expected_annual_orders, monthly_orders_override, uncertainty_level,
            replication=0, profile=profile,
        )
        screening_total = len(regimes_all) * 3

        def _screen_progress_cb(done: int, total: int) -> None:
            if done % 3 != 0 and done != total:
                return
            regime_idx = min(done // 3, len(regimes_all) - 1)
            pct = 10 + int(45 * done / max(1, total))
            _write_status(
                "running", "evaluating_base_regimes", pct,
                f"Screening workforce configurations ({done}/{total} simulations)…", started_at, run_mode="future",
                detail={
                    "phase": "Base screening", "regime": regime_idx + 1, "regime_total": len(regimes_all),
                    "completed_simulations": done, "estimated_total_simulations": total,
                },
            )

        df_screen = evaluate_monthly_capacity_cost(
            rep0_orders, checkpoint_path, cost_late_urgent, cost_late_normal,
            worker_cost_per_hour, hours_per_worker_month,
            months=[month_num], regime_names=regimes_all, root=ROOT, seed_offset=0,
            progress_cb=_screen_progress_cb,
        )

        finalists = select_finalists(df_screen, regimes_all, finalist_count) if n_reps > 1 else list(regimes_all)

        # ── Stage B: validation — finalists only, on the remaining replications ─────────
        extra_rep_orders: List[pd.DataFrame] = []
        per_rep_dfs_finalists = [df_screen[df_screen["regime"].isin(finalists)].reset_index(drop=True)]
        validation_total = len(finalists) * 3 * max(0, n_reps - 1)

        for r in range(1, n_reps):
            scenario_orders = generate_future_scenario_orders(
                planning_month, expected_annual_orders, monthly_orders_override, uncertainty_level,
                replication=r, profile=profile,
            )
            extra_rep_orders.append(scenario_orders)

            def _val_progress_cb(done: int, total: int, _r=r) -> None:
                if done % 3 != 0 and done != total:
                    return
                pct = 55 + int(20 * ((_r - 1) + done / max(1, total)) / max(1, n_reps - 1))
                _write_status(
                    "running", "validating_finalists", pct,
                    f"Validating finalists — replication {_r + 1}/{n_reps}…", started_at, run_mode="future",
                    detail={
                        "phase": "Finalist validation", "replication": _r + 1, "replication_total": n_reps,
                        "finalist": min(done // 3, len(finalists) - 1) + 1, "finalist_total": len(finalists),
                    },
                )

            df_r = evaluate_monthly_capacity_cost(
                scenario_orders, checkpoint_path, cost_late_urgent, cost_late_normal,
                worker_cost_per_hour, hours_per_worker_month,
                months=[month_num], regime_names=finalists, root=ROOT, seed_offset=0,
                progress_cb=_val_progress_cb,
            )
            per_rep_dfs_finalists.append(df_r)

        finalists_agg_df = aggregate_replications(per_rep_dfs_finalists)
        finalists_agg_df["evaluation_stage"] = "validated"

        non_finalists = [r for r in regimes_all if r not in finalists]
        if non_finalists:
            screening_only_raw = df_screen[df_screen["regime"].isin(non_finalists)].reset_index(drop=True)
            screening_only_agg_df = aggregate_replications([screening_only_raw])
            screening_only_agg_df["evaluation_stage"] = "screening"
            combined_df = pd.concat([finalists_agg_df, screening_only_agg_df], ignore_index=True, sort=False)
        else:
            combined_df = finalists_agg_df
        combined_df.to_csv(capacity_csv, index=False)

        # ── Bottleneck report + adaptive search (screen-then-validate, spec §8) ─────────
        _write_status("running", "adaptive_capacity_search", 78, "Running bottleneck-directed capacity search…",
                      started_at, run_mode="future", detail={"phase": "Bottleneck / adaptive search"})
        bottleneck_report = build_bottleneck_report(
            month_num, rep0_orders, combined_df, checkpoint_path, cost_params, root=ROOT, run_mode="future",
            extra_replication_orders=extra_rep_orders,
        )
        bottleneck_report["scenario_preview"] = preview
        bottleneck_report["replication_count"] = n_reps
        bottleneck_json.write_text(json.dumps(sanitize_for_json({"run_mode": "future", "months": [bottleneck_report]}), indent=2), encoding="utf-8")

        adaptive_sim_count = int(bottleneck_report.get("adaptive_search", {}).get("simulations_executed") or 0)
        screening_sim_count = screening_total
        validation_sim_count = validation_total
        theoretical_exhaustive = len(regimes_all) * 3 * n_reps
        base_total = screening_sim_count + validation_sim_count

        _write_status("running", "exporting_results", 93, "Exporting recommendations…", started_at,
                      run_mode="future", detail={"phase": "Aggregation / export"})
        summary_df = _build_summary(combined_df)
        summary_df.to_csv(summary_csv, index=False)
        full_df = _build_app_results(combined_df)
        full_df.to_csv(full_csv, index=False)

        # ── Run-scoped manifest (Demand & Complexity — spec §5) ─────────────────────────
        try:
            order_summary_df = build_order_summary(rep0_orders)
            _write_run_manifest(out_dir, "future", [month_num], order_summary_df, extra={
                "planning_month": month_num,
                "month_name": preview["month_name"],
                "forecast_source": preview["source"],
                "uncertainty_level": uncertainty_level,
                "expected_monthly_orders": preview["expected_monthly_orders"],
                "preview": preview,
            })
        except Exception:
            pass

        evaluation_strategy_meta = {
            "evaluation_strategy": "screening_then_validation",
            "total_replications": n_reps,
            "screening_replications": 1,
            "validation_replications": max(0, n_reps - 1),
            "screening_regime_count": len(regimes_all),
            "finalist_regime_count": len(finalists),
            "screening_simulation_count": screening_sim_count,
            "validation_simulation_count": validation_sim_count,
            "adaptive_simulation_count": adaptive_sim_count,
            "total_simulations_executed": base_total + adaptive_sim_count,
            "theoretical_exhaustive_simulations": theoretical_exhaustive,
            "simulations_saved": theoretical_exhaustive - base_total,
        }

        future_summary_json.write_text(json.dumps(sanitize_for_json({
            "run_mode": "future", "preview": preview, "replication_count": n_reps,
            "planning_month": planning_month, "expected_annual_orders": expected_annual_orders,
            "monthly_orders_override": monthly_orders_override, "uncertainty_level": uncertainty_level,
            "cost_params": cost_params,
            "finalists": finalists,
            "evaluation_strategy": evaluation_strategy_meta,
        }), indent=2), encoding="utf-8")

        elapsed = time.monotonic() - t0
        _write_status("completed", "done", 100, f"Future planning completed in {elapsed:.1f}s", started_at, run_mode="future")

        return {
            "run_id": run_id, "status": "completed", "elapsed_seconds": round(elapsed, 1),
            "output_paths": {
                "capacity_results": str(capacity_csv.relative_to(ROOT)),
                "recommendations_summary": str(summary_csv.relative_to(ROOT)),
                "full_results": str(full_csv.relative_to(ROOT)),
                "bottleneck_analysis": str(bottleneck_json.relative_to(ROOT)),
                "future_planning_summary": str(future_summary_json.relative_to(ROOT)),
            },
            "evaluation_strategy": evaluation_strategy_meta,
            "error": None,
        }
    except Exception as exc:
        err_detail = str(exc)[:3000]
        _write_status("failed", "unknown", 0, "Future planning failed — see error field", started_at, error=err_detail, run_mode="future")
        return {
            "run_id": run_id, "status": "failed", "elapsed_seconds": time.monotonic() - t0,
            "output_paths": {}, "error": err_detail,
        }
