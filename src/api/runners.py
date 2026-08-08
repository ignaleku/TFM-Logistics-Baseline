from __future__ import annotations

import calendar
import datetime
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = ROOT / "data" / "api_runs" / "latest"
STATUS_FILE = OUT_DIR / "status.json"

# Mode-separated result persistence (spec §33): the latest Future Planning result and the
# latest Historical Analysis result are kept independently so running one mode never overwrites
# the other's results. The background-job status file stays global/shared (§33 "can remain
# global if simpler") since only one run executes at a time (see src/api/main.py::_bg_lock).
FUTURE_DIR = OUT_DIR / "future"
HISTORICAL_DIR = OUT_DIR / "historical"


def mode_dir(run_mode: str) -> Path:
    if run_mode not in ("future", "historical"):
        raise ValueError(f"run_mode must be 'future' or 'historical', got {run_mode!r}")
    d = FUTURE_DIR if run_mode == "future" else HISTORICAL_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


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


def _write_run_manifest(
    out_dir: Path,
    run_mode: str,
    months: List[int],
    order_summary_df: pd.DataFrame,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a small, run-scoped manifest describing what the latest run of this mode
    actually covered — used by GET /results/latest/context?mode=... so the frontend can stay
    correct after a browser refresh or switching between Future Planning / Historical Analysis,
    instead of relying on ephemeral React state (spec §5/§35)."""
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


def _analytical_candidates_for_month(
    month_orders: pd.DataFrame,
    service_cfg: Dict,
    hours_per_worker_month: float,
    profile: Dict,
    current_workforce: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Any]:
    """Analytical capacity estimate + dynamic candidate set for one month's orders (spec
    §13/§14). Returns the raw estimate (for manifest/audit) alongside the regime->workers dict
    ready to hand to evaluate_monthly_capacity_cost's regimes_by_month."""
    from src.analysis.candidate_generation import generate_regime_candidates
    from src.analysis.capacity_estimate import estimate_workers

    target_util = float(profile["capacity_planning"]["target_utilisation"])
    candidate_count = int(profile["capacity_planning"]["candidate_count"])

    estimate = estimate_workers(month_orders, service_cfg, hours_per_worker_month, target_util)
    centre = (estimate["workers"]["picking"], estimate["workers"]["packing"], estimate["workers"]["dispatch"])
    candidates = generate_regime_candidates(centre, candidate_count, current_workforce)

    return {"estimate": estimate, "centre": centre, "candidates": candidates}


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
    current_picking_workers: Optional[int] = None,
    current_packing_workers: Optional[int] = None,
    current_dispatch_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Historical Analysis job (spec §17): real order history, replayed against each month's
    own operating-time horizon. For each month independently: compute an analytical capacity
    estimate from that month's actual workload, generate a dynamic candidate set around it, and
    evaluate every candidate x 3 policies. Runs fully in-process (mirrors run_future_planning)."""
    from src.analysis.bottleneck_report import sanitize_for_json
    from src.analysis.order_summary import build_order_summary
    from src.data.planning_profile import load_planning_profile
    from src.reporting.export_rl3_monthly_recommendations import _build_app_results, _build_summary
    from src.rl.evaluate_rl3_monthly_capacity_cost import evaluate_monthly_capacity_cost, parse_months

    run_id = uuid.uuid4().hex[:8]
    out_dir = mode_dir("historical")

    capacity_csv = out_dir / "rl3_monthly_capacity_cost_results.csv"
    summary_csv  = out_dir / "rl3_monthly_recommendations_summary.csv"
    full_csv     = out_dir / "rl3_monthly_capacity_cost_results_app.csv"
    bottleneck_json = out_dir / "bottleneck_analysis.json"
    historical_summary_json = out_dir / "historical_analysis_summary.json"

    t0 = time.monotonic()
    started_at = datetime.datetime.utcnow().isoformat()

    current_workforce = None
    if current_picking_workers is not None or current_packing_workers is not None or current_dispatch_workers is not None:
        current_workforce = (
            int(current_picking_workers or 1), int(current_packing_workers or 1), int(current_dispatch_workers or 1),
        )

    try:
        months_label = "full year" if not months else (months if isinstance(months, str) else ",".join(str(m) for m in months))
        _write_status("running", "evaluating_candidates", 5, f"Analysing historical demand ({months_label})…",
                      started_at, run_mode="historical")

        profile = load_planning_profile()
        checkpoint_path = ROOT / checkpoint
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
        if not resolved_months:
            raise ValueError("No matching months found in the orders data.")

        with open(ROOT / "configs" / "sim_multistage.yaml", encoding="utf-8") as f:
            service_cfg = yaml.safe_load(f)["service_time"]

        # Step 1: per-month analytical estimate + dynamic candidate set (spec §13/§14/§17).
        from src.simulation.multistage.operating_time import slice_month_operating_time, operating_horizon_minutes
        horizon_minutes = operating_horizon_minutes(hours_per_worker_month)

        regimes_by_month: Dict[int, Dict[str, Tuple[int, int, int]]] = {}
        analytical_by_month: Dict[int, Any] = {}
        for m in resolved_months:
            month_orders = slice_month_operating_time(orders_all, m, horizon_minutes)
            info = _analytical_candidates_for_month(month_orders, service_cfg, hours_per_worker_month, profile, current_workforce)
            regimes_by_month[m] = info["candidates"]
            analytical_by_month[m] = info

        total_expected = sum(len(v) for v in regimes_by_month.values()) * 3

        def _progress(done: int, total: int) -> None:
            pct = 10 + int(70 * done / max(1, total))
            _write_status("running", "evaluating_candidates", pct,
                          f"Evaluating workforce candidates ({done}/{total} simulations)…", started_at,
                          run_mode="historical", detail={"phase": "Dynamic candidate evaluation"})

        results_df = evaluate_monthly_capacity_cost(
            orders_all, checkpoint_path, cost_late_urgent, cost_late_normal,
            worker_cost_per_hour, hours_per_worker_month,
            months=resolved_months, regimes_by_month=regimes_by_month, root=ROOT,
            progress_cb=_progress,
        )
        results_df.to_csv(capacity_csv, index=False)

        _write_status("running", "exporting_results", 85, "Exporting monthly recommendations…", started_at, run_mode="historical")
        summary_df = _build_summary(results_df)
        summary_df.to_csv(summary_csv, index=False)
        full_df = _build_app_results(results_df)
        full_df.to_csv(full_csv, index=False)

        scoped_orders = orders_all[orders_all["month"].isin(resolved_months)]
        order_summary_df = build_order_summary(scoped_orders)
        _write_run_manifest(out_dir, "historical", resolved_months, order_summary_df, extra={
            "current_workforce": (
                {"picking": current_workforce[0], "packing": current_workforce[1], "dispatch": current_workforce[2]}
                if current_workforce else None
            ),
            "hours_per_worker_month": hours_per_worker_month,
            "cost_params": {
                "cost_late_urgent": cost_late_urgent, "cost_late_normal": cost_late_normal,
                "worker_cost_per_hour": worker_cost_per_hour, "hours_per_worker_month": hours_per_worker_month,
            },
        })

        _write_status("running", "analysing_bottlenecks", 92, "Analysing bottlenecks and capacity…", started_at, run_mode="historical")
        cost_params = {
            "cost_late_urgent": cost_late_urgent, "cost_late_normal": cost_late_normal,
            "worker_cost_per_hour": worker_cost_per_hour, "hours_per_worker_month": hours_per_worker_month,
        }
        try:
            analysis = _build_bottleneck_analysis(capacity_csv, ROOT / orders_path, checkpoint_path, cost_params, "historical")
            bottleneck_json.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        except Exception as exc:
            bottleneck_json.write_text(json.dumps({"error": str(exc)[:1000]}), encoding="utf-8")

        historical_summary_json.write_text(json.dumps(sanitize_for_json({
            "run_mode": "historical", "months": resolved_months,
            "month_names": [calendar.month_name[m] for m in resolved_months],
            "cost_params": cost_params,
            "analytical_estimate_by_month": {
                calendar.month_name[m]: {
                    "workers": analytical_by_month[m]["estimate"]["workers"],
                    "candidate_count": len(regimes_by_month[m]),
                    "candidates": {label: list(w) for label, w in regimes_by_month[m].items()},
                }
                for m in resolved_months
            },
        }), indent=2), encoding="utf-8")

        elapsed = time.monotonic() - t0
        _write_status("completed", "done", 100, f"Historical analysis completed in {elapsed:.1f}s", started_at, run_mode="historical")

        return {
            "run_id": run_id, "status": "completed", "elapsed_seconds": round(elapsed, 1),
            "output_paths": {
                "capacity_results":        str(capacity_csv.relative_to(ROOT)),
                "recommendations_summary": str(summary_csv.relative_to(ROOT)),
                "full_results":            str(full_csv.relative_to(ROOT)),
                "bottleneck_analysis":     str(bottleneck_json.relative_to(ROOT)),
                "historical_analysis_summary": str(historical_summary_json.relative_to(ROOT)),
            },
            "error": None,
        }
    except Exception as exc:
        err_detail = str(exc)[:3000]
        _write_status("failed", "unknown", 0, "Historical analysis failed — see error field", started_at,
                      error=err_detail, run_mode="historical")
        return {
            "run_id": run_id, "status": "failed", "elapsed_seconds": time.monotonic() - t0,
            "output_paths": {}, "error": err_detail,
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
    current_picking_workers: Optional[int] = None,
    current_packing_workers: Optional[int] = None,
    current_dispatch_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Future-planning background job (spec §7/§13/§14): generate the replication-#1 scenario,
    compute an analytical capacity estimate from it, generate ~16 dynamic workforce candidates
    around that estimate (or use the explicit `regimes` override, kept for research use), and
    screen all of them x 3 policies on replication #1; validate only the top
    `screening_finalists` regimes x 3 policies on the remaining replications; aggregate the
    finalists' 3 replications; build the bottleneck/adaptive-capacity report (itself
    screen-then-validate for future planning, spec §8). Runs fully in-process (no subprocess)."""
    from src.analysis.bottleneck_report import build_bottleneck_report, sanitize_for_json
    from src.analysis.future_screening import select_finalists
    from src.analysis.order_summary import build_order_summary
    from src.analysis.replication_aggregation import aggregate_replications
    from src.data.future_scenario import build_preview, generate_future_scenario_orders
    from src.data.planning_profile import load_planning_profile
    from src.reporting.export_rl3_monthly_recommendations import _build_app_results, _build_summary
    from src.rl.evaluate_rl3_monthly_capacity_cost import evaluate_monthly_capacity_cost

    run_id = uuid.uuid4().hex[:8]
    out_dir = mode_dir("future")

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
    current_workforce = None
    if current_picking_workers is not None or current_packing_workers is not None or current_dispatch_workers is not None:
        current_workforce = (
            int(current_picking_workers or 1), int(current_packing_workers or 1), int(current_dispatch_workers or 1),
        )

    try:
        _write_status("running", "generating_scenarios", 5, "Generating future demand scenario…", started_at,
                      run_mode="future", detail={"phase": "Generating scenarios / preparation"})
        profile = load_planning_profile()
        preview = build_preview(
            planning_month, expected_annual_orders, monthly_orders_override, uncertainty_level, profile,
            hours_per_worker_month=hours_per_worker_month,
        )
        month_num = preview["month"]
        n_reps = preview["replications"]
        checkpoint_path = ROOT / checkpoint
        finalist_count = int(profile["future_planning"].get("screening_finalists", 4))

        # ── Stage A: screening — every candidate x 3 policies on replication #1 only ────────
        rep0_orders = generate_future_scenario_orders(
            planning_month, expected_annual_orders, monthly_orders_override, uncertainty_level,
            replication=0, profile=profile,
        )

        # Analytical capacity estimate + dynamic candidate set (spec §13/§14), computed on the
        # replication-#1 scenario. `regimes` (explicit override) is kept for research/manual use
        # — when given, it bypasses dynamic generation entirely.
        with open(ROOT / "configs" / "sim_multistage.yaml", encoding="utf-8") as f:
            service_cfg = yaml.safe_load(f)["service_time"]
        from src.simulation.multistage.operating_time import slice_month_operating_time, operating_horizon_minutes
        horizon_minutes = operating_horizon_minutes(hours_per_worker_month)
        month_orders_for_estimate = slice_month_operating_time(rep0_orders, month_num, horizon_minutes)
        analytical = _analytical_candidates_for_month(
            month_orders_for_estimate, service_cfg, hours_per_worker_month, profile, current_workforce,
        )

        if regimes:
            regimes_all = regimes
            regimes_by_month = None
        else:
            regimes_all = list(analytical["candidates"].keys())
            regimes_by_month = {month_num: analytical["candidates"]}

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
            months=[month_num], regime_names=(regimes_all if regimes_by_month is None else None),
            regimes_by_month=regimes_by_month, root=ROOT, seed_offset=0,
            progress_cb=_screen_progress_cb,
        )

        finalists = select_finalists(df_screen, regimes_all, finalist_count) if n_reps > 1 else list(regimes_all)

        # ── Stage B: validation — finalists only, on the remaining replications ─────────
        extra_rep_orders: List[pd.DataFrame] = []
        per_rep_dfs_finalists = [df_screen[df_screen["regime"].isin(finalists)].reset_index(drop=True)]
        validation_total = len(finalists) * 3 * max(0, n_reps - 1)
        finalist_regimes_by_month = (
            {month_num: {label: w for label, w in analytical["candidates"].items() if label in finalists}}
            if regimes_by_month is not None else None
        )

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
                months=[month_num], regime_names=(finalists if finalist_regimes_by_month is None else None),
                regimes_by_month=finalist_regimes_by_month, root=ROOT, seed_offset=0,
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
                "current_workforce": (
                    {"picking": current_workforce[0], "packing": current_workforce[1], "dispatch": current_workforce[2]}
                    if current_workforce else None
                ),
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
            "analytical_estimate": {
                "workers": analytical["estimate"]["workers"],
                "expected_workload_minutes": analytical["estimate"]["expected_workload_minutes"],
                "target_utilisation": analytical["estimate"]["target_utilisation"],
                "capacity_per_worker_minutes": analytical["estimate"]["capacity_per_worker_minutes"],
            },
            "candidate_regimes": {label: list(w) for label, w in analytical["candidates"].items()},
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
