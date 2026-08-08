"""
Builds the Capacity & Bottlenecks report (spec §13.4) and the Policy Comparison
recommended-workforce comparison (spec §3).

Given the results grid already produced by
src/rl/evaluate_rl3_monthly_capacity_cost.py::evaluate_monthly_capacity_cost (base regimes ×
3 policies, with bottleneck metrics already flattened per row), this module:

  1. Selects the recommended (regime, policy) for a month using SLA feasibility rules (§10):
     cheapest feasible candidate, or — if none is feasible — the candidate with the smallest
     SLA violation, explicitly labelled "best available". For future-planning results carrying
     an `evaluation_stage` column (screening vs validated — see
     src/analysis/future_screening.py), only 3-replication *validated* rows are eligible, so
     the recommendation is never chosen from a single-replication screening artifact.
  2. Attaches the bottleneck ranking + primary-bottleneck explanation for that candidate.
  3. Computes extra-worker break-even economics (§8).
  4. Triggers a bottleneck-directed adaptive capacity search (§9) only when the recommended
     candidate is infeasible, near max base capacity on its bottleneck stage, or otherwise
     meets the documented trigger conditions — never unconditionally, to keep this bounded.
     For future planning, the search screens each candidate on one replication before
     validating it on the rest (spec §8).
  5. Builds `policy_comparison`: FIFO / Urgent-First / RL-3 evaluated at the SAME final
     recommended workforce, under identical scenario seeds — the primary Policy Comparison
     view (spec §3), instead of an aggregate across every tested regime.

Used by GET /results/latest/bottlenecks (src/api/main.py) for both historical and
future-planning runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.analysis.bottleneck import STAGES, score_bottlenecks
from src.analysis.capacity_estimate import estimate_workers
from src.analysis.capacity_search import (
    break_even_metrics,
    evaluate_regime_all_policies,
    evaluate_regime_all_policies_multiseed,
    run_adaptive_capacity_search,
    run_adaptive_capacity_search_validated,
)
from src.data.planning_profile import load_planning_profile
from src.rl.evaluate_rl3_monthly_capacity_cost import load_rl3_agent
from src.simulation.multistage.operating_time import slice_month_operating_time, with_operating_horizon
from src.simulation.multistage.service_time_map import build_service_time_map

_POLICY_ORDER = ("fifo", "urgent_first", "rl3_dqn")


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars and non-finite floats to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _stage_metrics_from_row(row: pd.Series) -> Dict[str, Dict[str, Any]]:
    return {
        stage: {
            "utilisation": float(row[f"{stage}_utilisation"]),
            "avg_wait_min": float(row[f"{stage}_avg_wait_min"]),
            "p95_wait_min": float(row[f"{stage}_p95_wait_min"]),
            "avg_queue_len": float(row[f"{stage}_avg_queue_len"]),
            "max_queue_len": int(row[f"{stage}_max_queue_len"]),
            "late_wait_share": float(row[f"{stage}_late_wait_share"]),
        }
        for stage in STAGES
    }


def _select_recommendation(month_df: pd.DataFrame) -> pd.Series:
    """Cheapest feasible candidate, or (if none feasible) lowest SLA violation then cost
    (spec §10). When `evaluation_stage` is present (future planning), only 3-replication
    *validated* rows are eligible — screening-only single-replication rows never win the
    final recommendation (spec §7.4)."""
    pool = month_df
    if "evaluation_stage" in month_df.columns:
        validated = month_df[month_df["evaluation_stage"] == "validated"]
        if not validated.empty:
            pool = validated

    feasible_df = pool[pool["feasible"]]
    if not feasible_df.empty:
        return feasible_df.loc[feasible_df["estimated_total_cost"].idxmin()]
    ranked = pool.sort_values(["sla_violation", "estimated_total_cost"])
    return ranked.iloc[0]


def _should_trigger_adaptive(row: pd.Series, top_bottleneck: Dict[str, Any], profile: Dict) -> bool:
    """Trigger the adaptive search when the recommended candidate is infeasible, or when its
    bottleneck stage is under high pressure with late orders already occurring — regardless of
    where the candidate sits in the tested range. Under dynamic candidate generation (§14) the
    tested range is already bracketed around the analytical estimate, so "near the edge of the
    tested grid" (the old global max_workers_per_stage check) is no longer a meaningful signal
    on its own; capacity_search.py's own max_extra_workers_per_stage bound (relative to the
    analytical estimate, §18) keeps the adaptive search itself from running away."""
    if not bool(row["feasible"]):
        return True
    high_pressure = float(top_bottleneck["utilisation"]) >= 0.85
    late_orders_exist = (float(row["urgent_late_orders"]) + float(row["normal_late_orders"])) > 0
    return bool(high_pressure and late_orders_exist)


def _recommendation_dict(row: pd.Series, regime_source: str) -> Dict[str, Any]:
    return {
        "regime": row["regime"], "policy": row["policy"],
        "picking_workers": int(row["picking_workers"]),
        "packing_workers": int(row["packing_workers"]),
        "dispatch_workers": int(row["dispatch_workers"]),
        "total_sla": float(row["total_sla"]), "urgent_sla": float(row["urgent_sla"]), "normal_sla": float(row["normal_sla"]),
        "estimated_total_cost": float(row["estimated_total_cost"]),
        "feasible": bool(row["feasible"]), "sla_violation": float(row["sla_violation"]),
        "regime_source": regime_source,
    }


def _starvation_pattern(urgent_sla: float, normal_sla: float, sla_targets: Dict[str, float]) -> bool:
    """A policy that meets the urgent floor by starving normal orders far below its own floor
    — the pathological pattern the spec calls out for RL-3 (§3.2/§12), but computed generically
    so any policy exhibiting it is flagged the same way."""
    return bool(urgent_sla >= sla_targets["urgent_target"] and normal_sla <= 0.10)


def _policy_comparison_row(
    policy: str, total_cost: float, total_sla: float, urgent_sla: float, normal_sla: float,
    urgent_late_orders: float, normal_late_orders: float, feasible: bool, sla_violation: float,
    workers: tuple, sla_targets: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "policy": policy,
        "total_cost": float(total_cost),
        "total_sla": float(total_sla), "urgent_sla": float(urgent_sla), "normal_sla": float(normal_sla),
        "urgent_late_orders": float(urgent_late_orders), "normal_late_orders": float(normal_late_orders),
        "late_orders": float(urgent_late_orders) + float(normal_late_orders),
        "feasible": bool(feasible), "sla_violation": float(sla_violation),
        "picking_workers": int(workers[0]), "packing_workers": int(workers[1]), "dispatch_workers": int(workers[2]),
        "starvation_pattern": _starvation_pattern(urgent_sla, normal_sla, sla_targets),
    }


def _policy_comparison_from_regime(month_df: pd.DataFrame, regime: str, sla_targets: Dict[str, float]) -> List[Dict[str, Any]]:
    rows = month_df[month_df["regime"] == regime]
    out = []
    for policy in _POLICY_ORDER:
        prow = rows[rows["policy"] == policy]
        if prow.empty:
            continue
        r = prow.iloc[0]
        out.append(_policy_comparison_row(
            policy, r["estimated_total_cost"], r["total_sla"], r["urgent_sla"], r["normal_sla"],
            r["urgent_late_orders"], r["normal_late_orders"], r["feasible"], r["sla_violation"],
            (r["picking_workers"], r["packing_workers"], r["dispatch_workers"]), sla_targets,
        ))
    return out


def _policy_comparison_from_results(
    results_by_policy: Dict[str, Dict[str, Any]], workers: tuple, sla_targets: Dict[str, float],
) -> List[Dict[str, Any]]:
    out = []
    for policy in _POLICY_ORDER:
        r = results_by_policy[policy]
        out.append(_policy_comparison_row(
            policy, r["total_cost"], r["metrics"]["sla_rate"], r["metrics"]["sla_urgent"], r["metrics"]["sla_normal"],
            r["urgent_late_orders"], r["normal_late_orders"], r["feasible"], r["sla_violation"], workers, sla_targets,
        ))
    return out


def _capacity_level_diagnostics(month_df: pd.DataFrame, adaptive_trail: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Diagnostic-only counts of how many *tested* workforce configurations each policy would
    be feasible at (spec §3.5) — never used to pick the recommendation. Always reports
    numerator/denominator together, and separates base-grid regimes from adaptive candidates."""
    base_regimes = sorted(month_df["regime"].unique())
    per_policy: Dict[str, Any] = {}
    for policy in _POLICY_ORDER:
        prows = month_df[month_df["policy"] == policy]
        per_policy[policy] = {
            "feasible_count": int(prows["feasible"].sum()),
            "tested_count": int(len(prows)),
        }
    return {
        "base_regimes_tested": len(base_regimes),
        "feasible_by_policy": per_policy,
        "adaptive_candidates_tested": len(adaptive_trail),
        "adaptive_candidates_accepted": sum(1 for t in adaptive_trail if t.get("accepted")),
    }


def build_bottleneck_report(
    month_num: int,
    orders_all: pd.DataFrame,
    results_df: pd.DataFrame,
    checkpoint_path: Path,
    cost_params: Dict[str, float],
    root: Optional[Path] = None,
    run_mode: str = "historical",
    seed_offset: int = 123,
    extra_replication_orders: Optional[List[pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """`extra_replication_orders` (future planning only): order DataFrames for scenario
    replications #2, #3 (same month) — when given, the adaptive search screens each candidate
    on replication #1 and validates promising ones on the rest (spec §8) instead of the
    single-seed historical search."""
    root = root or Path(__file__).resolve().parents[2]
    profile = load_planning_profile()
    sla_targets = profile["sla"]

    month_df = results_df[results_df["month"] == month_num]
    if month_df.empty:
        raise ValueError(f"No results for month {month_num} in results_df.")

    rec_row = _select_recommendation(month_df)
    bottleneck_rows = score_bottlenecks(_stage_metrics_from_row(rec_row))
    top = bottleneck_rows[0]

    be = break_even_metrics(
        cost_params["worker_cost_per_hour"], cost_params["hours_per_worker_month"],
        cost_params["cost_late_urgent"], cost_params["cost_late_normal"],
        float(rec_row["urgent_late_orders"]), float(rec_row["normal_late_orders"]),
    )

    policy_comparison = _policy_comparison_from_regime(month_df, rec_row["regime"], sla_targets)

    report: Dict[str, Any] = {
        "run_mode": run_mode,
        "month": int(month_num),
        "month_name": rec_row["month_name"],
        "selected_recommendation": _recommendation_dict(rec_row, "base"),
        "sla_targets": sla_targets,
        "bottleneck_ranking": bottleneck_rows,
        "primary_bottleneck": top["stage"],
        "break_even": be,
        "adaptive_search": {"triggered": False},
        "policy_comparison": policy_comparison,
        "recommended_policy": rec_row["policy"],
        "capacity_level_diagnostics": _capacity_level_diagnostics(month_df, []),
    }

    if _should_trigger_adaptive(rec_row, top, profile):
        with open(root / "configs" / "sim_multistage.yaml", encoding="utf-8") as f:
            sim_cfg_full = yaml.safe_load(f)
        with open(root / "configs" / "rl3.yaml", encoding="utf-8") as f:
            rl_cfg = yaml.safe_load(f)
        hpm = cost_params["hours_per_worker_month"]
        sim_cfg = with_operating_horizon(sim_cfg_full["simulation"], hpm)
        base_resources = sim_cfg_full["resources"]
        service_cfg = sim_cfg_full["service_time"]
        reward_cfg = rl_cfg.get("reward", {})
        rl_agent = load_rl3_agent(Path(checkpoint_path), rl_cfg)

        month_orders = slice_month_operating_time(orders_all, month_num, sim_cfg["operating_horizon_minutes"])
        seed = seed_offset + month_num
        service_time_map = build_service_time_map(month_orders, service_cfg, seed)
        # Parsed directly from the recommendation row rather than looked up in REGIME_LOOKUP —
        # the recommended regime may be a dynamically-generated candidate (spec §14) not present
        # in the static base-regime table.
        parent_workers = (int(rec_row["picking_workers"]), int(rec_row["packing_workers"]), int(rec_row["dispatch_workers"]))

        analytical = estimate_workers(month_orders, service_cfg, hpm, profile["capacity_planning"]["target_utilisation"])
        max_workers_by_stage = {
            stage: analytical["workers"][stage] + int(profile["adaptive_search"]["max_extra_workers_per_stage"])
            for stage in STAGES
        }
        # Never let the ceiling sit below the already-recommended workforce (the analytical
        # estimate is a screening anchor, not a hard cap — a validated recommendation above it
        # must still be able to search further, spec §13/§18).
        for stage, idx in zip(STAGES, (0, 1, 2)):
            max_workers_by_stage[stage] = max(max_workers_by_stage[stage], parent_workers[idx] + 1)

        if run_mode == "future" and extra_replication_orders:
            orders_by_rep = [month_orders] + [
                slice_month_operating_time(df, month_num, sim_cfg["operating_horizon_minutes"])
                for df in extra_replication_orders
            ]
            # Same seed convention as the base grid (src/rl/evaluate_rl3_monthly_capacity_cost.py:
            # seed = seed_offset + month_num, reused unchanged per replication — the differing
            # order composition per replication is what makes each scenario distinct, spec §11).
            seeds = [seed] * len(orders_by_rep)
            service_time_maps = [service_time_map] + [
                build_service_time_map(o, service_cfg, s) for o, s in zip(orders_by_rep[1:], seeds[1:])
            ]
            parent_results = evaluate_regime_all_policies_multiseed(
                orders_by_rep, service_time_maps, seeds, parent_workers, base_resources, sim_cfg, service_cfg,
                rl_agent, reward_cfg, cost_params, sla_targets,
            )
            search = run_adaptive_capacity_search_validated(
                orders_by_rep, service_time_maps, seeds, base_resources, sim_cfg, service_cfg,
                rl_agent, reward_cfg, rec_row["regime"], parent_results, cost_params, sla_targets, profile,
                max_workers_by_stage=max_workers_by_stage,
            )
        else:
            parent_results = evaluate_regime_all_policies(
                month_orders, parent_workers, base_resources, sim_cfg, service_cfg, service_time_map,
                rl_agent, reward_cfg, seed, cost_params, sla_targets,
            )
            search = run_adaptive_capacity_search(
                month_orders, base_resources, sim_cfg, service_cfg, service_time_map,
                rl_agent, reward_cfg, seed, rec_row["regime"], parent_results, cost_params, sla_targets, profile,
                max_workers_by_stage=max_workers_by_stage,
            )

        final = search["final_result"]
        changed = search["final_regime"] != rec_row["regime"]
        report["adaptive_search"] = {
            "triggered": True,
            "trail": search["trail"],
            "stop_reason": search["stop_reason"],
            "final_regime": search["final_regime"],
            "final_policy": search["final_policy"],
            "regime_changed": changed,
            "simulations_executed": search.get("simulations_executed"),
        }
        report["capacity_level_diagnostics"] = _capacity_level_diagnostics(month_df, search["trail"])

        if changed:
            # final["workers"] is the actual (pick, pack, disp) tuple evaluated — adaptive
            # search can produce regimes beyond the dynamic candidate set, so it's read directly
            # rather than looked up by label anywhere.
            pick, pack, disp = final["workers"]
            report["selected_recommendation"] = {
                "regime": search["final_regime"], "policy": search["final_policy"],
                "picking_workers": pick, "packing_workers": pack, "dispatch_workers": disp,
                "total_sla": float(final["metrics"]["sla_rate"]),
                "urgent_sla": float(final["metrics"]["sla_urgent"]),
                "normal_sla": float(final["metrics"]["sla_normal"]),
                "estimated_total_cost": round(float(final["total_cost"]), 2),
                "feasible": bool(final["feasible"]), "sla_violation": float(final["sla_violation"]),
                "regime_source": "adaptive",
            }
            bottleneck_rows = score_bottlenecks(final["metrics"]["stage_metrics"])
            top = bottleneck_rows[0]
            report["bottleneck_ranking"] = bottleneck_rows
            report["primary_bottleneck"] = top["stage"]
            report["break_even"] = break_even_metrics(
                cost_params["worker_cost_per_hour"], cost_params["hours_per_worker_month"],
                cost_params["cost_late_urgent"], cost_params["cost_late_normal"],
                float(final["urgent_late_orders"]), float(final["normal_late_orders"]),
            )
            policy_comparison = _policy_comparison_from_results(
                search["final_results_by_policy"], final["workers"], sla_targets,
            )
            report["policy_comparison"] = policy_comparison
            report["recommended_policy"] = search["final_policy"]

    if not report["selected_recommendation"]["feasible"]:
        report["explanation"] = f"Best available configuration; SLA targets not fully met. {top['explanation']}"
    else:
        report["explanation"] = top["explanation"]

    return sanitize_for_json(report)
