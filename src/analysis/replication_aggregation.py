"""
Aggregates multiple future-scenario replications into mean/p90 summary rows (spec §4.1).

A future forecast is uncertain — we never present one random realisation as certainty.
evaluate_monthly_capacity_cost() is run once per replication (different scenario seed each
time, from src/data/future_scenario.py), and this module folds the per-replication rows for
the same (month, regime, policy) into one row carrying: mean total cost, p90 total cost
(when >=3 replications), mean SLA / urgent SLA / normal SLA, probability of meeting SLA
targets, mean late orders, and mean bottleneck metrics. Output column names match the
single-replication schema (CSV_COLS in evaluate_rl3_monthly_capacity_cost.py) so the existing
frontend/export pipeline needs no changes for future-planning results — extra columns
(p90_total_cost, prob_meets_sla_targets, replication_count) are appended.
"""
from __future__ import annotations

from typing import List

import pandas as pd

_GROUP_COLS = [
    "month", "month_name", "regime", "policy",
    "picking_workers", "packing_workers", "dispatch_workers", "total_workers",
]

_MEAN_COLS = [
    "total_orders", "urgent_orders", "normal_orders", "urgent_share",
    "total_sla", "urgent_sla", "normal_sla",
    "mean_system_time_min", "p90_system_time_min",
    "urgent_late_orders", "normal_late_orders",
    "completed_orders", "unfinished_orders", "unfinished_urgent_orders",
    "unfinished_normal_orders", "backlog_share",
    "estimated_late_cost", "estimated_worker_cost", "estimated_total_cost",
    "sla_violation",
] + [
    f"{stage}_{col}" for stage in ("picking", "packing", "dispatch")
    for col in ("utilisation", "avg_wait_min", "p95_wait_min", "avg_queue_len", "max_queue_len", "late_wait_share", "pressure_score")
]


def aggregate_replications(per_replication_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Each input DataFrame is one evaluate_monthly_capacity_cost() result for one scenario
    replication (same month/regime/policy grid, different seed). Returns one row per
    (month, regime, policy) with mean_* columns replacing the raw values, plus
    p90_total_cost, prob_meets_sla_targets, replication_count."""
    if not per_replication_dfs:
        raise ValueError("aggregate_replications: no replications given.")

    combined = pd.concat(per_replication_dfs, ignore_index=True)
    n_reps = len(per_replication_dfs)

    rows = []
    for keys, grp in combined.groupby(_GROUP_COLS, dropna=False):
        row = dict(zip(_GROUP_COLS, keys))
        for col in _MEAN_COLS:
            if col in grp.columns:
                row[col] = float(grp[col].mean())
        row["p90_total_cost"] = float(grp["estimated_total_cost"].quantile(0.9)) if n_reps >= 3 else None
        row["prob_meets_sla_targets"] = float(grp["feasible"].mean())
        row["feasible"] = bool(row["prob_meets_sla_targets"] >= 0.5)
        row["replication_count"] = n_reps
        row["urgent_sla_target"] = float(grp["urgent_sla_target"].iloc[0])
        row["normal_sla_target"] = float(grp["normal_sla_target"].iloc[0])
        row["cost_late_urgent"] = float(grp["cost_late_urgent"].iloc[0])
        row["cost_late_normal"] = float(grp["cost_late_normal"].iloc[0])
        row["worker_cost_per_hour"] = float(grp["worker_cost_per_hour"].iloc[0])
        row["hours_per_worker_month"] = float(grp["hours_per_worker_month"].iloc[0])
        for extra in ("p_urgent_overall", "p_urgent_pick", "p_urgent_pack", "p_urgent_dispatch",
                      "decisions_total", "decisions_pick", "decisions_pack", "decisions_dispatch"):
            if extra in grp.columns:
                row[extra] = float(grp[extra].mean())
        rows.append(row)

    return pd.DataFrame(rows)
