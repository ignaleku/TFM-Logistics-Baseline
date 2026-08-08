"""
Analytical (pre-simulation) workforce estimate — spec §13.

Before running any SimPy scenario, estimate the deterministic expected service workload per
stage (same service-time formula as service_times_multistage.py::sample_service_minutes, but
with the noise multiplier fixed at 1.0 instead of sampled), divide by paid capacity per worker
at a target utilisation, and ceil. This is a cheap screening ANCHOR — not the final
recommendation. src/analysis/candidate_generation.py builds a bounded set of dynamic workforce
candidates around this centre, and SimPy (via the existing screening+validation /
adaptive-search machinery) determines the true recommendation after accounting for queueing,
arrival variability, sequencing, and SLA effects.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd

from src.simulation.multistage.operating_time import operating_horizon_minutes

STAGES = ("picking", "packing", "dispatch")
_UNIT_COLS = {"picking": "picking_units", "packing": "packing_units", "dispatch": "dispatch_units"}


def expected_stage_minutes(orders: pd.DataFrame, service_cfg: Dict, stage: str) -> float:
    """Deterministic expected service minutes for one stage, summed over every order, using
    noise=1.0 (i.e. the mean of the service-time formula, not a sampled draw)."""
    cfg = service_cfg[stage]
    base = float(cfg["base_minutes"])
    per_unit = float(cfg["minutes_per_unit"])
    min_min = float(cfg.get("min_minutes", 0.2))

    units = orders[_UNIT_COLS[stage]].to_numpy(dtype=float)
    minutes = (base + per_unit * units).clip(min=min_min)
    return float(minutes.sum())


def estimate_expected_workload_minutes(orders: pd.DataFrame, service_cfg: Dict) -> Dict[str, float]:
    return {stage: expected_stage_minutes(orders, service_cfg, stage) for stage in STAGES}


def estimate_workers(
    orders: pd.DataFrame,
    service_cfg: Dict,
    hours_per_worker_month: float,
    target_utilisation: float,
) -> Dict[str, Any]:
    """Analytical workforce centre (spec §13). Returns per-stage estimated worker counts (min
    1 per stage) plus the underlying workload/capacity figures for transparency/audit."""
    capacity_per_worker_minutes = operating_horizon_minutes(hours_per_worker_month)
    effective_capacity = capacity_per_worker_minutes * float(target_utilisation)

    workload = estimate_expected_workload_minutes(orders, service_cfg)
    workers: Dict[str, int] = {}
    for stage in STAGES:
        if effective_capacity <= 0:
            workers[stage] = 1
        else:
            workers[stage] = max(1, math.ceil(workload[stage] / effective_capacity))

    return {
        "workers": workers,
        "expected_workload_minutes": {s: round(v, 1) for s, v in workload.items()},
        "capacity_per_worker_minutes": round(capacity_per_worker_minutes, 1),
        "target_utilisation": float(target_utilisation),
        "effective_capacity_per_worker_minutes": round(effective_capacity, 1),
    }
