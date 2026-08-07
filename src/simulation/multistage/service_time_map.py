"""
Common-random-number service-time sampling.

Precomputes picking/packing/dispatch service-time minutes for every order ONCE per
scenario seed, sorted by order_id (never by queue/dequeue order). FIFO, urgent_first, and
RL-3 all look up the same map, so for the same scenario/seed/order/stage the sampled service
time is identical across policies — required for a fair comparison (see README "Fair policy
comparison" section) and used throughout the RL-3 audit, adaptive capacity search, and
future-planning replications.

Without this, each engine drew from its own independently-advancing RNG stream ordered by
dequeue time, which differs by policy (urgent_first/RL-3 reorder queues) — so "same seed"
did NOT mean "same service times" before this module existed.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

STAGES: tuple[str, str, str] = ("picking", "packing", "dispatch")

_UNIT_COLS = {"picking": "picking_units", "packing": "packing_units", "dispatch": "dispatch_units"}


def build_service_time_map(
    orders: pd.DataFrame,
    service_cfg: Dict,
    seed: int,
) -> Dict[Tuple[int, str], float]:
    """Sample service-time minutes for every (order_id, stage).

    Uses the same formula as service_times_multistage.py::sample_service_minutes:
        minutes = max(min_minutes, (base + per_unit * units) * noise)
        noise ~ clip(Normal(1.0, 0.12), noise_clip_lo, noise_clip_hi)
    Orders are processed sorted by order_id, and stages in a fixed order (picking, packing,
    dispatch), so the result depends only on (orders, service_cfg, seed) — never on which
    policy or dequeue order calls it.
    """
    missing = set(_UNIT_COLS.values()) - set(orders.columns)
    if missing:
        raise ValueError(f"orders is missing required workload columns: {sorted(missing)}")

    orders_sorted = orders.sort_values("order_id").reset_index(drop=True)
    order_ids = orders_sorted["order_id"].to_numpy()
    n = len(orders_sorted)
    rng = np.random.default_rng(int(seed))

    result: Dict[Tuple[int, str], float] = {}

    for stage in STAGES:
        cfg = service_cfg[stage]
        base = float(cfg["base_minutes"])
        per_unit = float(cfg["minutes_per_unit"])
        lo = float(cfg.get("noise_clip_lo", 0.80))
        hi = float(cfg.get("noise_clip_hi", 1.25))
        min_min = float(cfg.get("min_minutes", 0.2))

        units = orders_sorted[_UNIT_COLS[stage]].to_numpy(dtype=float)
        noise = np.clip(rng.normal(1.0, 0.12, size=n), lo, hi)
        minutes = np.maximum(min_min, (base + per_unit * units) * noise)

        for oid, m in zip(order_ids, minutes):
            result[(int(oid), stage)] = float(m)

    return result
