"""
Loader and validator for configs/planning_profile.yaml — the single source of truth for
client-calibrated seasonal/operational planning assumptions.

Used by:
  - src/data/generate_orders_seasonal.py  (full-year CLI generator)
  - src/data/future_scenario.py           (future-planning aggregate-forecast generator)
  - src/api/utils.py                      (enrich_orders_df — uploaded historical CSVs)
  - src/api/main.py                       (GET /planning/profile, POST /planning/preview)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "configs" / "planning_profile.yaml"


def _validate_profile(profile: Dict[str, Any]) -> None:
    months = profile["months"]
    month_keys = set(months.keys())
    if month_keys != set(range(1, 13)):
        raise ValueError(f"planning_profile.months must have keys 1..12, got {sorted(month_keys)}")

    total_annual = sum(float(m["annual_share"]) for m in months.values())
    if not (0.99 <= total_annual <= 1.01):
        raise ValueError(f"planning_profile monthly annual_share values sum to {total_annual}, expected ~1.0")

    for m, mp in months.items():
        u = float(mp["urgent_share"])
        if not (0.0 <= u <= 1.0):
            raise ValueError(f"month {m}: urgent_share={u} out of [0,1]")
        pcm = mp["product_class_mix"]
        if abs(sum(pcm) - 1.0) > 0.01:
            raise ValueError(f"month {m}: product_class_mix sums to {sum(pcm)}, expected ~1.0")

    for grp in ("normal", "urgent"):
        fam_sum = sum(float(v) for v in profile["family_distribution"][grp].values())
        if abs(fam_sum - 1.0) > 0.01:
            raise ValueError(f"family_distribution.{grp} sums to {fam_sum}, expected ~1.0")
        cpl_sum = sum(float(v) for v in profile["complexity_distribution"][grp].values())
        if abs(cpl_sum - 1.0) > 0.01:
            raise ValueError(f"complexity_distribution.{grp} sums to {cpl_sum}, expected ~1.0")

    bw = profile["bottleneck_score"]
    bw_sum = sum(float(v) for v in bw.values())
    if abs(bw_sum - 1.0) > 0.01:
        raise ValueError(f"bottleneck_score weights sum to {bw_sum}, expected 1.0")

    for level, m in profile["uncertainty_levels"].items():
        if not (0.0 <= float(m["demand_cv"]) <= 1.0) or not (0.0 <= float(m["arrival_cv"]) <= 1.0):
            raise ValueError(f"uncertainty_levels.{level} has an out-of-range coefficient of variation")


@lru_cache(maxsize=1)
def load_planning_profile(path: str | None = None) -> Dict[str, Any]:
    p = Path(path) if path else PROFILE_PATH
    with open(p, "r", encoding="utf-8") as f:
        profile = yaml.safe_load(f)
    _validate_profile(profile)
    return profile


def get_month_profile(profile: Dict[str, Any], month: int) -> Dict[str, Any]:
    if month not in profile["months"]:
        raise ValueError(f"Unknown month {month!r}; expected 1-12.")
    return profile["months"][month]


def compute_workload_units(
    profile: Dict[str, Any],
    num_items: np.ndarray,
    order_types: np.ndarray,
    families: np.ndarray,
    complexities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised workload-unit calculation — the single definition used everywhere.

    picking_units  = num_items * family_mult * complexity_mult
    packing_units  = (1 + 0.25*num_items) * family_mult * complexity_mult
    dispatch_units = urgency_mult * family_mult * complexity_mult
    """
    wm = profile["workload_multipliers"]
    n = num_items.astype(float)

    fp = np.vectorize(wm["picking"]["family"].get)(families, 1.0).astype(float)
    pp = np.vectorize(wm["packing"]["family"].get)(families, 1.0).astype(float)
    dp = np.vectorize(wm["dispatch"]["family"].get)(families, 1.0).astype(float)

    cp = np.vectorize(wm["picking"]["complexity"].get)(complexities, 1.0).astype(float)
    cp2 = np.vectorize(wm["packing"]["complexity"].get)(complexities, 1.0).astype(float)
    cp3 = np.vectorize(wm["dispatch"]["complexity"].get)(complexities, 1.0).astype(float)

    dup = np.vectorize(wm["dispatch"]["urgency"].get)(order_types, 1.0).astype(float)

    picking_units = np.round(n * fp * cp, 2)
    packing_units = np.round((1 + 0.25 * n) * pp * cp2, 2)
    dispatch_units = np.round(dup * dp * cp3, 2)

    return picking_units, packing_units, dispatch_units
