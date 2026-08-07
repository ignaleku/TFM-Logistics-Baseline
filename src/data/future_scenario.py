"""
Future-planning scenario generator.

Transforms a small set of aggregate runtime inputs (planning_month, expected_annual_orders,
optional monthly_orders_override, uncertainty_level) into simulated individual orders, using
the SAME family/complexity/workload-unit logic as the historical seasonal generator
(src/data/order_generation_core.py) and the SAME calibrated assumptions
(configs/planning_profile.yaml).

The system does not predict every individual future order — it turns an aggregate demand
forecast into plausible simulated operational scenarios. Volume uncertainty is expressed via
a small number of replications (different seeds), each independently jittered by the
uncertainty level's demand/arrival coefficients of variation.
"""
from __future__ import annotations

import calendar
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.data.order_generation_core import generate_month_orders
from src.data.planning_profile import get_month_profile, load_planning_profile


def resolve_month(month: Union[str, int], profile: Dict[str, Any]) -> int:
    if isinstance(month, (int, np.integer)):
        m = int(month)
        if m not in profile["months"]:
            raise ValueError(f"Invalid month number: {m}. Expected 1-12.")
        return m

    key = str(month).strip().lower()
    if key.isdigit():
        m = int(key)
        if m not in profile["months"]:
            raise ValueError(f"Invalid month number: {m}. Expected 1-12.")
        return m

    name_to_num = {v["name"].lower(): k for k, v in profile["months"].items()}
    abbr_to_num = {v["name"][:3].lower(): k for k, v in profile["months"].items()}
    if key in name_to_num:
        return name_to_num[key]
    if key in abbr_to_num:
        return abbr_to_num[key]
    raise ValueError(f"Unrecognised month: {month!r}")


def resolve_monthly_volume(
    profile: Dict[str, Any],
    month: int,
    expected_annual_orders: float,
    monthly_orders_override: Optional[float],
) -> tuple[int, str]:
    """Returns (expected_monthly_orders, source) where source is 'monthly_override' or
    'annual_forecast'."""
    if monthly_orders_override is not None and float(monthly_orders_override) > 0:
        return int(round(float(monthly_orders_override))), "monthly_override"
    if expected_annual_orders is None or float(expected_annual_orders) <= 0:
        raise ValueError("expected_annual_orders must be > 0 when no monthly_orders_override is given.")
    mp = get_month_profile(profile, month)
    return int(round(float(expected_annual_orders) * float(mp["annual_share"]))), "annual_forecast"


def replication_seeds(profile: Dict[str, Any], month: int, n_replications: int) -> List[int]:
    base_seed = int(profile["meta"]["base_seed"])
    offset = int(profile["future_planning"]["replication_seed_offset"])
    return [base_seed + month * 10 + r * offset for r in range(n_replications)]


def build_preview(
    planning_month: Union[str, int],
    expected_annual_orders: float,
    monthly_orders_override: Optional[float] = None,
    uncertainty_level: str = "standard",
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Derived, read-only planning assumptions for the Future Planning preview — no simulation."""
    profile = profile or load_planning_profile()
    month = resolve_month(planning_month, profile)
    mp = get_month_profile(profile, month)

    unc = profile["uncertainty_levels"].get(uncertainty_level)
    if unc is None:
        raise ValueError(
            f"Unknown uncertainty_level: {uncertainty_level!r}. "
            f"Expected one of {sorted(profile['uncertainty_levels'])}."
        )

    monthly_orders, source = resolve_monthly_volume(
        profile, month, expected_annual_orders, monthly_orders_override
    )

    n_days = calendar.monthrange(int(profile["meta"]["base_year"]), month)[1]
    operating_hours_per_day = float(profile["calendar_profile"]["operating_hours_per_day"])
    total_operating_hours = n_days * operating_hours_per_day
    orders_per_hour = (monthly_orders / total_operating_hours) if total_operating_hours > 0 else 0.0

    u = float(mp["urgent_share"])
    fam_normal = profile["family_distribution"]["normal"]
    fam_urgent = profile["family_distribution"]["urgent"]
    cpl_normal = profile["complexity_distribution"]["normal"]
    cpl_urgent = profile["complexity_distribution"]["urgent"]

    family_shares = {
        k: round(u * fam_urgent.get(k, 0.0) + (1 - u) * fam_normal.get(k, 0.0), 4)
        for k in dict.fromkeys(list(fam_normal) + list(fam_urgent))
    }
    complexity_shares = {
        k: round(u * cpl_urgent.get(k, 0.0) + (1 - u) * cpl_normal.get(k, 0.0), 4)
        for k in dict.fromkeys(list(cpl_normal) + list(cpl_urgent))
    }

    return {
        "month": month,
        "month_name": mp["name"],
        "expected_monthly_orders": monthly_orders,
        "source": source,
        "annual_share": mp["annual_share"],
        "urgent_share": mp["urgent_share"],
        "expected_avg_items": mp["mean_num_items"],
        "product_family_shares": family_shares,
        "complexity_shares": complexity_shares,
        "operating_days": n_days,
        "operating_hours_per_day": operating_hours_per_day,
        "expected_orders_per_operating_hour": round(orders_per_hour, 2),
        "uncertainty_level": uncertainty_level,
        "uncertainty_assumptions": unc,
        "sla_targets": {"urgent_target": profile["sla"]["urgent_target"], "normal_target": profile["sla"]["normal_target"]},
        "replications": int(profile["future_planning"]["default_replications"]),
    }


def generate_future_scenario_orders(
    planning_month: Union[str, int],
    expected_annual_orders: float,
    monthly_orders_override: Optional[float] = None,
    uncertainty_level: str = "standard",
    replication: int = 0,
    profile: Optional[Dict[str, Any]] = None,
    order_id_start: int = 1,
) -> pd.DataFrame:
    """Generate one replication (seeded, reproducible) of simulated future orders for a month.

    Reuses generate_month_orders (src/data/order_generation_core.py) — the exact same
    family/complexity/workload logic as the historical seasonal generator. Demand uncertainty
    jitters the sampled monthly order count (lognormal around the expected volume); arrival
    uncertainty adds extra day-to-day dispersion on top of the seasonal campaign-burst pattern.
    """
    profile = profile or load_planning_profile()
    month = resolve_month(planning_month, profile)

    unc = profile["uncertainty_levels"].get(uncertainty_level)
    if unc is None:
        raise ValueError(
            f"Unknown uncertainty_level: {uncertainty_level!r}. "
            f"Expected one of {sorted(profile['uncertainty_levels'])}."
        )
    demand_cv = float(unc["demand_cv"])
    arrival_cv = float(unc["arrival_cv"])

    monthly_target, source = resolve_monthly_volume(
        profile, month, expected_annual_orders, monthly_orders_override
    )

    seed = replication_seeds(profile, month, replication + 1)[replication]
    rng = np.random.default_rng(seed)

    if demand_cv > 0:
        sigma = demand_cv
        mu = float(np.log(max(1, monthly_target))) - 0.5 * sigma ** 2
        n_orders = max(1, int(round(rng.lognormal(mu, sigma))))
    else:
        n_orders = monthly_target

    base_year = int(profile["meta"]["base_year"])
    df = generate_month_orders(profile, month, n_orders, rng, base_year=base_year, arrival_cv=arrival_cv)
    df = df.reset_index(drop=True)

    mp = get_month_profile(profile, month)
    df.insert(0, "order_id", np.arange(order_id_start, order_id_start + len(df)))
    df["month"] = df["arrival_time"].dt.month
    df["weekday"] = df["arrival_time"].dt.day_name().str.lower()
    df["hour"] = df["arrival_time"].dt.hour
    df["scenario"] = f"future_{mp['name'].lower()}_{uncertainty_level}_r{replication}"
    df.attrs["scenario_seed"] = seed
    df.attrs["source"] = source
    df.attrs["expected_monthly_orders"] = monthly_target
    return df
