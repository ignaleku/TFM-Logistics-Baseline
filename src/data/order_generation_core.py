"""
Reusable order-generation core shared by:
  - src/data/generate_orders_seasonal.py  (full-year CLI generator, no volume/arrival jitter)
  - src/data/future_scenario.py           (future-planning generator, with demand/arrival uncertainty)

There is exactly one definition of family/complexity assignment, workload-unit calculation,
and arrival-time sampling. Do not duplicate this logic elsewhere.
"""
from __future__ import annotations

import calendar
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.data.planning_profile import compute_workload_units, get_month_profile

ITEM_LN_SIGMA_DEFAULT = 0.5


def daily_weights(
    rng: np.random.Generator,
    profile: Dict[str, Any],
    month: int,
    base_year: int,
    arrival_cv: float = 0.0,
) -> np.ndarray:
    """Per-day arrival weight vector for a month, with campaign bursts and optional extra
    dispersion (arrival_cv) layered on top for future-scenario uncertainty."""
    cal = profile["calendar_profile"]
    n_days = calendar.monthrange(base_year, month)[1]
    weights = np.ones(n_days, dtype=float)

    if month in set(cal["campaign_months"]):
        burst_k = float(cal["campaign_burst_multiplier"])
        burst_pct = float(cal["campaign_burst_day_pct"])
        n_burst = max(2, int(round(n_days * burst_pct)))
        burst_days = rng.choice(n_days, size=n_burst, replace=False)
        weights[burst_days] *= burst_k

    if arrival_cv > 0:
        # Positive multiplicative noise (lognormal, mean 1) — adds day-to-day dispersion beyond
        # the deterministic campaign-burst pattern, without disturbing the intraday profile.
        sigma = float(arrival_cv)
        mu = -0.5 * sigma ** 2
        weights = weights * rng.lognormal(mu, sigma, size=n_days)

    return weights / weights.sum()


def hourly_weights(profile: Dict[str, Any]) -> np.ndarray:
    raw = np.array(profile["calendar_profile"]["intraday_hourly_weights"], dtype=float)
    return raw / raw.sum()


def generate_month_orders(
    profile: Dict[str, Any],
    month: int,
    n_orders: int,
    rng: np.random.Generator,
    base_year: Optional[int] = None,
    arrival_cv: float = 0.0,
) -> pd.DataFrame:
    """Generate `n_orders` heterogeneous orders for a single month.

    Returns a DataFrame with columns: arrival_time, order_type, sla_minutes, num_items,
    product_class, product_family, complexity_level, picking_units, packing_units,
    dispatch_units — sorted by arrival_time. Caller assigns order_id / month / scenario.
    """
    base_year = base_year or int(profile["meta"]["base_year"])
    mp = get_month_profile(profile, month)
    sla = profile["sla"]
    ic = profile["item_count"]

    daily_w = daily_weights(rng, profile, month, base_year, arrival_cv=arrival_cv)
    n_days = len(daily_w)
    hourly_w = hourly_weights(profile)

    day_counts = rng.multinomial(n_orders, daily_w)

    arrival_times: list[datetime] = []
    for day_idx, count in enumerate(day_counts):
        if count == 0:
            continue
        day = day_idx + 1
        hourly_counts = rng.multinomial(int(count), hourly_w)
        for hour, h_count in enumerate(hourly_counts):
            if h_count == 0:
                continue
            minutes = rng.integers(0, 60, size=int(h_count))
            seconds = rng.integers(0, 60, size=int(h_count))
            for minute, second in zip(minutes, seconds):
                arrival_times.append(datetime(base_year, month, day, int(hour), int(minute), int(second)))

    n = len(arrival_times)
    if n == 0:
        return pd.DataFrame(columns=[
            "arrival_time", "order_type", "sla_minutes", "num_items", "product_class",
            "product_family", "complexity_level", "picking_units", "packing_units", "dispatch_units",
        ])

    urgent_p = float(mp["urgent_share"])
    order_types = rng.choice(["urgent", "normal"], size=n, p=[urgent_p, 1.0 - urgent_p])
    sla_minutes = np.where(order_types == "urgent", int(sla["urgent_minutes"]), int(sla["normal_minutes"]))

    ln_sigma = float(profile["item_count"]["ln_sigma"])
    target_mean = float(mp["mean_num_items"])
    ln_mu = float(np.log(target_mean) - ln_sigma ** 2 / 2)
    raw_items = rng.lognormal(ln_mu, ln_sigma, size=n)
    num_items = np.clip(np.rint(raw_items).astype(int), int(ic["min_items"]), int(ic["max_items"]))

    pa, pb, pc = mp["product_class_mix"]
    product_class = rng.choice(["A", "B", "C"], size=n, p=[pa, pb, pc])

    fam_dist = profile["family_distribution"]
    cpl_dist = profile["complexity_distribution"]

    families = np.empty(n, dtype=object)
    complexities = np.empty(n, dtype=object)
    urgent_mask = order_types == "urgent"
    n_urg = int(urgent_mask.sum())
    n_nrm = n - n_urg

    # RNG call order matters for reproducibility: families (urgent, then normal), then
    # complexities (urgent, then normal) — matches the original generator's draw sequence.
    if n_urg > 0:
        fam_labels, fam_probs = zip(*fam_dist["urgent"].items())
        families[urgent_mask] = rng.choice(list(fam_labels), size=n_urg, p=list(fam_probs))
    if n_nrm > 0:
        fam_labels, fam_probs = zip(*fam_dist["normal"].items())
        families[~urgent_mask] = rng.choice(list(fam_labels), size=n_nrm, p=list(fam_probs))

    if n_urg > 0:
        cpl_labels, cpl_probs = zip(*cpl_dist["urgent"].items())
        complexities[urgent_mask] = rng.choice(list(cpl_labels), size=n_urg, p=list(cpl_probs))
    if n_nrm > 0:
        cpl_labels, cpl_probs = zip(*cpl_dist["normal"].items())
        complexities[~urgent_mask] = rng.choice(list(cpl_labels), size=n_nrm, p=list(cpl_probs))

    picking_u, packing_u, dispatch_u = compute_workload_units(
        profile, num_items, order_types, families, complexities
    )

    df = pd.DataFrame({
        "arrival_time":     pd.to_datetime(pd.Series(arrival_times)),
        "order_type":       order_types,
        "sla_minutes":      sla_minutes,
        "num_items":        num_items,
        "product_class":    product_class,
        "product_family":   families,
        "complexity_level": complexities,
        "picking_units":    picking_u,
        "packing_units":    packing_u,
        "dispatch_units":   dispatch_u,
    })
    return df.sort_values("arrival_time").reset_index(drop=True)
