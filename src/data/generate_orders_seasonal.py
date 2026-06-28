"""
Seasonal synthetic order generator for RL-3 logistics baseline.

Strong winter peak (Jan/Feb/Dec), high-demand November (pre-Christmas ramp),
low-demand summer valley (May–Aug), moderate autumn ramp-up (Sep–Oct).

Heterogeneous orders: product_family (standard/fragile/bulky) and
complexity_level (low/medium/high) drive differentiated workload at each stage.

Output:
  data/orders_base_seasonal.csv          — 240,000 orders
  data/orders_base_seasonal_summary.csv  — one row per month with key statistics

Usage:
  python -m src.data.generate_orders_seasonal
"""
from __future__ import annotations

import calendar
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────────
SEED         = 42
TOTAL_ORDERS = 240_000
BASE_YEAR    = 2026

MONTHLY_SHARE: dict[int, float] = {
    1: 0.130, 2: 0.110, 3: 0.070, 4: 0.070,
    5: 0.045, 6: 0.040, 7: 0.035, 8: 0.035,
    9: 0.070, 10: 0.085, 11: 0.140, 12: 0.170,
}

URGENT_SHARE: dict[int, float] = {
    1: 0.20, 2: 0.18, 3: 0.12, 4: 0.12,
    5: 0.09, 6: 0.09, 7: 0.08, 8: 0.08,
    9: 0.12, 10: 0.15, 11: 0.22, 12: 0.25,
}

# Target mean num_items per month
_ITEM_TARGET_MEAN: dict[int, float] = {
    1: 3.6, 2: 3.6, 3: 2.8, 4: 2.8,
    5: 2.2, 6: 2.2, 7: 2.2, 8: 2.2,
    9: 3.0, 10: 3.0, 11: 4.0, 12: 4.4,
}
_ITEM_LN_SIGMA = 0.5
# For lognormal, E[X] = exp(μ + σ²/2)  →  μ = ln(target) − σ²/2
_ITEM_LN_MU: dict[int, float] = {
    m: float(np.log(v) - _ITEM_LN_SIGMA ** 2 / 2)
    for m, v in _ITEM_TARGET_MEAN.items()
}
ITEM_MIN = 1
ITEM_MAX = 12

# Product class mix (A, B, C) — kept for backward compat
PRODUCT_MIX: dict[int, tuple[float, float, float]] = {
    1:  (0.43, 0.37, 0.20), 2:  (0.43, 0.37, 0.20),
    3:  (0.50, 0.35, 0.15), 4:  (0.50, 0.35, 0.15),
    5:  (0.50, 0.35, 0.15), 6:  (0.50, 0.35, 0.15),
    7:  (0.50, 0.35, 0.15), 8:  (0.50, 0.35, 0.15),
    9:  (0.50, 0.35, 0.15), 10: (0.45, 0.37, 0.18),
    11: (0.40, 0.38, 0.22), 12: (0.40, 0.38, 0.22),
}

# Product family distributions (split by order_type to create richer heterogeneity)
_FAMILY_NORMAL = ["standard", "fragile", "bulky"]
_FAMILY_P_NORMAL = [0.65, 0.20, 0.15]
_FAMILY_URGENT = ["standard", "fragile", "bulky"]
_FAMILY_P_URGENT = [0.45, 0.35, 0.20]

# Complexity distributions (split by order_type)
_COMPL_NORMAL = ["low", "medium", "high"]
_COMPL_P_NORMAL = [0.50, 0.35, 0.15]
_COMPL_URGENT = ["low", "medium", "high"]
_COMPL_P_URGENT = [0.25, 0.45, 0.30]

# Workload unit multipliers
_PICK_FAM  = {"standard": 1.0, "fragile": 1.1, "bulky": 1.3}
_PACK_FAM  = {"standard": 1.0, "fragile": 1.8, "bulky": 1.6}
_DISP_FAM  = {"standard": 1.0, "fragile": 1.1, "bulky": 1.2}
_PICK_CPL  = {"low": 0.9, "medium": 1.1, "high": 1.4}
_PACK_CPL  = {"low": 0.8, "medium": 1.2, "high": 1.7}
_DISP_CPL  = {"low": 0.9, "medium": 1.1, "high": 1.4}
_DISP_URG  = {"urgent": 1.3, "normal": 1.0}

# Campaign months get bursty days (higher within-month concentration)
_CAMPAIGN_MONTHS    = frozenset({1, 2, 11, 12})
_CAMPAIGN_BURST_K   = 3.0   # burst days are K× the average day weight
_CAMPAIGN_BURST_PCT = 0.15  # ~15% of days in the month are burst days

SLA_URGENT = 240
SLA_NORMAL = 1440

# Intraday hourly weights (0–23); biased toward working hours
_HOURLY_RAW = np.array([
    0.10, 0.05, 0.05, 0.05, 0.15, 0.50,   # 00–05
    1.00, 2.50, 4.00, 4.50, 5.00, 5.00,   # 06–11
    4.50, 4.00, 4.00, 4.00, 4.50, 5.00,   # 12–17
    4.50, 3.00, 2.00, 1.00, 0.50, 0.20,   # 18–23
], dtype=float)
_HOURLY_WEIGHTS = _HOURLY_RAW / _HOURLY_RAW.sum()


# ── helpers ────────────────────────────────────────────────────────────────────

def _daily_weights(rng: np.random.Generator, month: int) -> np.ndarray:
    n_days = calendar.monthrange(BASE_YEAR, month)[1]
    weights = np.ones(n_days, dtype=float)
    if month in _CAMPAIGN_MONTHS:
        n_burst = max(2, int(round(n_days * _CAMPAIGN_BURST_PCT)))
        burst_days = rng.choice(n_days, size=n_burst, replace=False)
        weights[burst_days] *= _CAMPAIGN_BURST_K
    return weights / weights.sum()


def _compute_units(
    num_items: np.ndarray,
    order_types: np.ndarray,
    families: np.ndarray,
    complexities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = num_items.astype(float)
    fp = np.vectorize(_PICK_FAM.get)(families, 1.0).astype(float)
    pp = np.vectorize(_PACK_FAM.get)(families, 1.0).astype(float)
    dp = np.vectorize(_DISP_FAM.get)(families, 1.0).astype(float)
    cp = np.vectorize(_PICK_CPL.get)(complexities, 1.0).astype(float)
    cp2 = np.vectorize(_PACK_CPL.get)(complexities, 1.0).astype(float)
    cp3 = np.vectorize(_DISP_CPL.get)(complexities, 1.0).astype(float)
    dup = np.vectorize(_DISP_URG.get)(order_types, 1.0).astype(float)

    picking_units  = np.round(n * fp * cp, 2)
    packing_units  = np.round((1 + 0.25 * n) * pp * cp2, 2)
    dispatch_units = np.round(dup * dp * cp3, 2)

    return picking_units, packing_units, dispatch_units


def _generate_month(rng: np.random.Generator, month: int, n_orders: int) -> pd.DataFrame:
    daily_w   = _daily_weights(rng, month)
    n_days    = len(daily_w)
    day_counts = rng.multinomial(n_orders, daily_w)

    arrival_times: list[datetime] = []
    for day_idx, count in enumerate(day_counts):
        if count == 0:
            continue
        day = day_idx + 1
        hourly_counts = rng.multinomial(int(count), _HOURLY_WEIGHTS)
        for hour, h_count in enumerate(hourly_counts):
            if h_count == 0:
                continue
            minutes = rng.integers(0, 60, size=int(h_count))
            seconds = rng.integers(0, 60, size=int(h_count))
            for minute, second in zip(minutes, seconds):
                arrival_times.append(
                    datetime(BASE_YEAR, month, day, int(hour), int(minute), int(second))
                )

    n = len(arrival_times)
    urgent_p   = URGENT_SHARE[month]
    order_types = rng.choice(["urgent", "normal"], size=n, p=[urgent_p, 1.0 - urgent_p])
    sla_minutes = np.where(order_types == "urgent", SLA_URGENT, SLA_NORMAL)

    raw_items = rng.lognormal(_ITEM_LN_MU[month], _ITEM_LN_SIGMA, size=n)
    num_items = np.clip(np.rint(raw_items).astype(int), ITEM_MIN, ITEM_MAX)

    pa, pb, pc = PRODUCT_MIX[month]
    product_class = rng.choice(["A", "B", "C"], size=n, p=[pa, pb, pc])

    # Product family: urgent orders skew toward fragile/bulky
    families = np.empty(n, dtype=object)
    urgent_mask = order_types == "urgent"
    n_urg = int(urgent_mask.sum())
    n_nrm = n - n_urg
    if n_urg > 0:
        families[urgent_mask] = rng.choice(_FAMILY_URGENT, size=n_urg, p=_FAMILY_P_URGENT)
    if n_nrm > 0:
        families[~urgent_mask] = rng.choice(_FAMILY_NORMAL, size=n_nrm, p=_FAMILY_P_NORMAL)

    # Complexity: urgent orders skew toward medium/high
    complexities = np.empty(n, dtype=object)
    if n_urg > 0:
        complexities[urgent_mask] = rng.choice(_COMPL_URGENT, size=n_urg, p=_COMPL_P_URGENT)
    if n_nrm > 0:
        complexities[~urgent_mask] = rng.choice(_COMPL_NORMAL, size=n_nrm, p=_COMPL_P_NORMAL)

    picking_u, packing_u, dispatch_u = _compute_units(
        num_items, order_types, families, complexities
    )

    df = pd.DataFrame({
        "arrival_time":    pd.to_datetime(pd.Series(arrival_times)),
        "order_type":      order_types,
        "sla_minutes":     sla_minutes,
        "num_items":       num_items,
        "product_class":   product_class,
        "product_family":  families,
        "complexity_level": complexities,
        "picking_units":   picking_u,
        "packing_units":   packing_u,
        "dispatch_units":  dispatch_u,
    })
    return df.sort_values("arrival_time").reset_index(drop=True)


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in range(1, 13):
        mdf = df[df["month"] == m]
        n   = len(mdf)
        urg = (mdf["order_type"] == "urgent").mean()
        mi  = mdf["num_items"].mean()

        fam_vc = mdf["product_family"].value_counts(normalize=True)
        cpl_vc = mdf["complexity_level"].value_counts(normalize=True)

        rows.append({
            "month":           m,
            "month_name":      calendar.month_name[m],
            "orders":          n,
            "urgent_share":    round(float(urg), 4),
            "mean_num_items":  round(float(mi), 3),
            "pct_standard":    round(float(fam_vc.get("standard", 0.0)), 4),
            "pct_fragile":     round(float(fam_vc.get("fragile",  0.0)), 4),
            "pct_bulky":       round(float(fam_vc.get("bulky",    0.0)), 4),
            "pct_low":         round(float(cpl_vc.get("low",    0.0)), 4),
            "pct_medium":      round(float(cpl_vc.get("medium", 0.0)), 4),
            "pct_high":        round(float(cpl_vc.get("high",   0.0)), 4),
            "avg_picking_units":  round(float(mdf["picking_units"].mean()),  3),
            "avg_packing_units":  round(float(mdf["packing_units"].mean()),  3),
            "avg_dispatch_units": round(float(mdf["dispatch_units"].mean()), 3),
        })
    return pd.DataFrame(rows)


def _print_summary(summary: pd.DataFrame) -> None:
    print(f"\n{'Month':<11} {'Orders':>7}  {'Urgent':>7}  {'Items':>6}  "
          f"{'Std':>6} {'Frag':>6} {'Bulk':>6}  "
          f"{'Low':>6} {'Med':>6} {'High':>6}  "
          f"{'pick_u':>7} {'pack_u':>7} {'disp_u':>7}")
    print("-" * 104)
    for _, r in summary.iterrows():
        print(
            f"  {r['month_name']:<9} {int(r['orders']):>7,}  "
            f"{r['urgent_share']:>7.1%}  {r['mean_num_items']:>6.2f}  "
            f"{r['pct_standard']:>6.1%} {r['pct_fragile']:>6.1%} {r['pct_bulky']:>6.1%}  "
            f"{r['pct_low']:>6.1%} {r['pct_medium']:>6.1%} {r['pct_high']:>6.1%}  "
            f"{r['avg_picking_units']:>7.2f} {r['avg_packing_units']:>7.2f} {r['avg_dispatch_units']:>7.2f}"
        )
    print("-" * 104)
    print(f"  {'Total':<9} {int(summary['orders'].sum()):>7,}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = np.random.default_rng(SEED)

    monthly_target = {m: int(round(TOTAL_ORDERS * MONTHLY_SHARE[m])) for m in range(1, 13)}
    diff = TOTAL_ORDERS - sum(monthly_target.values())
    monthly_target[12] += diff  # absorb rounding drift in December

    frames = []
    for month in range(1, 13):
        frames.append(_generate_month(rng, month, monthly_target[month]))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("arrival_time").reset_index(drop=True)
    df.insert(0, "order_id", np.arange(1, len(df) + 1))
    df["month"]    = df["arrival_time"].dt.month
    df["weekday"]  = df["arrival_time"].dt.day_name().str.lower()
    df["hour"]     = df["arrival_time"].dt.hour
    df["scenario"] = "seasonal_base"

    df = df[[
        "order_id", "arrival_time", "month", "weekday", "hour",
        "order_type", "sla_minutes", "num_items", "product_class",
        "product_family", "complexity_level",
        "picking_units", "packing_units", "dispatch_units",
        "scenario",
    ]]

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "orders_base_seasonal.csv"
    df.to_csv(csv_path, index=False)

    summary = _build_summary(df)
    _print_summary(summary)
    summary_path = out_dir / "orders_base_seasonal_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nDataset  -> {csv_path}  ({len(df):,} orders)")
    print(f"Summary  -> {summary_path}")


if __name__ == "__main__":
    main()
