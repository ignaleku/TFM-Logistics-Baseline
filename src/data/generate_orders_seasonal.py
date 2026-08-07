"""
Seasonal synthetic order generator for RL-3 logistics baseline.

Strong winter peak (Jan/Feb/Dec), high-demand November (pre-Christmas ramp),
low-demand summer valley (May–Aug), moderate autumn ramp-up (Sep–Oct).

Heterogeneous orders: product_family (standard/fragile/bulky) and
complexity_level (low/medium/high) drive differentiated workload at each stage.

All seasonal/operational assumptions are read from configs/planning_profile.yaml — the
single source of truth also used by the future-planning generator (src/data/future_scenario.py)
and by uploaded-CSV enrichment (src/api/utils.py).

Output:
  data/orders_base_seasonal.csv          — 240,000 orders
  data/orders_base_seasonal_summary.csv  — one row per month with key statistics

Usage:
  python -m src.data.generate_orders_seasonal
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.order_generation_core import generate_month_orders
from src.data.planning_profile import load_planning_profile


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


def main() -> None:
    profile = load_planning_profile()
    base_year = int(profile["meta"]["base_year"])
    total_orders = int(profile["meta"]["total_annual_orders"])
    seed = int(profile["meta"]["base_seed"])

    rng = np.random.default_rng(seed)

    monthly_target = {
        m: int(round(total_orders * profile["months"][m]["annual_share"]))
        for m in range(1, 13)
    }
    diff = total_orders - sum(monthly_target.values())
    monthly_target[12] += diff  # absorb rounding drift in December

    frames = []
    for month in range(1, 13):
        frames.append(generate_month_orders(profile, month, monthly_target[month], rng, base_year=base_year))

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
