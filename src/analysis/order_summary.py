"""
Per-month order distribution summary (orders, urgent share, family/complexity mix, average
workload units) — the same statistics shape as
src/data/generate_orders_seasonal.py::_build_summary, generalised to any subset of months so
it can summarise a scoped run (a single future-planning month, or a chosen set of historical
months) rather than always assuming the full 12-month baseline.

Used by src/api/runners.py to build the run-scoped order summary attached to
data/api_runs/latest/run_manifest.json (see spec §4/§5 — Demand & Complexity must reflect the
current run, not the whole year).
"""
from __future__ import annotations

import calendar

import pandas as pd


def build_order_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per month present in `df` (sorted ascending), with the same columns as
    orders_base_seasonal_summary.csv. `df` must already carry product_family,
    complexity_level, picking_units, packing_units, dispatch_units (i.e. an
    enriched/generated orders frame — see src/data/order_generation_core.py)."""
    rows = []
    for m in sorted(int(x) for x in df["month"].unique()):
        mdf = df[df["month"] == m]
        n = len(mdf)
        urg = (mdf["order_type"] == "urgent").mean() if n else 0.0
        mi = mdf["num_items"].mean() if n else 0.0

        fam_vc = mdf["product_family"].value_counts(normalize=True)
        cpl_vc = mdf["complexity_level"].value_counts(normalize=True)

        rows.append({
            "month": m,
            "month_name": calendar.month_name[m],
            "orders": n,
            "urgent_share": round(float(urg), 4),
            "mean_num_items": round(float(mi), 3),
            "pct_standard": round(float(fam_vc.get("standard", 0.0)), 4),
            "pct_fragile": round(float(fam_vc.get("fragile", 0.0)), 4),
            "pct_bulky": round(float(fam_vc.get("bulky", 0.0)), 4),
            "pct_low": round(float(cpl_vc.get("low", 0.0)), 4),
            "pct_medium": round(float(cpl_vc.get("medium", 0.0)), 4),
            "pct_high": round(float(cpl_vc.get("high", 0.0)), 4),
            "avg_picking_units": round(float(mdf["picking_units"].mean()), 3) if n else 0.0,
            "avg_packing_units": round(float(mdf["packing_units"].mean()), 3) if n else 0.0,
            "avg_dispatch_units": round(float(mdf["dispatch_units"].mean()), 3) if n else 0.0,
        })
    return pd.DataFrame(rows)
