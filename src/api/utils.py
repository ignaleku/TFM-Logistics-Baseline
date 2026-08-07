from __future__ import annotations
import calendar
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data.planning_profile import compute_workload_units, load_planning_profile

REQUIRED_ORDER_COLS = {"order_id", "arrival_time", "order_type", "num_items", "product_class"}

# Fallback family/complexity mix for legacy uploads that omit these columns entirely
# (unweighted by order_type, unlike the calibrated profile — a simple, documented default).
_FALLBACK_FAMILY_MIX = {"standard": 0.60, "fragile": 0.25, "bulky": 0.15}
_FALLBACK_COMPLEXITY_MIX = {"low": 0.45, "medium": 0.38, "high": 0.17}


def enrich_orders_df(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Derive missing product_family, complexity_level, and workload unit columns.

    Called on uploaded CSVs that lack the new columns so that the simulation can
    use the unit-based service-time model even for legacy/external data. Workload-unit
    multipliers are the single definition in configs/planning_profile.yaml — see
    src/data/planning_profile.py::compute_workload_units.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    if "product_family" not in df.columns:
        labels, probs = zip(*_FALLBACK_FAMILY_MIX.items())
        df["product_family"] = rng.choice(list(labels), size=len(df), p=list(probs))

    if "complexity_level" not in df.columns:
        labels, probs = zip(*_FALLBACK_COMPLEXITY_MIX.items())
        df["complexity_level"] = rng.choice(list(labels), size=len(df), p=list(probs))

    # Always recompute units from the current family/complexity/num_items values
    profile = load_planning_profile()
    n = df["num_items"].to_numpy(dtype=float)
    fam = df["product_family"].astype(str).to_numpy()
    cpl = df["complexity_level"].astype(str).to_numpy()
    otype = (
        df["order_type"].astype(str).to_numpy()
        if "order_type" in df.columns
        else np.full(len(df), "normal")
    )

    picking_u, packing_u, dispatch_u = compute_workload_units(profile, n, otype, fam, cpl)
    df["picking_units"] = picking_u
    df["packing_units"] = packing_u
    df["dispatch_units"] = dispatch_u

    return df


def validate_orders_csv(path: Path) -> Dict[str, Any]:
    try:
        df = pd.read_csv(path, parse_dates=["arrival_time"])
    except Exception as exc:
        return {"valid": False, "error": f"Could not read CSV: {exc}"}

    missing = REQUIRED_ORDER_COLS - set(df.columns)
    if missing:
        return {"valid": False, "error": f"Missing required columns: {sorted(missing)}"}

    total_rows = len(df)

    date_range: str | None = None
    month_names: list[str] = []
    if "arrival_time" in df.columns and not df["arrival_time"].isna().all():
        min_date = df["arrival_time"].min()
        max_date = df["arrival_time"].max()
        date_range = f"{min_date.date()} → {max_date.date()}"
        months = sorted(df["arrival_time"].dt.month.dropna().unique().astype(int))
        month_names = [calendar.month_name[m] for m in months]

    urgent_share = 0.0
    if "order_type" in df.columns:
        urgent_share = round(float((df["order_type"] == "urgent").mean()), 4)

    new_cols = {"product_family", "complexity_level", "picking_units", "packing_units", "dispatch_units"}
    had_workload = new_cols.issubset(set(df.columns))

    return {
        "valid": True,
        "total_rows": total_rows,
        "date_range": date_range,
        "detected_months": month_names,
        "urgent_share": urgent_share,
        "had_workload_columns": had_workload,
    }
