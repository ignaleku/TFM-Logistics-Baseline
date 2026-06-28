from __future__ import annotations
import calendar
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REQUIRED_ORDER_COLS = {"order_id", "arrival_time", "order_type", "num_items", "product_class"}

# Workload unit multipliers — must match generate_orders_seasonal.py
_PICK_FAM  = {"standard": 1.0, "fragile": 1.1, "bulky": 1.3}
_PACK_FAM  = {"standard": 1.0, "fragile": 1.8, "bulky": 1.6}
_DISP_FAM  = {"standard": 1.0, "fragile": 1.1, "bulky": 1.2}
_PICK_CPL  = {"low": 0.9, "medium": 1.1, "high": 1.4}
_PACK_CPL  = {"low": 0.8, "medium": 1.2, "high": 1.7}
_DISP_CPL  = {"low": 0.9, "medium": 1.1, "high": 1.4}
_DISP_URG  = {"urgent": 1.3, "normal": 1.0}


def enrich_orders_df(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Derive missing product_family, complexity_level, and workload unit columns.

    Called on uploaded CSVs that lack the new columns so that the simulation can
    use the unit-based service-time model even for legacy/external data.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    if "product_family" not in df.columns:
        df["product_family"] = rng.choice(
            ["standard", "fragile", "bulky"],
            size=len(df),
            p=[0.60, 0.25, 0.15],
        )

    if "complexity_level" not in df.columns:
        df["complexity_level"] = rng.choice(
            ["low", "medium", "high"],
            size=len(df),
            p=[0.45, 0.38, 0.17],
        )

    # Always recompute units from the current family/complexity/num_items values
    n   = df["num_items"].astype(float)
    fam = df["product_family"].astype(str)
    cpl = df["complexity_level"].astype(str)
    otype = df["order_type"].astype(str) if "order_type" in df.columns else pd.Series(["normal"] * len(df))

    fp  = fam.map(_PICK_FAM).fillna(1.0)
    pp  = fam.map(_PACK_FAM).fillna(1.0)
    dp  = fam.map(_DISP_FAM).fillna(1.0)
    cp  = cpl.map(_PICK_CPL).fillna(1.0)
    cp2 = cpl.map(_PACK_CPL).fillna(1.0)
    cp3 = cpl.map(_DISP_CPL).fillna(1.0)
    dup = otype.map(_DISP_URG).fillna(1.0)

    df["picking_units"]  = (n * fp * cp).round(2)
    df["packing_units"]  = ((1 + 0.25 * n) * pp * cp2).round(2)
    df["dispatch_units"] = (dup * dp * cp3).round(2)

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
