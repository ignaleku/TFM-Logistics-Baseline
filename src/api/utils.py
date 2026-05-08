from __future__ import annotations
import calendar
from pathlib import Path
from typing import Any, Dict

import pandas as pd

REQUIRED_ORDER_COLS = {"order_id", "arrival_time", "order_type", "num_items", "product_class"}


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

    return {
        "valid": True,
        "total_rows": total_rows,
        "date_range": date_range,
        "detected_months": month_names,
        "urgent_share": urgent_share,
    }
