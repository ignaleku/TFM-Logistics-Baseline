# src/validation/quick_project_checks.py
"""
Sanity checks for generated data and RL evaluation artefacts.
Prints PASS / FAIL / SKIP for each file and check.
Does not modify any files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ── result tracking ────────────────────────────────────────────────────────────

_results: list[tuple[str, str, str]] = []   # (file, check, status)


def _record(file: str, check: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    _results.append((file, check, status if not detail else f"{status} — {detail}"))


# ── generic helpers ────────────────────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, required: list[str], file: str) -> bool:
    missing = [c for c in required if c not in df.columns]
    ok = len(missing) == 0
    _record(file, "required columns", ok, f"missing: {missing}" if not ok else "")
    return ok


def _check_no_nan(df: pd.DataFrame, cols: list[str], file: str) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        n = int(df[col].isna().sum())
        _record(file, f"no NaN in {col}", n == 0, f"{n} nulls" if n else "")


def _check_between(series: pd.Series, lo: float, hi: float, label: str, file: str) -> None:
    bad = int(((series < lo) | (series > hi)).sum())
    _record(file, label, bad == 0, f"{bad} out-of-range values" if bad else "")


def _check_non_negative(series: pd.Series, label: str, file: str) -> None:
    bad = int((series < 0).sum())
    _record(file, label, bad == 0, f"{bad} negative values" if bad else "")


# ── order file checks ──────────────────────────────────────────────────────────

ORDER_REQUIRED = [
    "order_id", "arrival_time", "order_type",
    "num_items", "product_class", "sla_minutes", "scenario",
]
ORDER_NO_NAN = ["order_id", "arrival_time", "order_type", "num_items", "product_class"]


def _check_order_file(path: Path) -> None:
    name = path.name
    df = pd.read_csv(path)

    if not _check_columns(df, ORDER_REQUIRED, name):
        _record(name, "(remaining checks skipped)", False, "columns missing")
        return

    # arrival_time parseable
    try:
        df["arrival_time"] = pd.to_datetime(df["arrival_time"])
        _record(name, "arrival_time parseable", True)
    except Exception as exc:
        _record(name, "arrival_time parseable", False, str(exc))
        return

    # order_id unique
    dupes = int(df["order_id"].duplicated().sum())
    _record(name, "order_id unique", dupes == 0, f"{dupes} duplicates" if dupes else "")

    # arrival_time sorted
    sorted_ok = df["arrival_time"].is_monotonic_increasing
    _record(name, "arrival_time sorted", sorted_ok)

    # num_items > 0
    bad_items = int((df["num_items"] <= 0).sum())
    _record(name, "num_items > 0", bad_items == 0, f"{bad_items} rows ≤ 0" if bad_items else "")

    # order_type values
    allowed_types = {"urgent", "normal"}
    bad_types = set(df["order_type"].unique()) - allowed_types
    _record(name, "order_type in {urgent, normal}", len(bad_types) == 0,
            f"unexpected: {bad_types}" if bad_types else "")

    # product_class values
    allowed_classes = {"A", "B", "C"}
    bad_classes = set(df["product_class"].unique()) - allowed_classes
    _record(name, "product_class in {A, B, C}", len(bad_classes) == 0,
            f"unexpected: {bad_classes}" if bad_classes else "")

    _check_no_nan(df, ORDER_NO_NAN, name)


# ── eval file checks ───────────────────────────────────────────────────────────

EVAL_REQUIRED = ["regime", "policy", "total_sla", "urgent_sla", "normal_sla",
                 "mean_system_time_min", "p90_system_time_min"]

MULTISEED_REQUIRED = ["regime", "policy", "window_id", "total_sla", "urgent_sla",
                      "normal_sla", "mean_system_time_min", "p90_system_time_min"]

RL5_MONTHLY_CAPACITY_REQUIRED = [
    "month", "regime", "policy",
    "total_sla", "urgent_sla", "normal_sla", "total_workers",
    "urgent_late_orders", "normal_late_orders",
]

APP_SUMMARY_REQUIRED = [
    "month", "month_name",
    "best_total_regime", "best_total_policy", "best_total_workers",
    "best_total_sla", "rl5_best_total_sla",
]

APP_RESULTS_REQUIRED = [
    "month", "regime", "policy",
    "total_sla", "urgent_sla", "normal_sla", "total_workers",
]


def _check_eval_file(path: Path, required: list[str],
                     sla_cols: list[str], time_cols: list[str]) -> None:
    name = path.name
    df = pd.read_csv(path)

    if not _check_columns(df, required, name):
        _record(name, "(remaining checks skipped)", False, "columns missing")
        return

    for col in sla_cols:
        if col in df.columns:
            _check_between(df[col].dropna(), 0.0, 1.0, f"{col} in [0, 1]", name)

    for col in time_cols:
        if col in df.columns:
            _check_non_negative(df[col].dropna(), f"{col} >= 0", name)


def _check_app_file(path: Path, required: list[str]) -> None:
    name = path.name
    df = pd.read_csv(path)
    _check_columns(df, required, name)
    row_ok = len(df) > 0
    _record(name, "non-empty", row_ok, f"{len(df)} rows" if not row_ok else f"{len(df)} rows")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    order_files = [
        ROOT / "data" / "orders_base.csv",
        ROOT / "data" / "orders_peak_campaign.csv",
        ROOT / "data" / "orders_stress.csv",
    ]
    eval_files = [
        (
            ROOT / "data" / "rl3_eval_results.csv",
            EVAL_REQUIRED,
            # rate cols checked in [0, 1]; optional decision-rate cols included here
            ["total_sla", "urgent_sla", "normal_sla",
             "p_urgent_overall", "p_urgent_pick", "p_urgent_pack", "p_urgent_dispatch"],
            # non-negative cols; optional decision counts included here
            ["mean_system_time_min", "p90_system_time_min",
             "decisions_total", "decisions_pick", "decisions_pack", "decisions_dispatch"],
        ),
        (
            ROOT / "data" / "rl3_eval_multiseed_results.csv",
            MULTISEED_REQUIRED,
            ["total_sla", "urgent_sla", "normal_sla"],
            ["mean_system_time_min", "p90_system_time_min"],
        ),
        # ── RL-5 eval files ───────────────────────────────────────────────────
        (
            ROOT / "data" / "rl5_eval_results.csv",
            EVAL_REQUIRED,
            ["total_sla", "urgent_sla", "normal_sla",
             "p_urgent_overall", "p_urgent_pick", "p_urgent_quality_check",
             "p_urgent_pack", "p_urgent_labelling", "p_urgent_dispatch"],
            ["mean_system_time_min", "p90_system_time_min",
             "decisions_total", "decisions_pick", "decisions_quality_check",
             "decisions_pack", "decisions_labelling", "decisions_dispatch"],
        ),
        (
            ROOT / "data" / "rl5_eval_multiseed_results.csv",
            MULTISEED_REQUIRED,
            ["total_sla", "urgent_sla", "normal_sla"],
            ["mean_system_time_min", "p90_system_time_min"],
        ),
        (
            ROOT / "data" / "rl5_monthly_capacity_cost_results.csv",
            RL5_MONTHLY_CAPACITY_REQUIRED,
            ["total_sla", "urgent_sla", "normal_sla"],
            ["mean_system_time_min", "p90_system_time_min",
             "urgent_late_orders", "normal_late_orders", "total_workers"],
        ),
    ]

    app_exports_files = [
        (ROOT / "data" / "app_exports" / "rl5_monthly_recommendations_summary.csv", APP_SUMMARY_REQUIRED),
        (ROOT / "data" / "app_exports" / "rl5_monthly_capacity_cost_results_app.csv", APP_RESULTS_REQUIRED),
    ]

    skipped = 0

    for path in order_files:
        if not path.exists():
            _results.append((path.name, "file present", "SKIP"))
            skipped += 1
        else:
            _check_order_file(path)

    for path, required, sla_cols, time_cols in eval_files:
        if not path.exists():
            _results.append((path.name, "file present", "SKIP"))
            skipped += 1
        else:
            _check_eval_file(path, required, sla_cols, time_cols)

    for path, required in app_exports_files:
        if not path.exists():
            _results.append((path.name, "file present", "SKIP"))
            skipped += 1
        else:
            _check_app_file(path, required)

    # ── print summary ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("  Sanity checks")
    print("=" * 70)

    current_file = None
    fails = 0
    passes = 0

    for file, check, status in _results:
        if file != current_file:
            current_file = file
            print(f"\n  {file}")
            print(f"  {'-' * (len(file))}")
        tag = status.split(" —")[0]
        if tag == "PASS":
            passes += 1
        elif tag == "FAIL":
            fails += 1
        detail = status[len(tag):]
        print(f"    [{tag:4}]  {check}{detail}")

    print("\n" + "=" * 70)
    total_checked = passes + fails
    if skipped:
        print(f"  {passes}/{total_checked} checks passed   {fails} failed   {skipped} file(s) skipped (not yet generated)")
    else:
        print(f"  {passes}/{total_checked} checks passed   {fails} failed")

    if fails == 0 and total_checked > 0:
        print("  Result: PASS")
    elif fails > 0:
        print("  Result: FAIL")
    else:
        print("  Result: SKIP (no files found — run data generation first)")
    print("=" * 70)


if __name__ == "__main__":
    main()
