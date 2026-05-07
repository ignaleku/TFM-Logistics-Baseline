# src/rl/evaluate_rl5_worker_cost_sensitivity.py
"""
Worker-cost sensitivity analysis for the RL-5 monthly capacity planner.

Reads the pre-computed monthly capacity cost results and recalculates
estimated_total_cost for a range of worker hourly rates without re-running
any simulations.

Usage:
    python -m src.rl.evaluate_rl5_worker_cost_sensitivity
"""
from __future__ import annotations

import calendar
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Sensitivity sweep — edit to change the range
# ---------------------------------------------------------------------------
WORKER_COST_PER_HOUR_VALUES = [8, 10, 12, 15, 18, 20, 25, 30]
HOURS_PER_WORKER_PER_MONTH  = 160.0

# Threshold separating "lean" from "high-capacity" regimes
LOW_WORKER_MAX = 7   # total_workers <= 7 → low-worker; > 7 → high-worker

CSV_COLS = [
    "worker_cost_per_hour",
    "month",
    "month_name",
    "best_regime",
    "best_policy",
    "total_workers",
    "total_sla",
    "urgent_sla",
    "normal_sla",
    "estimated_late_cost",
    "estimated_worker_cost",
    "estimated_total_cost",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(
            f"[ERROR] Input file not found: {path}\n"
            "Please generate it first by running:\n"
            "    python -m src.rl.evaluate_rl5_monthly_capacity_cost"
        )
        sys.exit(1)
    df = pd.read_csv(path)
    required = {
        "month", "month_name", "regime", "policy",
        "total_workers", "total_sla", "urgent_sla", "normal_sla",
        "estimated_late_cost",
    }
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns in input: {sorted(missing)}")
        sys.exit(1)
    return df


def _best_per_month(df_base: pd.DataFrame, wc: float) -> pd.DataFrame:
    """Return one row per month: the (regime, policy) with lowest total cost."""
    df = df_base.copy()
    df["estimated_worker_cost"] = df["total_workers"] * wc * HOURS_PER_WORKER_PER_MONTH
    df["estimated_total_cost"]  = df["estimated_late_cost"] + df["estimated_worker_cost"]
    idx = df.groupby("month")["estimated_total_cost"].idxmin()
    return df.loc[idx].reset_index(drop=True)


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(results: pd.DataFrame) -> None:
    print("\n=== Summary ===\n")

    # 1. Regime selection count per worker cost
    print("  Regime selection count per worker_cost_per_hour (across all months):\n")
    regime_tbl = (
        results.groupby(["worker_cost_per_hour", "best_regime"])
        .size()
        .unstack(fill_value=0)
    )
    print(regime_tbl.to_string())
    print()

    # 2. Policy selection count per worker cost
    print("  Policy selection count per worker_cost_per_hour (across all months):\n")
    policy_tbl = (
        results.groupby(["worker_cost_per_hour", "best_policy"])
        .size()
        .unstack(fill_value=0)
    )
    print(policy_tbl.to_string())
    print()

    # 3. Thresholds: months where best (regime, policy) changes as cost rises
    print("  Thresholds — where the optimal configuration changes per month:\n")
    months    = sorted(results["month"].unique())
    wc_values = sorted(results["worker_cost_per_hour"].unique())
    any_change = False

    for m in months:
        sub = (
            results[results["month"] == m]
            .sort_values("worker_cost_per_hour")
            .reset_index(drop=True)
        )
        prev_reg = prev_pol = None
        changes = []
        for _, row in sub.iterrows():
            reg, pol = row["best_regime"], row["best_policy"]
            if prev_reg is not None and (reg != prev_reg or pol != prev_pol):
                changes.append(
                    f"${row['worker_cost_per_hour']}/hr: "
                    f"{prev_reg}/{prev_pol} → {reg}/{pol}"
                )
            prev_reg, prev_pol = reg, pol

        if changes:
            any_change = True
            print(f"  {calendar.month_name[m]}:")
            for ch in changes:
                print(f"    {ch}")

    if not any_change:
        print("  No changes — the same configuration is optimal at all tested costs.")
    print()

    # 4. Worker costs where RL-5 is selected in at least one month
    rl5_wcs = sorted(
        results[results["best_policy"] == "rl5_dqn"]["worker_cost_per_hour"].unique()
    )
    if rl5_wcs:
        print(f"  Worker costs ($/hr) where RL-5 is best in ≥1 month: {rl5_wcs}")
    else:
        print("  RL-5 is not the best policy at any tested worker cost level.")
    print()

    # 5. Worker costs where high-worker regimes (>LOW_WORKER_MAX) are NOT selected
    wcs_with_high = set(
        results[results["total_workers"] > LOW_WORKER_MAX]["worker_cost_per_hour"].unique()
    )
    absent = [wc for wc in WORKER_COST_PER_HOUR_VALUES if wc not in wcs_with_high]
    if absent:
        print(
            f"  Worker costs ($/hr) where no high-capacity regime (>{LOW_WORKER_MAX} workers) "
            f"is selected: {absent}"
        )
    else:
        print(
            f"  High-capacity regimes (>{LOW_WORKER_MAX} workers) are selected "
            "at every tested worker cost level."
        )
    print()


# ── Interpretation ────────────────────────────────────────────────────────────

def _print_interpretation(results: pd.DataFrame) -> None:
    print("=== Interpretation ===\n")

    wc_values = sorted(results["worker_cost_per_hour"].unique())
    wc_min, wc_max = wc_values[0], wc_values[-1]

    avg_workers_low  = results[results["worker_cost_per_hour"] == wc_min]["total_workers"].mean()
    avg_workers_high = results[results["worker_cost_per_hour"] == wc_max]["total_workers"].mean()

    rl5_wins    = results["best_policy"].eq("rl5_dqn").any()
    high_exists = results["total_workers"].gt(LOW_WORKER_MAX).any()

    direction = "falls" if avg_workers_high < avg_workers_low else "stays flat"

    print(
        f"  • Labour cost and optimal workforce size:\n"
        f"    At ${wc_min}/hr the average optimal worker count is {avg_workers_low:.1f}.\n"
        f"    At ${wc_max}/hr it {direction} to {avg_workers_high:.1f}.\n"
        f"    Lower labour costs make larger workforces more attractive: the SLA penalty\n"
        f"    savings from extra workers exceed the additional headcount cost."
    )
    print()
    print(
        "  • Higher labour costs shift the optimum toward leaner configurations.\n"
        "    When workers are expensive, a smaller team with a smarter dispatch policy\n"
        "    can achieve lower total cost even if raw SLA rates decline slightly."
    )
    print()

    if rl5_wins:
        print(
            "  • RL-5 is selected as the optimal policy in at least one scenario.\n"
            "    The learned dispatch policy reduces SLA penalty costs enough to reach\n"
            "    a lower total cost than FIFO or urgent_first at that configuration,\n"
            "    potentially allowing a leaner workforce to remain competitive."
        )
    else:
        print(
            "  • RL-5 is not the cheapest policy at any tested worker cost level.\n"
            "    urgent_first (or FIFO) achieves lower total cost across all scenarios.\n"
            "    Further RL-5 training (e.g. v2 with focused difficult regimes) may\n"
            "    be required before the policy can justify staffing decisions."
        )
    print()

    if high_exists:
        print(
            f"  • High-capacity regimes (>{LOW_WORKER_MAX} workers) are economically\n"
            "    justified under at least some labour cost assumptions, suggesting that\n"
            "    months with high demand or high urgent share benefit from extra headcount\n"
            "    regardless of the dispatch policy used."
        )
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    root        = Path(__file__).resolve().parents[2]
    input_path  = root / "data" / "rl5_monthly_capacity_cost_results.csv"
    output_path = root / "data" / "rl5_worker_cost_sensitivity_results.csv"

    df_base = _load(input_path)
    months  = sorted(df_base["month"].unique())

    print(f"Input  : {input_path}  ({len(df_base):,} rows)")
    print(f"Months : {[calendar.month_abbr[m] for m in months]}")
    print(f"Worker cost sweep ($/hr): {WORKER_COST_PER_HOUR_VALUES}")
    print(f"Hours/worker/month : {HOURS_PER_WORKER_PER_MONTH}\n")

    # Console table header
    hdr = (
        f"{'$/hr':>5}  {'Month':<10} {'Regime':<8} {'Policy':<14} "
        f"{'W':>2} {'SLA':>6} {'SLA-U':>6} {'Late $':>9} {'Total $':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    result_rows = []

    for wc in WORKER_COST_PER_HOUR_VALUES:
        best = _best_per_month(df_base, wc)

        for _, row in best.sort_values("month").iterrows():
            result_rows.append({
                "worker_cost_per_hour":  wc,
                "month":                 int(row["month"]),
                "month_name":            row["month_name"],
                "best_regime":           row["regime"],
                "best_policy":           row["policy"],
                "total_workers":         int(row["total_workers"]),
                "total_sla":             round(float(row["total_sla"]), 4),
                "urgent_sla":            round(float(row["urgent_sla"]), 4),
                "normal_sla":            round(float(row["normal_sla"]), 4),
                "estimated_late_cost":   round(float(row["estimated_late_cost"]), 2),
                "estimated_worker_cost": round(float(row["estimated_worker_cost"]), 2),
                "estimated_total_cost":  round(float(row["estimated_total_cost"]), 2),
            })
            print(
                f"{wc:>5}  {row['month_name']:<10} {row['regime']:<8} "
                f"{row['policy']:<14} {int(row['total_workers']):>2} "
                f"{float(row['total_sla']):>6.4f} {float(row['urgent_sla']):>6.4f} "
                f"{float(row['estimated_late_cost']):>9,.0f} "
                f"{float(row['estimated_total_cost']):>10,.0f}"
            )

    results_df = pd.DataFrame(result_rows)[CSV_COLS]
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}  ({len(results_df)} rows)")

    _print_summary(results_df)
    _print_interpretation(results_df)


if __name__ == "__main__":
    main()
