"""
SLA Cost Calculator for RL-3 evaluation results.

Translates SLA performance into estimated business penalty costs using
editable assumptions defined at the top of this file.
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Business assumptions — edit these values to recalculate
# ---------------------------------------------------------------------------
TOTAL_ORDERS = 10000
URGENT_SHARE = 0.12
COST_LATE_URGENT = 20.0
COST_LATE_NORMAL = 5.0

# Derived
URGENT_ORDERS = TOTAL_ORDERS * URGENT_SHARE
NORMAL_ORDERS = TOTAL_ORDERS * (1 - URGENT_SHARE)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_CSV = os.path.join("data", "rl3_eval_results.csv")
OUTPUT_CSV = os.path.join("data", "rl3_sla_cost_analysis.csv")

REQUIRED_COLUMNS = {"regime", "policy", "urgent_sla", "normal_sla", "total_sla"}


def load_and_validate(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(
            f"[ERROR] Input file not found: {path}\n"
            "Please generate it first by running:\n"
            "    python -m src.rl.evaluate_rl3"
        )
        sys.exit(1)

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"[ERROR] Missing required columns in {path}: {sorted(missing)}")
        sys.exit(1)

    return df


def compute_costs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["urgent_late_orders"] = URGENT_ORDERS * (1 - df["urgent_sla"])
    df["normal_late_orders"] = NORMAL_ORDERS * (1 - df["normal_sla"])
    df["estimated_late_cost"] = (
        df["urgent_late_orders"] * COST_LATE_URGENT
        + df["normal_late_orders"] * COST_LATE_NORMAL
    )

    # Build lookup: regime -> cost for baseline policies
    fifo_cost = (
        df[df["policy"] == "fifo"]
        .set_index("regime")["estimated_late_cost"]
        .rename("fifo_cost")
    )
    urgent_first_cost = (
        df[df["policy"] == "urgent_first"]
        .set_index("regime")["estimated_late_cost"]
        .rename("urgent_first_cost")
    )

    df = df.join(fifo_cost, on="regime")
    df = df.join(urgent_first_cost, on="regime")

    df["savings_vs_fifo"] = df["fifo_cost"] - df["estimated_late_cost"]
    df["savings_vs_urgent_first"] = df["urgent_first_cost"] - df["estimated_late_cost"]

    df["cost_late_urgent"] = COST_LATE_URGENT
    df["cost_late_normal"] = COST_LATE_NORMAL
    df["urgent_orders_assumed"] = URGENT_ORDERS
    df["normal_orders_assumed"] = NORMAL_ORDERS

    output_columns = [
        "regime",
        "policy",
        "total_sla",
        "urgent_sla",
        "normal_sla",
        "urgent_late_orders",
        "normal_late_orders",
        "estimated_late_cost",
        "savings_vs_fifo",
        "savings_vs_urgent_first",
        "cost_late_urgent",
        "cost_late_normal",
        "urgent_orders_assumed",
        "normal_orders_assumed",
    ]
    return df[output_columns]


def print_summary_table(df: pd.DataFrame) -> None:
    display_cols = [
        "regime",
        "policy",
        "urgent_sla",
        "normal_sla",
        "estimated_late_cost",
        "savings_vs_fifo",
        "savings_vs_urgent_first",
    ]
    fmt = {
        "urgent_sla": "{:.3f}".format,
        "normal_sla": "{:.3f}".format,
        "estimated_late_cost": "{:,.0f}".format,
        "savings_vs_fifo": "{:,.0f}".format,
        "savings_vs_urgent_first": "{:,.0f}".format,
    }
    print("\n=== SLA Cost Analysis ===")
    print(df[display_cols].to_string(index=False, formatters=fmt))


def print_interpretation(df: pd.DataFrame) -> None:
    print("\n--- Interpretation ---")

    # Best policy by cost per regime
    print("\nBest policy by estimated cost per regime:")
    best = df.loc[df.groupby("regime")["estimated_late_cost"].idxmin(), ["regime", "policy", "estimated_late_cost"]]
    for _, row in best.iterrows():
        print(f"  {row['regime']}: {row['policy']}  (${row['estimated_late_cost']:,.0f})")

    # Regimes where RL-3 has lowest cost
    rl3_best_regimes = best[best["policy"] == "rl3_dqn"]["regime"].tolist()
    if rl3_best_regimes:
        print(f"\nRL-3 has the lowest estimated cost in: {', '.join(rl3_best_regimes)}")
    else:
        print("\nRL-3 does not have the lowest estimated cost in any regime.")

    # Total estimated cost by policy across all regimes
    print("\nTotal estimated cost by policy (sum across all regimes):")
    totals = df.groupby("policy")["estimated_late_cost"].sum().sort_values()
    for policy, total in totals.items():
        print(f"  {policy}: ${total:,.0f}")


def main() -> None:
    df_raw = load_and_validate(INPUT_CSV)
    df_out = compute_costs(df_raw)

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Output saved to: {OUTPUT_CSV}")

    print_summary_table(df_out)
    print_interpretation(df_out)


if __name__ == "__main__":
    main()