# src/analysis/calibrate_capacity_cost_assumptions.py
"""
Economic cost calibration for the RL-5 monthly capacity planner.

Finds cost-assumption combinations that produce diverse and informative
monthly recommendations without re-running any simulations.

WARNING: These are scenario assumptions for sensitivity analysis,
not real company cost figures.

Usage:
    python -m src.analysis.calibrate_capacity_cost_assumptions
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Sensitivity grid — edit to change the sweep
# ---------------------------------------------------------------------------
COST_LATE_URGENT_VALUES    = [10, 15, 20, 30, 40, 60]
COST_LATE_NORMAL_VALUES    = [3, 5, 8, 10, 15]
WORKER_COST_PER_HOUR_VALUES = [8, 12, 15, 20, 25, 30]
HOURS_PER_WORKER_PER_MONTH  = 160.0

ALL_REGIMES  = [
    "s11111", "s21111", "s31111",
    "s32111", "s32121", "s32112",
    "s32211", "s32212", "s32221",
    "s33211", "s42211", "s33322",
]
ALL_POLICIES = ["fifo", "urgent_first", "rl5_dqn"]

PEAK_MONTHS = {11: "nov", 12: "dec"}   # November, December

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
    required = {"month", "regime", "policy", "total_workers",
                "urgent_late_orders", "normal_late_orders"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns: {sorted(missing)}")
        sys.exit(1)
    return df


def _evaluate_combo(
    df: pd.DataFrame,
    cu: float,
    cn: float,
    wc: float,
) -> dict:
    """Recalculate costs and find best config per month for one assumption combo."""
    late   = df["urgent_late_orders"] * cu + df["normal_late_orders"] * cn
    labour = df["total_workers"] * wc * HOURS_PER_WORKER_PER_MONTH
    total  = late + labour

    df2 = df.assign(
        _late=late,
        _labour=labour,
        _total=total,
    )
    best = df2.loc[df2.groupby("month")["_total"].idxmin()].copy()

    # Per-regime and per-policy counts
    regime_counts = best["regime"].value_counts()
    policy_counts = best["policy"].value_counts()

    n_unique_reg = int(best["regime"].nunique())
    n_unique_pol = int(best["policy"].nunique())
    rl5_months   = int((best["policy"] == "rl5_dqn").sum())
    uf_months    = int((best["policy"] == "urgent_first").sum())
    fifo_months  = int((best["policy"] == "fifo").sum())
    avg_w        = float(best["total_workers"].mean())
    min_w        = int(best["total_workers"].min())
    max_w        = int(best["total_workers"].max())

    # Diversity score
    score = n_unique_reg + n_unique_pol
    if rl5_months   > 0:                    score += 1
    if uf_months    > 0:                    score += 1
    if max_w        > min_w:               score += 1
    if regime_counts.max() > 10:           score -= 2
    if policy_counts.max() > 10:           score -= 2

    row: dict = {
        "cost_late_urgent":             cu,
        "cost_late_normal":             cn,
        "worker_cost_per_hour":         wc,
        "diversity_score":              score,
        "number_unique_regimes_selected":  n_unique_reg,
        "number_unique_policies_selected": n_unique_pol,
        "rl5_selected_months":          rl5_months,
        "urgent_first_selected_months": uf_months,
        "fifo_selected_months":         fifo_months,
        "avg_selected_workers":         round(avg_w, 2),
        "min_selected_workers":         min_w,
        "max_selected_workers":         max_w,
    }

    # Per-regime counts
    for r in ALL_REGIMES:
        row[f"count_{r}"] = int(regime_counts.get(r, 0))

    # Per-policy counts
    for p in ALL_POLICIES:
        row[f"count_{p.replace('_', '')}"] = int(policy_counts.get(p, 0))

    # Peak-month regime/policy (November, December)
    best_idx = best.set_index("month")
    for m_num, suffix in PEAK_MONTHS.items():
        if m_num in best_idx.index:
            row[f"peak_month_regime_{suffix}"] = best_idx.loc[m_num, "regime"]
            row[f"peak_month_policy_{suffix}"] = best_idx.loc[m_num, "policy"]
        else:
            row[f"peak_month_regime_{suffix}"] = None
            row[f"peak_month_policy_{suffix}"] = None

    return row, best


# ── Console output ────────────────────────────────────────────────────────────

def _print_top10(top10: pd.DataFrame, all_best: dict) -> None:
    """Print the top-10 combos with their November / December recommendations."""
    print("\n" + "=" * 80)
    print("TOP 10 ECONOMIC ASSUMPTION COMBINATIONS BY DIVERSITY SCORE")
    print("=" * 80 + "\n")

    for rank, (_, r) in enumerate(top10.iterrows(), start=1):
        cu  = r["cost_late_urgent"]
        cn  = r["cost_late_normal"]
        wc  = r["worker_cost_per_hour"]
        key = (cu, cn, wc)

        print(f"  #{rank:>2}  urgent_penalty=${cu}  normal_penalty=${cn}  "
              f"worker_cost=${wc}/hr")
        print(f"        diversity_score         = {int(r['diversity_score'])}")
        print(f"        unique regimes selected = {int(r['number_unique_regimes_selected'])}")
        print(f"        unique policies selected= {int(r['number_unique_policies_selected'])}")
        print(f"        RL-5 selected months    = {int(r['rl5_selected_months'])}")
        print(f"        urgent_first months     = {int(r['urgent_first_selected_months'])}")
        print(f"        avg workers selected    = {r['avg_selected_workers']:.1f}")

        reg_cols = [c for c in r.index if c.startswith("count_s")]
        counts_str = "  ".join(
            f"{c.replace('count_', '')}:{int(r[c])}"
            for c in reg_cols if int(r[c]) > 0
        )
        print(f"        regime counts           = {counts_str}")

        nov_r = r.get("peak_month_regime_nov", "—")
        nov_p = r.get("peak_month_policy_nov", "—")
        dec_r = r.get("peak_month_regime_dec", "—")
        dec_p = r.get("peak_month_policy_dec", "—")
        print(f"        November recommendation = {nov_r} / {nov_p}")
        print(f"        December recommendation = {dec_r} / {dec_p}")
        print()


def _print_warning() -> None:
    print("─" * 80)
    print("WARNING: These are not real company costs.")
    print("They are scenario assumptions used for sensitivity analysis only.")
    print("Use them to understand how different economic parameters affect")
    print("recommendations — not as inputs to actual financial decisions.")
    print("─" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    root        = Path(__file__).resolve().parents[2]
    input_path  = root / "data" / "rl5_monthly_capacity_cost_results.csv"
    output_path = root / "data" / "capacity_cost_calibration_results.csv"
    top10_path  = root / "data" / "capacity_cost_calibration_top10.csv"

    df_base = _load(input_path)

    combos = list(itertools.product(
        COST_LATE_URGENT_VALUES,
        COST_LATE_NORMAL_VALUES,
        WORKER_COST_PER_HOUR_VALUES,
    ))
    total = len(combos)   # 6 × 5 × 6 = 180

    print(f"Input   : {input_path}  ({len(df_base):,} rows)")
    print(f"Combos  : {total}  "
          f"({len(COST_LATE_URGENT_VALUES)} urgent × "
          f"{len(COST_LATE_NORMAL_VALUES)} normal × "
          f"{len(WORKER_COST_PER_HOUR_VALUES)} worker costs)")
    print(f"Months  : {sorted(df_base['month'].unique())}")
    print(f"Regimes : {sorted(df_base['regime'].unique())}")
    print()

    rows  = []
    done  = 0

    for cu, cn, wc in combos:
        result_row, _ = _evaluate_combo(df_base, cu, cn, wc)
        rows.append(result_row)
        done += 1
        if done % 30 == 0 or done == total:
            print(f"  Processed {done}/{total} combinations...")

    results_df = pd.DataFrame(rows)

    # Sort for readability in CSV
    results_df = results_df.sort_values(
        ["diversity_score", "rl5_selected_months",
         "number_unique_regimes_selected", "avg_selected_workers"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    results_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}  ({len(results_df)} rows)")

    # Top 10
    top10 = results_df.head(10).copy()
    top10.to_csv(top10_path, index=False)
    print(f"Saved: {top10_path}  (top 10)")

    _print_top10(top10, {})
    _print_warning()


if __name__ == "__main__":
    main()
