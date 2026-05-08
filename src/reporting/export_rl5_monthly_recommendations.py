# src/reporting/export_rl5_monthly_recommendations.py
"""
Export webapp-ready monthly RL-5 capacity recommendations from simulation results.

Reads  : data/rl5_monthly_capacity_cost_results.csv
Writes : data/app_exports/rl5_monthly_recommendations_summary.csv
         data/app_exports/rl5_monthly_capacity_cost_results_app.csv

Usage:
    python -m src.reporting.export_rl5_monthly_recommendations
    python -m src.reporting.export_rl5_monthly_recommendations \\
        --input data/rl5_monthly_capacity_cost_results.csv \\
        --output-dir data/app_exports
"""
from __future__ import annotations

import calendar
import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

POLICY_LABELS = {
    "fifo":          "FIFO",
    "urgent_first":  "Urgent-First",
    "rl5_dqn":       "RL-5 DQN",
}

# Abbreviated labels for compact table display
POLICY_ABBR = {
    "fifo":          "FIFO",
    "urgent_first":  "UF",
    "rl5_dqn":       "RL5",
}

REQUIRED_COLS = {
    "month", "regime", "policy",
    "total_workers", "total_sla", "urgent_sla", "normal_sla",
    "urgent_late_orders", "normal_late_orders",
}


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(
            f"[ERROR] Input file not found: {path}\n"
            "Please generate it first by running:\n"
            "    python -m src.rl.evaluate_rl5_monthly_capacity_cost"
        )
        sys.exit(1)
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns in input: {sorted(missing)}")
        sys.exit(1)
    return df


def _best_row(sub: pd.DataFrame, by: str, asc: bool) -> pd.Series:
    """Return the row with min/max value of `by` within sub."""
    idx = sub[by].idxmin() if asc else sub[by].idxmax()
    return sub.loc[idx]


def _managerial_label(best_policy: str, rl5_vs_best_sla: float) -> str:
    """Short managerial interpretation for the best-total-cost strategy."""
    label = POLICY_LABELS.get(best_policy, best_policy)
    if best_policy == "rl5_dqn":
        return f"{label} minimises total cost this month"
    if abs(rl5_vs_best_sla) < 0.005:
        return f"{label} chosen; RL-5 SLA within 0.5pp — consider RL-5 if cost tie"
    if rl5_vs_best_sla < -0.02:
        return f"{label} chosen; RL-5 SLA significantly lower — review capacity"
    return f"{label} is cost-optimal; RL-5 is alternative if SLA is priority"


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build one compact row per month with key recommendation fields."""
    rows = []

    for month_num in sorted(df["month"].unique()):
        mdf = df[df["month"] == month_num].copy()

        # ── A. Demand context ────────────────────────────────────────────────
        fifo_row = mdf[mdf["policy"] == "fifo"].iloc[0] if not mdf[mdf["policy"] == "fifo"].empty else None
        total_orders  = int(fifo_row["total_orders"])   if fifo_row is not None and "total_orders"  in mdf.columns else np.nan
        urgent_orders = int(fifo_row["urgent_orders"])  if fifo_row is not None and "urgent_orders" in mdf.columns else np.nan
        urgent_share  = float(fifo_row["urgent_share"]) if fifo_row is not None and "urgent_share"  in mdf.columns else np.nan

        # ── B. Best total-cost configuration (regime + policy) ───────────────
        best_total = _best_row(mdf, "estimated_total_cost", asc=True)
        bt_regime  = best_total["regime"]
        bt_policy  = best_total["policy"]
        bt_workers = int(best_total["total_workers"])
        bt_cost    = float(best_total["estimated_total_cost"]) if "estimated_total_cost" in best_total.index else np.nan
        bt_sla     = float(best_total["total_sla"])

        # ── C. Best SLA-only (regardless of cost) ───────────────────────────
        best_sla     = _best_row(mdf, "total_sla", asc=False)
        bsla_regime  = best_sla["regime"]
        bsla_policy  = best_sla["policy"]
        bsla_sla     = float(best_sla["total_sla"])
        bsla_workers = int(best_sla["total_workers"])

        # ── D. Min workers (fewest headcount across all configs) ─────────────
        min_workers_row = _best_row(mdf, "total_workers", asc=True)
        mw_regime  = min_workers_row["regime"]
        mw_policy  = min_workers_row["policy"]
        mw_workers = int(min_workers_row["total_workers"])
        mw_sla     = float(min_workers_row["total_sla"])

        # ── E. Best RL-5 row (rl5_dqn, lowest cost among RL rows) ───────────
        rl5_rows = mdf[mdf["policy"] == "rl5_dqn"]
        if not rl5_rows.empty:
            best_rl5        = _best_row(rl5_rows, "estimated_total_cost", asc=True)
            rl5_regime      = best_rl5["regime"]
            rl5_workers     = int(best_rl5["total_workers"])
            rl5_cost        = float(best_rl5["estimated_total_cost"]) if "estimated_total_cost" in best_rl5.index else np.nan
            rl5_sla         = float(best_rl5["total_sla"])
            rl5_urgent_sla  = float(best_rl5["urgent_sla"])
            rl5_normal_sla  = float(best_rl5["normal_sla"])
            rl5_late_cost   = float(best_rl5["estimated_late_cost"])   if "estimated_late_cost"  in best_rl5.index else np.nan
            rl5_labour_cost = float(best_rl5["estimated_worker_cost"]) if "estimated_worker_cost" in best_rl5.index else np.nan
        else:
            rl5_regime = rl5_workers = rl5_cost = rl5_sla = rl5_urgent_sla = rl5_normal_sla = np.nan
            rl5_late_cost = rl5_labour_cost = np.nan

        # ── F. Best urgent_first row ─────────────────────────────────────────
        uf_rows = mdf[mdf["policy"] == "urgent_first"]
        if not uf_rows.empty:
            best_uf    = _best_row(uf_rows, "estimated_total_cost", asc=True)
            uf_regime  = best_uf["regime"]
            uf_workers = int(best_uf["total_workers"])
            uf_cost    = float(best_uf["estimated_total_cost"]) if "estimated_total_cost" in best_uf.index else np.nan
            uf_sla     = float(best_uf["total_sla"])
        else:
            uf_regime = uf_workers = uf_cost = uf_sla = np.nan

        # ── G. Comparison: RL-5 vs best-total ───────────────────────────────
        rl5_vs_best_sla     = (rl5_sla  - bt_sla)  if not np.isnan(rl5_sla)  else np.nan
        rl5_vs_best_cost    = (rl5_cost - bt_cost)  if not np.isnan(rl5_cost) else np.nan
        rl5_gap_vs_cheapest = rl5_vs_best_cost

        label = _managerial_label(bt_policy, rl5_vs_best_sla if not np.isnan(rl5_vs_best_sla) else 0.0)

        # ── H. Best balanced service option ─────────────────────────────────
        # balanced_score = estimated_total_cost + 50_000 * |urgent_sla - normal_sla|
        # Penalises policies that protect urgent orders while leaving normal orders behind.
        if "estimated_total_cost" in mdf.columns:
            mdf_bal = mdf.copy()
            mdf_bal["_balanced_score"] = (
                mdf_bal["estimated_total_cost"]
                + 50_000 * (mdf_bal["urgent_sla"] - mdf_bal["normal_sla"]).abs()
            )
            bal_row         = mdf_bal.loc[mdf_bal["_balanced_score"].idxmin()]
            bal_regime      = bal_row["regime"]
            bal_policy      = bal_row["policy"]
            bal_workers     = int(bal_row["total_workers"])
            bal_sla         = float(bal_row["total_sla"])
            bal_urgent_sla  = float(bal_row["urgent_sla"])
            bal_normal_sla  = float(bal_row["normal_sla"])
            bal_late_cost   = float(bal_row["estimated_late_cost"])   if "estimated_late_cost"  in bal_row.index else np.nan
            bal_labour_cost = float(bal_row["estimated_worker_cost"]) if "estimated_worker_cost" in bal_row.index else np.nan
            bal_total_cost  = float(bal_row["estimated_total_cost"])
            bal_score       = float(bal_row["_balanced_score"])
        else:
            bal_regime = bal_policy = np.nan
            bal_workers = bal_sla = bal_urgent_sla = bal_normal_sla = np.nan
            bal_late_cost = bal_labour_cost = bal_total_cost = bal_score = np.nan

        # ── I. Best service under +10% budget ────────────────────────────────
        # Baseline: cheapest option's total cost. Allow up to +10%, then pick highest SLA.
        if not np.isnan(bt_cost) and "estimated_total_cost" in mdf.columns:
            max_budget    = bt_cost * 1.10
            within_budget = mdf[mdf["estimated_total_cost"] <= max_budget]
            if not within_budget.empty:
                ub_row         = _best_row(within_budget, "total_sla", asc=False)
                ub_regime      = ub_row["regime"]
                ub_policy      = ub_row["policy"]
                ub_workers     = int(ub_row["total_workers"])
                ub_sla         = float(ub_row["total_sla"])
                ub_urgent_sla  = float(ub_row["urgent_sla"])
                ub_normal_sla  = float(ub_row["normal_sla"])
                ub_total_cost  = float(ub_row["estimated_total_cost"])
            else:
                ub_regime = ub_policy = np.nan
                ub_workers = ub_sla = ub_urgent_sla = ub_normal_sla = ub_total_cost = np.nan
        else:
            ub_regime = ub_policy = np.nan
            ub_workers = ub_sla = ub_urgent_sla = ub_normal_sla = ub_total_cost = np.nan

        # ── J. Managerial labels ─────────────────────────────────────────────
        cheapest_label = (
            f"Cheapest: {bt_regime} + {POLICY_LABELS.get(bt_policy, bt_policy)}"
        )

        if not isinstance(rl5_regime, float):
            gap_str   = f"+${rl5_gap_vs_cheapest:,.0f} vs cheapest" if not np.isnan(rl5_gap_vs_cheapest) else "vs cheapest n/a"
            rl5_label = f"Best RL-5: {rl5_regime} + rl5_dqn ({gap_str})"
        else:
            rl5_label = "Best RL-5: n/a"

        if not isinstance(bal_regime, float):
            balanced_label = (
                f"Balanced service: {bal_regime} + {POLICY_LABELS.get(bal_policy, bal_policy)}"
            )
        else:
            balanced_label = "Balanced service: n/a"

        if not isinstance(ub_regime, float):
            under_budget_label = (
                f"Best service within +10% budget: {ub_regime} + {POLICY_LABELS.get(ub_policy, ub_policy)}"
            )
        else:
            under_budget_label = "Best service within +10% budget: n/a"

        rows.append({
            # identity
            "month":        month_num,
            "month_name":   calendar.month_name[month_num],
            # demand
            "total_orders":  total_orders,
            "urgent_orders": urgent_orders,
            "urgent_share":  round(urgent_share, 4) if not np.isnan(urgent_share) else np.nan,
            # B – best total cost
            "best_total_regime":         bt_regime,
            "best_total_policy":         bt_policy,
            "best_total_strategy_label": POLICY_LABELS.get(bt_policy, bt_policy),
            "best_total_workers":        bt_workers,
            "best_total_cost":           round(bt_cost, 2) if not np.isnan(bt_cost) else np.nan,
            "best_total_sla":            round(bt_sla, 4),
            # C – best SLA
            "best_sla_regime":   bsla_regime,
            "best_sla_policy":   bsla_policy,
            "best_sla_value":    round(bsla_sla, 4),
            "best_sla_workers":  bsla_workers,
            # D – min workers
            "min_workers_regime":  mw_regime,
            "min_workers_policy":  mw_policy,
            "min_workers_count":   mw_workers,
            "min_workers_sla":     round(mw_sla, 4),
            # E – best RL-5 (legacy naming, kept for backward compatibility)
            "rl5_best_regime":      rl5_regime,
            "rl5_best_workers":     rl5_workers if not np.isnan(rl5_workers) else np.nan,
            "rl5_best_cost":        round(rl5_cost, 2) if not np.isnan(rl5_cost) else np.nan,
            "rl5_best_total_sla":   round(rl5_sla, 4)         if not np.isnan(rl5_sla)         else np.nan,
            "rl5_best_urgent_sla":  round(rl5_urgent_sla, 4)  if not np.isnan(rl5_urgent_sla)  else np.nan,
            "rl5_best_normal_sla":  round(rl5_normal_sla, 4)  if not np.isnan(rl5_normal_sla)  else np.nan,
            # F – best urgent_first
            "uf_best_regime":   uf_regime,
            "uf_best_workers":  uf_workers if not np.isnan(uf_workers) else np.nan,
            "uf_best_cost":     round(uf_cost, 2) if not np.isnan(uf_cost) else np.nan,
            "uf_best_sla":      round(uf_sla, 4) if not np.isnan(uf_sla) else np.nan,
            # G – comparison
            "rl5_vs_best_sla_diff":  round(rl5_vs_best_sla, 4)  if not np.isnan(rl5_vs_best_sla)  else np.nan,
            "rl5_vs_best_cost_diff": round(rl5_vs_best_cost, 2) if not np.isnan(rl5_vs_best_cost) else np.nan,
            # Interpretation (legacy)
            "managerial_interpretation_short": label,
            # H – best RL-5 (decision-mode naming)
            "best_rl5_regime":        rl5_regime,
            "best_rl5_workers":       rl5_workers     if not np.isnan(rl5_workers)     else np.nan,
            "best_rl5_sla":           round(rl5_sla, 4)         if not np.isnan(rl5_sla)         else np.nan,
            "best_rl5_urgent_sla":    round(rl5_urgent_sla, 4)  if not np.isnan(rl5_urgent_sla)  else np.nan,
            "best_rl5_normal_sla":    round(rl5_normal_sla, 4)  if not np.isnan(rl5_normal_sla)  else np.nan,
            "best_rl5_late_cost":     round(rl5_late_cost, 2)   if not np.isnan(rl5_late_cost)   else np.nan,
            "best_rl5_labour_cost":   round(rl5_labour_cost, 2) if not np.isnan(rl5_labour_cost) else np.nan,
            "best_rl5_total_cost":    round(rl5_cost, 2)        if not np.isnan(rl5_cost)        else np.nan,
            "best_rl5_gap_vs_cheapest": round(rl5_gap_vs_cheapest, 2) if not np.isnan(rl5_gap_vs_cheapest) else np.nan,
            # I – balanced service
            "balanced_regime":      bal_regime,
            "balanced_policy":      bal_policy,
            "balanced_workers":     bal_workers,
            "balanced_sla":         round(bal_sla, 4)         if not np.isnan(bal_sla)         else np.nan,
            "balanced_urgent_sla":  round(bal_urgent_sla, 4)  if not np.isnan(bal_urgent_sla)  else np.nan,
            "balanced_normal_sla":  round(bal_normal_sla, 4)  if not np.isnan(bal_normal_sla)  else np.nan,
            "balanced_late_cost":   round(bal_late_cost, 2)   if not np.isnan(bal_late_cost)   else np.nan,
            "balanced_labour_cost": round(bal_labour_cost, 2) if not np.isnan(bal_labour_cost) else np.nan,
            "balanced_total_cost":  round(bal_total_cost, 2)  if not np.isnan(bal_total_cost)  else np.nan,
            "balanced_score":       round(bal_score, 2)       if not np.isnan(bal_score)       else np.nan,
            # J – best service under +10% budget
            "best_under_budget_regime":     ub_regime,
            "best_under_budget_policy":     ub_policy,
            "best_under_budget_workers":    ub_workers,
            "best_under_budget_sla":        round(ub_sla, 4)        if not np.isnan(ub_sla)        else np.nan,
            "best_under_budget_urgent_sla": round(ub_urgent_sla, 4) if not np.isnan(ub_urgent_sla) else np.nan,
            "best_under_budget_normal_sla": round(ub_normal_sla, 4) if not np.isnan(ub_normal_sla) else np.nan,
            "best_under_budget_total_cost": round(ub_total_cost, 2) if not np.isnan(ub_total_cost) else np.nan,
            # K – managerial labels
            "cheapest_label":      cheapest_label,
            "rl5_label":           rl5_label,
            "balanced_label":      balanced_label,
            "under_budget_label":  under_budget_label,
        })

    return pd.DataFrame(rows)


def _print_summary_table(summary: pd.DataFrame) -> None:
    """Print decision-mode summary: Month | Cheapest | Best RL-5 | Balanced | Under Budget."""
    W = 118
    print("\n" + "=" * W)
    print("MONTHLY RECOMMENDATION SUMMARY — DECISION MODES")
    print("=" * W)
    print(
        f"{'':12}"
        f"{'── CHEAPEST ─────────────────':<30}"
        f"{'── BEST RL-5 ──────────────':<27}"
        f"{'── BALANCED ───────────────':<27}"
        f"{'── UNDER BUDGET (+10%) ────'}"
    )
    print(
        f"{'Month':<12}"
        f"{'regime/policy':<16}{'cost':>10}{'SLA':>6}  "
        f"{'regime':<12}{'Δcost':>10}{'SLA':>6}  "
        f"{'regime/policy':<16}{'SLA':>6}  "
        f"{'regime/policy':<16}{'SLA':>6}"
    )
    print("-" * W)

    for _, r in summary.iterrows():
        # Cheapest
        cp_abbr    = POLICY_ABBR.get(r["best_total_policy"], r["best_total_policy"][:4])
        cheap_rp   = f"{r['best_total_regime']}/{cp_abbr}"
        cheap_cost = f"${r['best_total_cost']:>9,.0f}" if pd.notna(r.get("best_total_cost")) else "         —"
        cheap_sla  = f"{r['best_total_sla']:6.1%}"     if pd.notna(r.get("best_total_sla"))  else "     —"

        # Best RL-5
        if pd.notna(r.get("best_rl5_total_cost")):
            gap = r["best_rl5_gap_vs_cheapest"]
            rl5_rg      = str(r["best_rl5_regime"])
            rl5_gap_str = f"{gap:>+10,.0f}" if pd.notna(gap) else "         —"
            rl5_sla_str = f"{r['best_rl5_sla']:6.1%}"
        else:
            rl5_rg      = "—"
            rl5_gap_str = "         —"
            rl5_sla_str = "     —"

        # Balanced
        if pd.notna(r.get("balanced_total_cost")):
            bal_abbr = POLICY_ABBR.get(r["balanced_policy"], str(r["balanced_policy"])[:4])
            bal_rp   = f"{r['balanced_regime']}/{bal_abbr}"
            bal_sla  = f"{r['balanced_sla']:6.1%}"
        else:
            bal_rp  = "—"
            bal_sla = "     —"

        # Under budget
        if pd.notna(r.get("best_under_budget_total_cost")):
            ub_abbr = POLICY_ABBR.get(r["best_under_budget_policy"], str(r["best_under_budget_policy"])[:4])
            ub_rp   = f"{r['best_under_budget_regime']}/{ub_abbr}"
            ub_sla  = f"{r['best_under_budget_sla']:6.1%}"
        else:
            ub_rp  = "—"
            ub_sla = "     —"

        print(
            f"{r['month_name']:<12}"
            f"{cheap_rp:<16}{cheap_cost}{cheap_sla}  "
            f"{rl5_rg:<12}{rl5_gap_str}{rl5_sla_str}  "
            f"{bal_rp:<16}{bal_sla}  "
            f"{ub_rp:<16}{ub_sla}"
        )
    print()


def _build_app_results(df: pd.DataFrame) -> pd.DataFrame:
    """Clean full results for webapp consumption."""
    keep = [
        "month", "month_name", "regime", "policy",
        "total_workers", "total_orders", "urgent_orders", "normal_orders", "urgent_share",
        "total_sla", "urgent_sla", "normal_sla",
        "mean_system_time_min", "p90_system_time_min",
        "urgent_late_orders", "normal_late_orders",
        "estimated_late_cost", "estimated_worker_cost", "estimated_total_cost",
        "savings_vs_fifo", "savings_vs_urgent_first",
        "cost_late_urgent", "cost_late_normal", "worker_cost_per_hour", "hours_per_worker_month",
    ]
    available = [c for c in keep if c in df.columns]
    return df[available].copy()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Export webapp-ready monthly RL-5 capacity recommendations"
    )
    parser.add_argument(
        "--input",
        default="data/rl5_monthly_capacity_cost_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/app_exports",
    )
    args = parser.parse_args()

    input_path = root / args.input
    out_dir    = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "rl5_monthly_recommendations_summary.csv"
    app_path     = out_dir / "rl5_monthly_capacity_cost_results_app.csv"

    print(f"Input   : {input_path}")
    print(f"Out dir : {out_dir}\n")

    df = _load(input_path)

    print(f"Loaded  : {len(df):,} rows  |  "
          f"{df['month'].nunique()} months  |  "
          f"{df['regime'].nunique()} regimes  |  "
          f"{df['policy'].nunique()} policies")

    if "month_name" not in df.columns:
        df["month_name"] = df["month"].apply(lambda m: calendar.month_name[m])

    # ── Summary export ────────────────────────────────────────────────────────
    summary = _build_summary(df)
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}  ({len(summary)} rows, {len(summary.columns)} cols)")

    _print_summary_table(summary)

    # ── App results export ────────────────────────────────────────────────────
    app_df = _build_app_results(df)
    app_df.to_csv(app_path, index=False)
    print(f"Saved: {app_path}  ({len(app_df)} rows, {len(app_df.columns)} cols)")

    # ── Quick stats ───────────────────────────────────────────────────────────
    bt_counts = summary["best_total_policy"].value_counts()
    print("\n  Policy wins (best total cost):")
    for pol, cnt in bt_counts.items():
        label = POLICY_LABELS.get(pol, pol)
        months = summary.loc[summary["best_total_policy"] == pol, "month_name"].tolist()
        print(f"    {label:<14}: {cnt:>2} month(s) — {', '.join(months)}")

    bsla_counts = summary["best_sla_policy"].value_counts()
    print("\n  Policy wins (best SLA only):")
    for pol, cnt in bsla_counts.items():
        label = POLICY_LABELS.get(pol, pol)
        print(f"    {label:<14}: {cnt:>2} month(s)")

    bal_counts = summary["balanced_policy"].value_counts()
    print("\n  Policy wins (balanced service):")
    for pol, cnt in bal_counts.items():
        label = POLICY_LABELS.get(pol, str(pol))
        print(f"    {label:<14}: {cnt:>2} month(s)")

    ub_counts = summary["best_under_budget_policy"].value_counts()
    print("\n  Policy wins (best under budget):")
    for pol, cnt in ub_counts.items():
        label = POLICY_LABELS.get(pol, str(pol))
        print(f"    {label:<14}: {cnt:>2} month(s)")

    rl5_saves = summary["rl5_vs_best_cost_diff"].dropna()
    if not rl5_saves.empty:
        print(f"\n  RL-5 cost vs best-total: "
              f"mean={rl5_saves.mean():+,.0f}  "
              f"min={rl5_saves.min():+,.0f}  "
              f"max={rl5_saves.max():+,.0f}")

    print("\n  Decision-mode labels:")
    for _, r in summary.iterrows():
        print(f"  [{r['month_name']:<10}]  {r['cheapest_label']}")
        print(f"  {'':<12}  {r['rl5_label']}")
        print(f"  {'':<12}  {r['balanced_label']}")
        print(f"  {'':<12}  {r['under_budget_label']}")
    print()


if __name__ == "__main__":
    main()
