# src/reporting/export_rl3_monthly_recommendations.py
"""
Export webapp-ready monthly RL-3 capacity recommendations.

Reads  : data/rl3_monthly_capacity_cost_results.csv
Writes : data/app_exports/rl3_monthly_recommendations_summary.csv   (1 row/month)
         data/app_exports/rl3_monthly_capacity_cost_results_app.csv  (full results)

Usage:
    python -m src.reporting.export_rl3_monthly_recommendations
    python -m src.reporting.export_rl3_monthly_recommendations \\
        --input data/rl3_monthly_capacity_cost_results.csv \\
        --output-summary data/app_exports/rl3_monthly_recommendations_summary.csv \\
        --output-full data/app_exports/rl3_monthly_capacity_cost_results_app.csv
"""
from __future__ import annotations

import calendar
import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


POLICY_LABELS = {
    "fifo":         "FIFO",
    "urgent_first": "Urgent-First",
    "rl3_dqn":      "RL-3 DQN",
}

POLICY_ABBR = {
    "fifo":         "FIFO",
    "urgent_first": "UF",
    "rl3_dqn":      "RL3",
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
            "    python -m src.rl.evaluate_rl3_monthly_capacity_cost"
        )
        sys.exit(1)
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns: {sorted(missing)}")
        sys.exit(1)
    return df


def _safe(series, idx, key, default=np.nan):
    try:
        v = series.loc[idx, key]
        return v if pd.notna(v) else default
    except (KeyError, TypeError):
        return default


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for month_num in sorted(df["month"].unique()):
        mdf_all = df[df["month"] == month_num].copy()
        # Future-planning results tag screening-only (single-replication) rows separately
        # from 3-replication validated ones (src/analysis/future_screening.py, spec §7.4) —
        # the recommendation picks below must never be chosen from a screening-only row.
        # Historical results carry no such column and are unaffected.
        if "evaluation_stage" in mdf_all.columns:
            validated = mdf_all[mdf_all["evaluation_stage"] == "validated"]
            mdf = validated.copy() if not validated.empty else mdf_all
        else:
            mdf = mdf_all

        # ── Demand context ────────────────────────────────────────────────
        ref_row = mdf[mdf["policy"] == "fifo"]
        if ref_row.empty:
            ref_row = mdf
        ref = ref_row.iloc[0]
        total_orders  = int(ref["total_orders"])  if "total_orders"  in mdf.columns else np.nan
        urgent_orders = int(ref["urgent_orders"]) if "urgent_orders" in mdf.columns else np.nan
        normal_orders = int(ref["normal_orders"]) if "normal_orders" in mdf.columns else np.nan
        urgent_share  = float(ref["urgent_share"])if "urgent_share"  in mdf.columns else np.nan

        # ── Cheapest option ───────────────────────────────────────────────
        bt_idx     = mdf["estimated_total_cost"].idxmin()
        bt         = mdf.loc[bt_idx]
        bt_regime  = bt["regime"]
        bt_policy  = bt["policy"]
        bt_workers = int(bt["total_workers"])
        bt_pick    = int(bt["picking_workers"])  if "picking_workers"  in mdf.columns else np.nan
        bt_pack    = int(bt["packing_workers"])  if "packing_workers"  in mdf.columns else np.nan
        bt_disp    = int(bt["dispatch_workers"]) if "dispatch_workers" in mdf.columns else np.nan
        bt_sla     = float(bt["total_sla"])
        bt_u_sla   = float(bt["urgent_sla"])
        bt_n_sla   = float(bt["normal_sla"])
        bt_late    = float(bt["estimated_late_cost"])   if "estimated_late_cost"   in bt.index else np.nan
        bt_labour  = float(bt["estimated_worker_cost"]) if "estimated_worker_cost" in bt.index else np.nan
        bt_cost    = float(bt["estimated_total_cost"])

        # ── Best RL-3 ─────────────────────────────────────────────────────
        rl3_rows = mdf[mdf["policy"] == "rl3_dqn"]
        if not rl3_rows.empty:
            rl3_idx    = rl3_rows["estimated_total_cost"].idxmin()
            rl3        = rl3_rows.loc[rl3_idx]
            rl3_regime = rl3["regime"]
            rl3_workers= int(rl3["total_workers"])
            rl3_sla    = float(rl3["total_sla"])
            rl3_u_sla  = float(rl3["urgent_sla"])
            rl3_n_sla  = float(rl3["normal_sla"])
            rl3_late   = float(rl3["estimated_late_cost"])   if "estimated_late_cost"   in rl3.index else np.nan
            rl3_labour = float(rl3["estimated_worker_cost"]) if "estimated_worker_cost" in rl3.index else np.nan
            rl3_cost   = float(rl3["estimated_total_cost"])
            rl3_gap    = round(rl3_cost - bt_cost, 2)
        else:
            rl3_regime = rl3_workers = rl3_sla = rl3_u_sla = rl3_n_sla = np.nan
            rl3_late = rl3_labour = rl3_cost = rl3_gap = np.nan

        # ── Min workers with urgent_sla >= 0.95 ──────────────────────────
        u95 = mdf[mdf["urgent_sla"] >= 0.95]
        if not u95.empty:
            u95_idx    = u95["total_workers"].idxmin()
            u95r       = u95.loc[u95_idx]
            mu_regime  = u95r["regime"]
            mu_policy  = u95r["policy"]
            mu_workers = int(u95r["total_workers"])
            mu_sla     = float(u95r["total_sla"])
            mu_u_sla   = float(u95r["urgent_sla"])
            mu_n_sla   = float(u95r["normal_sla"])
            mu_late    = float(u95r["estimated_late_cost"])   if "estimated_late_cost"   in u95r.index else np.nan
            mu_labour  = float(u95r["estimated_worker_cost"]) if "estimated_worker_cost" in u95r.index else np.nan
            mu_cost    = float(u95r["estimated_total_cost"])
        else:
            mu_regime = mu_policy = mu_workers = mu_sla = mu_u_sla = mu_n_sla = np.nan
            mu_late = mu_labour = mu_cost = np.nan

        # ── Minimum FEASIBLE workforce: BOTH urgent AND normal SLA floors met ────────────
        # (spec §10/§24.1) — replaces "min total SLA >= 80%" as the promoted minimum-workforce
        # card; a config with total SLA 88% but urgent SLA 64% is never eligible here.
        feas = mdf[mdf["feasible"] == True] if "feasible" in mdf.columns else mdf.iloc[0:0]
        if not feas.empty:
            mf_idx     = feas["total_workers"].idxmin()
            mfr        = feas.loc[mf_idx]
            mf_regime  = mfr["regime"]
            mf_policy  = mfr["policy"]
            mf_workers = int(mfr["total_workers"])
            mf_sla     = float(mfr["total_sla"])
            mf_u_sla   = float(mfr["urgent_sla"])
            mf_n_sla   = float(mfr["normal_sla"])
            mf_late    = float(mfr["estimated_late_cost"])   if "estimated_late_cost"   in mfr.index else np.nan
            mf_labour  = float(mfr["estimated_worker_cost"]) if "estimated_worker_cost" in mfr.index else np.nan
            mf_cost    = float(mfr["estimated_total_cost"])
        else:
            mf_regime = mf_policy = mf_workers = mf_sla = mf_u_sla = mf_n_sla = np.nan
            mf_late = mf_labour = mf_cost = np.nan

        # ── Min workers with total_sla >= 0.80 (diagnostic only — NOT feasibility-gated) ──
        t80 = mdf[mdf["total_sla"] >= 0.80]
        if not t80.empty:
            t80_idx    = t80["total_workers"].idxmin()
            t80r       = t80.loc[t80_idx]
            mt_regime  = t80r["regime"]
            mt_policy  = t80r["policy"]
            mt_workers = int(t80r["total_workers"])
            mt_sla     = float(t80r["total_sla"])
            mt_u_sla   = float(t80r["urgent_sla"])
            mt_n_sla   = float(t80r["normal_sla"])
            mt_late    = float(t80r["estimated_late_cost"])   if "estimated_late_cost"   in t80r.index else np.nan
            mt_labour  = float(t80r["estimated_worker_cost"]) if "estimated_worker_cost" in t80r.index else np.nan
            mt_cost    = float(t80r["estimated_total_cost"])
        else:
            mt_regime = mt_policy = mt_workers = mt_sla = mt_u_sla = mt_n_sla = np.nan
            mt_late = mt_labour = mt_cost = np.nan

        # ── Best urgent_first ─────────────────────────────────────────────
        uf_rows = mdf[mdf["policy"] == "urgent_first"]
        if not uf_rows.empty:
            uf_idx    = uf_rows["estimated_total_cost"].idxmin()
            uf        = uf_rows.loc[uf_idx]
            uf_regime = uf["regime"]
            uf_workers= int(uf["total_workers"])
            uf_sla    = float(uf["total_sla"])
            uf_u_sla  = float(uf["urgent_sla"])
            uf_n_sla  = float(uf["normal_sla"])
            uf_late   = float(uf["estimated_late_cost"])   if "estimated_late_cost"   in uf.index else np.nan
            uf_labour = float(uf["estimated_worker_cost"]) if "estimated_worker_cost" in uf.index else np.nan
            uf_cost   = float(uf["estimated_total_cost"])
        else:
            uf_regime = uf_workers = uf_sla = uf_u_sla = uf_n_sla = np.nan
            uf_late = uf_labour = uf_cost = np.nan

        # ── Comparison ────────────────────────────────────────────────────
        rl3_minus_uf = (
            round(rl3_cost - uf_cost, 2)
            if not (np.isnan(rl3_cost) or np.isnan(uf_cost))
            else np.nan
        )

        # ── Interpretation ────────────────────────────────────────────────
        bt_label_policy = POLICY_LABELS.get(bt_policy, bt_policy)
        if bt_policy == "rl3_dqn":
            interp = f"RL-3 DQN minimises total cost this month"
        elif not np.isnan(rl3_cost) and not np.isnan(bt_cost) and abs(rl3_cost - bt_cost) < bt_cost * 0.02:
            interp = f"{bt_label_policy} chosen; RL-3 within 2% — consider RL-3 for SLA"
        else:
            interp = f"{bt_label_policy} is cost-optimal this month"

        # ── Labels ───────────────────────────────────────────────────────
        cheapest_label = f"Cheapest: {bt_regime} + {POLICY_LABELS.get(bt_policy, bt_policy)}"
        rl3_label = (
            f"Best RL-3: {rl3_regime} + rl3_dqn (+${rl3_gap:,.0f} vs cheapest)"
            if not (isinstance(rl3_regime, float) or np.isnan(rl3_gap))
            else "Best RL-3: n/a"
        )
        min_urgent_label = (
            f"Min urgent SLA≥95%: {mu_regime} + {POLICY_LABELS.get(mu_policy, mu_policy)} ({mu_workers} workers)"
            if not (isinstance(mu_regime, float))
            else "Min urgent SLA≥95%: none"
        )
        min_total_sla_label = (
            f"Min total SLA≥80%: {mt_regime} + {POLICY_LABELS.get(mt_policy, mt_policy)} ({mt_workers} workers)"
            if not (isinstance(mt_regime, float))
            else "Min total SLA≥80%: none"
        )
        min_feasible_label = (
            f"Minimum feasible workforce: {mf_regime} + {POLICY_LABELS.get(mf_policy, mf_policy)} ({mf_workers} workers)"
            if not (isinstance(mf_regime, float))
            else "Minimum feasible workforce: none reach both SLA targets"
        )

        rows.append({
            "month":        month_num,
            "month_name":   calendar.month_name[month_num],
            "total_orders":  total_orders,
            "urgent_orders": urgent_orders,
            "normal_orders": normal_orders,
            "urgent_share":  round(urgent_share, 4) if not np.isnan(urgent_share) else np.nan,
            # Cheapest
            "best_total_regime":            bt_regime,
            "best_total_policy":            bt_policy,
            "best_total_workers":           bt_workers,
            "best_total_picking_workers":   bt_pick,
            "best_total_packing_workers":   bt_pack,
            "best_total_dispatch_workers":  bt_disp,
            "best_total_sla":               round(bt_sla,   4),
            "best_total_urgent_sla":        round(bt_u_sla, 4),
            "best_total_normal_sla":        round(bt_n_sla, 4),
            "best_total_late_cost":         round(bt_late,   2) if not np.isnan(bt_late)   else np.nan,
            "best_total_labour_cost":       round(bt_labour, 2) if not np.isnan(bt_labour) else np.nan,
            "best_total_cost":              round(bt_cost,   2),
            # Best RL-3
            "best_rl3_regime":         rl3_regime if not isinstance(rl3_regime, float) else np.nan,
            "best_rl3_workers":        rl3_workers if not np.isnan(rl3_workers) else np.nan,
            "best_rl3_sla":            round(rl3_sla,    4) if not np.isnan(rl3_sla)    else np.nan,
            "best_rl3_urgent_sla":     round(rl3_u_sla,  4) if not np.isnan(rl3_u_sla)  else np.nan,
            "best_rl3_normal_sla":     round(rl3_n_sla,  4) if not np.isnan(rl3_n_sla)  else np.nan,
            "best_rl3_late_cost":      round(rl3_late,   2) if not np.isnan(rl3_late)   else np.nan,
            "best_rl3_labour_cost":    round(rl3_labour, 2) if not np.isnan(rl3_labour) else np.nan,
            "best_rl3_total_cost":     round(rl3_cost,   2) if not np.isnan(rl3_cost)   else np.nan,
            "best_rl3_gap_vs_cheapest":rl3_gap if not np.isnan(rl3_gap) else np.nan,
            # Min urgent SLA >= 0.95
            "min_urgent_regime":      mu_regime  if not isinstance(mu_regime, float) else np.nan,
            "min_urgent_policy":      mu_policy  if not isinstance(mu_policy, float) else np.nan,
            "min_urgent_workers":     mu_workers if not np.isnan(mu_workers) else np.nan,
            "min_urgent_sla":         round(mu_sla,    4) if not np.isnan(mu_sla)    else np.nan,
            "min_urgent_urgent_sla":  round(mu_u_sla,  4) if not np.isnan(mu_u_sla)  else np.nan,
            "min_urgent_normal_sla":  round(mu_n_sla,  4) if not np.isnan(mu_n_sla)  else np.nan,
            "min_urgent_late_cost":   round(mu_late,   2) if not np.isnan(mu_late)   else np.nan,
            "min_urgent_labour_cost": round(mu_labour, 2) if not np.isnan(mu_labour) else np.nan,
            "min_urgent_total_cost":  round(mu_cost,   2) if not np.isnan(mu_cost)   else np.nan,
            # Min total SLA >= 0.80
            "min_total_sla_regime":      mt_regime  if not isinstance(mt_regime, float) else np.nan,
            "min_total_sla_policy":      mt_policy  if not isinstance(mt_policy, float) else np.nan,
            "min_total_sla_workers":     mt_workers if not np.isnan(mt_workers) else np.nan,
            "min_total_sla":             round(mt_sla,    4) if not np.isnan(mt_sla)    else np.nan,
            "min_total_sla_urgent_sla":  round(mt_u_sla,  4) if not np.isnan(mt_u_sla)  else np.nan,
            "min_total_sla_normal_sla":  round(mt_n_sla,  4) if not np.isnan(mt_n_sla)  else np.nan,
            "min_total_sla_late_cost":   round(mt_late,   2) if not np.isnan(mt_late)   else np.nan,
            "min_total_sla_labour_cost": round(mt_labour, 2) if not np.isnan(mt_labour) else np.nan,
            "min_total_sla_total_cost":  round(mt_cost,   2) if not np.isnan(mt_cost)   else np.nan,
            # Best urgent_first
            "best_urgent_first_regime":       uf_regime  if not isinstance(uf_regime, float) else np.nan,
            "best_urgent_first_workers":      uf_workers if not np.isnan(uf_workers) else np.nan,
            "best_urgent_first_sla":          round(uf_sla,    4) if not np.isnan(uf_sla)    else np.nan,
            "best_urgent_first_urgent_sla":   round(uf_u_sla,  4) if not np.isnan(uf_u_sla)  else np.nan,
            "best_urgent_first_normal_sla":   round(uf_n_sla,  4) if not np.isnan(uf_n_sla)  else np.nan,
            "best_urgent_first_late_cost":    round(uf_late,   2) if not np.isnan(uf_late)   else np.nan,
            "best_urgent_first_labour_cost":  round(uf_labour, 2) if not np.isnan(uf_labour) else np.nan,
            "best_urgent_first_total_cost":   round(uf_cost,   2) if not np.isnan(uf_cost)   else np.nan,
            # Minimum FEASIBLE workforce (both urgent + normal SLA floors met, spec §10/§24.1)
            "min_feasible_regime":       mf_regime  if not isinstance(mf_regime, float) else np.nan,
            "min_feasible_policy":       mf_policy  if not isinstance(mf_policy, float) else np.nan,
            "min_feasible_workers":      mf_workers if not np.isnan(mf_workers) else np.nan,
            "min_feasible_sla":          round(mf_sla,    4) if not np.isnan(mf_sla)    else np.nan,
            "min_feasible_urgent_sla":   round(mf_u_sla,  4) if not np.isnan(mf_u_sla)  else np.nan,
            "min_feasible_normal_sla":   round(mf_n_sla,  4) if not np.isnan(mf_n_sla)  else np.nan,
            "min_feasible_late_cost":    round(mf_late,   2) if not np.isnan(mf_late)   else np.nan,
            "min_feasible_labour_cost":  round(mf_labour, 2) if not np.isnan(mf_labour) else np.nan,
            "min_feasible_total_cost":   round(mf_cost,   2) if not np.isnan(mf_cost)   else np.nan,
            # Comparison
            "rl3_minus_urgent_first_total_cost": rl3_minus_uf,
            # Labels
            "cheapest_label":       cheapest_label,
            "rl3_label":            rl3_label,
            "min_urgent_label":     min_urgent_label,
            "min_total_sla_label":  min_total_sla_label,
            "min_feasible_label":   min_feasible_label,
            "managerial_interpretation_short": interp,
        })

    return pd.DataFrame(rows)


def _print_summary_table(summary: pd.DataFrame) -> None:
    W = 110
    print("\n" + "=" * W)
    print("MONTHLY RECOMMENDATION SUMMARY")
    print("=" * W)
    print(
        f"{'Month':<12}"
        f"{'-- CHEAPEST ---------------------':<30}"
        f"{'-- BEST RL-3 ---------------':<27}"
        f"{'-- MIN URGENT SLA>=95% -----':<27}"
        f"{'-- MIN TOTAL SLA>=80% ------'}"
    )
    print(
        f"{'':12}"
        f"{'regime/policy':<16}{'cost':>10}{'SLA':>6}  "
        f"{'regime':<12}{'dcost':>10}{'SLA':>6}  "
        f"{'regime/policy':<16}{'SLA':>6}  "
        f"{'regime/policy':<16}{'SLA':>6}"
    )
    print("-" * W)

    for _, r in summary.iterrows():
        cp_abbr  = POLICY_ABBR.get(r["best_total_policy"], r["best_total_policy"][:4])
        cheap_rp = f"{r['best_total_regime']}/{cp_abbr}"
        cheap_cost = f"${r['best_total_cost']:>9,.0f}" if pd.notna(r.get("best_total_cost")) else "         —"
        cheap_sla  = f"{r['best_total_sla']:6.1%}"     if pd.notna(r.get("best_total_sla"))  else "     —"

        if pd.notna(r.get("best_rl3_total_cost")):
            gap = r["best_rl3_gap_vs_cheapest"]
            rl3_rg      = str(r["best_rl3_regime"])
            rl3_gap_str = f"{gap:>+10,.0f}" if pd.notna(gap) else "         —"
            rl3_sla_str = f"{r['best_rl3_sla']:6.1%}"
        else:
            rl3_rg, rl3_gap_str, rl3_sla_str = "—", "         —", "     —"

        if pd.notna(r.get("min_urgent_total_cost")):
            mu_abbr = POLICY_ABBR.get(r["min_urgent_policy"], str(r["min_urgent_policy"])[:4])
            mu_rp   = f"{r['min_urgent_regime']}/{mu_abbr}"
            mu_sla  = f"{r['min_urgent_sla']:6.1%}"
        else:
            mu_rp, mu_sla = "—", "     —"

        if pd.notna(r.get("min_total_sla_total_cost")):
            mt_abbr = POLICY_ABBR.get(r["min_total_sla_policy"], str(r["min_total_sla_policy"])[:4])
            mt_rp   = f"{r['min_total_sla_regime']}/{mt_abbr}"
            mt_sla  = f"{r['min_total_sla']:6.1%}"
        else:
            mt_rp, mt_sla = "—", "     —"

        print(
            f"{r['month_name']:<12}"
            f"{cheap_rp:<16}{cheap_cost}{cheap_sla}  "
            f"{rl3_rg:<12}{rl3_gap_str}{rl3_sla_str}  "
            f"{mu_rp:<16}{mu_sla}  "
            f"{mt_rp:<16}{mt_sla}"
        )
    print()


def _build_app_results(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "month", "month_name", "regime", "policy",
        "picking_workers", "packing_workers", "dispatch_workers", "total_workers",
        "total_orders", "urgent_orders", "normal_orders", "urgent_share",
        "total_sla", "urgent_sla", "normal_sla",
        "mean_system_time_min", "p90_system_time_min",
        "urgent_late_orders", "normal_late_orders",
        "completed_orders", "unfinished_orders", "unfinished_urgent_orders",
        "unfinished_normal_orders", "backlog_share",
        "estimated_late_cost", "estimated_worker_cost", "estimated_total_cost",
        "p_urgent_overall", "p_urgent_pick", "p_urgent_pack", "p_urgent_dispatch",
        "decisions_total", "decisions_pick", "decisions_pack", "decisions_dispatch",
        "cost_late_urgent", "cost_late_normal", "worker_cost_per_hour", "hours_per_worker_month",
        "feasible", "urgent_sla_target", "normal_sla_target", "sla_violation",
        "p90_total_cost", "prob_meets_sla_targets", "replication_count", "evaluation_stage",
    ]
    available = [c for c in keep if c in df.columns]
    return df[available].copy()


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Export webapp-ready monthly RL-3 capacity recommendations"
    )
    parser.add_argument("--input", default="data/rl3_monthly_capacity_cost_results.csv")
    parser.add_argument("--output-summary",
                        default="data/app_exports/rl3_monthly_recommendations_summary.csv")
    parser.add_argument("--output-full",
                        default="data/app_exports/rl3_monthly_capacity_cost_results_app.csv")
    args = parser.parse_args()

    input_path   = root / args.input
    summary_path = root / args.output_summary
    full_path    = root / args.output_full

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input          : {input_path}")
    print(f"Output summary : {summary_path}")
    print(f"Output full    : {full_path}\n")

    df = _load(input_path)

    print(f"Loaded: {len(df):,} rows  |  "
          f"{df['month'].nunique()} months  |  "
          f"{df['regime'].nunique()} regimes  |  "
          f"{df['policy'].nunique()} policies")

    if "month_name" not in df.columns:
        df["month_name"] = df["month"].apply(lambda m: calendar.month_name[m])

    summary = _build_summary(df)
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary : {summary_path}  ({len(summary)} rows, {len(summary.columns)} cols)")

    app_df = _build_app_results(df)
    app_df.to_csv(full_path, index=False)
    print(f"Saved full    : {full_path}  ({len(app_df)} rows, {len(app_df.columns)} cols)")

    # Display-only sections — wrapped so that console encoding issues never affect exit code.
    # The CSV files above are already fully written before this block runs.
    try:
        _print_summary_table(summary)

        bt_counts = summary["best_total_policy"].value_counts()
        print("\n  Policy wins (cheapest total cost):")
        for pol, cnt in bt_counts.items():
            months_list = summary.loc[summary["best_total_policy"] == pol, "month_name"].tolist()
            print(f"    {POLICY_LABELS.get(pol, pol):<16}: {cnt:>2} month(s) - {', '.join(months_list)}")

        rl3_saves = summary["rl3_minus_urgent_first_total_cost"].dropna()
        if not rl3_saves.empty:
            cheaper = int((rl3_saves < 0).sum())
            print(f"\n  RL-3 cheaper than urgent_first: {cheaper}/{len(rl3_saves)} months  "
                  f"(negative = RL-3 cheaper)")

        print("\n  Labels per month:")
        for _, r in summary.iterrows():
            print(f"  [{r['month_name']:<10}]  {r['cheapest_label']}")
            print(f"  {'':<12}  {r['rl3_label']}")
            print(f"  {'':<12}  {r['min_urgent_label']}")
            print(f"  {'':<12}  {r['min_total_sla_label']}")
        print()
    except Exception as display_err:
        print(f"\n[Note: summary display skipped ({type(display_err).__name__}: {display_err})]")


if __name__ == "__main__":
    main()
