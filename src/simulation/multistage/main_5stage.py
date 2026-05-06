# src/simulation/multistage/main_5stage.py
"""
Phase 1 — 5-stage baseline simulation runner.

Evaluates FIFO and urgent_first across seven capacity regimes using the
5-stage logistics process: Picking → Quality Check → Packing → Labelling → Dispatch.

Usage:
    python -m src.simulation.multistage.main_5stage
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.simulation.multistage.sim_5stage import run_simulation_5stage

EPISODE_ORDERS = 10_000
OUTPUT_FILE    = "data/sim_5stage_baseline_results.csv"

# (label, pick, qc, pack, lab, disp)
# Matrix designed to shift the bottleneck progressively across stages:
#   s11111 : full congestion
#   s21111 : picking partially relieved
#   s31111 : picking comfortable → QC / packing should gain weight
#   s32111 : QC relieved → packing should emerge
#   s32211 : packing relieved → labelling / dispatch should emerge
#   s32221 : labelling relieved → dispatch should emerge
#   s33322 : abundant — no single dominant bottleneck
REGIMES = [
    ("s11111", 1, 1, 1, 1, 1),
    ("s21111", 2, 1, 1, 1, 1),
    ("s31111", 3, 1, 1, 1, 1),
    ("s32111", 3, 2, 1, 1, 1),
    ("s32211", 3, 2, 2, 1, 1),
    ("s32221", 3, 2, 2, 2, 1),
    ("s33322", 3, 3, 3, 2, 2),
]

POLICIES = ("fifo", "urgent_first")

STAGE_WAIT_COLS = {
    "picking":       "mean_wait_picking_min",
    "quality_check": "mean_wait_quality_check_min",
    "packing":       "mean_wait_packing_min",
    "labelling":     "mean_wait_labelling_min",
    "dispatch":      "mean_wait_dispatch_min",
}


# ── Console helpers ───────────────────────────────────────────────────────────

def _print_interpretation(df: pd.DataFrame) -> None:
    print("\n" + "=" * 68)
    print("INTERPRETATION")
    print("=" * 68)

    # 1. Best policy per regime
    print("\n  1. Best policy per regime (total_sla):")
    print(f"  {'Regime':<10} {'Best policy':<16} {'SLA':>6}  {'vs other':>8}")
    print(f"  {'-'*10} {'-'*16} {'-'*6}  {'-'*8}")

    for regime_name, *_ in REGIMES:
        sub = df[df["regime"] == regime_name]
        if sub.empty:
            continue
        best = sub.loc[sub["total_sla"].idxmax()]
        other = sub.loc[sub["total_sla"].idxmin()]
        delta = best["total_sla"] - other["total_sla"]
        print(
            f"  {regime_name:<10} {best['policy']:<16} {best['total_sla']:6.4f}  "
            f"{delta:>+8.4f}"
        )

    # 2. Stage with highest mean wait per regime/policy
    print("\n  2. Stage with highest mean queue wait per regime × policy:")
    print(
        f"  {'Regime':<10} {'Policy':<14} {'Bottleneck stage':<18} "
        f"{'mean wait (min)':>16}  {'2nd stage':>14}  {'wait':>6}"
    )
    print(f"  {'-'*10} {'-'*14} {'-'*18} {'-'*16}  {'-'*14}  {'-'*6}")

    for _, row in df.iterrows():
        waits = {stage: row[col] for stage, col in STAGE_WAIT_COLS.items()}
        sorted_stages = sorted(waits, key=waits.get, reverse=True)
        top1, top2 = sorted_stages[0], sorted_stages[1]
        print(
            f"  {row['regime']:<10} {row['policy']:<14} {top1:<18} "
            f"{waits[top1]:>16.2f}  {top2:>14}  {waits[top2]:>6.2f}"
        )

    # 3. Per-regime: how much does adding workers at picking help?
    print("\n  3. SLA gain when doubling picking workers (s11111 → s21111):")
    for policy in POLICIES:
        r1 = df[(df["regime"] == "s11111") & (df["policy"] == policy)]
        r2 = df[(df["regime"] == "s21111") & (df["policy"] == policy)]
        if r1.empty or r2.empty:
            continue
        delta = r2["total_sla"].values[0] - r1["total_sla"].values[0]
        print(f"  {policy:<14}  {delta:>+.4f}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    root = Path(__file__).resolve().parents[3]

    with open(root / "configs" / "sim_5stage.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sim_cfg       = cfg["simulation"]
    base_resources = cfg["resources"]
    service_cfg   = cfg["service_time"]

    orders_all = (
        pd.read_csv(root / "data" / "orders_base.csv", parse_dates=["arrival_time"])
        .sort_values("arrival_time")
        .reset_index(drop=True)
    )
    orders = orders_all.iloc[:EPISODE_ORDERS].copy()

    print("5-Stage Simulation — Baseline (FIFO + urgent_first)")
    print(f"Process : Picking → Quality Check → Packing → Labelling → Dispatch")
    print(f"Orders  : {len(orders):,}  |  Regimes: {len(REGIMES)}  |  Policies: {len(POLICIES)}")
    print()

    hdr = (
        f"{'Regime':<8} {'Policy':<14} "
        f"{'SLA':>6} {'SLA-U':>6} {'SLA-N':>6} "
        f"{'mean(min)':>10} {'p90(min)':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    output_rows: list[dict] = []

    for regime_name, n_pick, n_qc, n_pack, n_lab, n_disp in REGIMES:
        resources_cfg = {
            **base_resources,
            "picking_workers":       n_pick,
            "quality_check_workers": n_qc,
            "packing_workers":       n_pack,
            "labelling_workers":     n_lab,
            "dispatch_workers":      n_disp,
        }

        best_sla    = -1.0
        best_policy = ""

        for policy in POLICIES:
            run_cfg           = dict(sim_cfg)
            run_cfg["policy"] = policy

            df_ep, summary = run_simulation_5stage(
                orders, run_cfg, resources_cfg, service_cfg
            )

            urgent_n = int((df_ep["order_type"] == "urgent").sum())
            normal_n = int((df_ep["order_type"] == "normal").sum())

            output_rows.append({
                "regime":                      regime_name,
                "policy":                      policy,
                "picking_workers":             n_pick,
                "quality_check_workers":       n_qc,
                "packing_workers":             n_pack,
                "labelling_workers":           n_lab,
                "dispatch_workers":            n_disp,
                "total_orders":                len(df_ep),
                "urgent_orders":               urgent_n,
                "normal_orders":               normal_n,
                "total_sla":                   summary["sla_rate"],
                "urgent_sla":                  summary["sla_urgent"],
                "normal_sla":                  summary["sla_normal"],
                "mean_system_time_min":        summary["mean_system_min"],
                "p90_system_time_min":         summary["p90_system_min"],
                "mean_wait_picking_min":       summary["mean_wait_picking_min"],
                "mean_wait_quality_check_min": summary["mean_wait_quality_check_min"],
                "mean_wait_packing_min":       summary["mean_wait_packing_min"],
                "mean_wait_labelling_min":     summary["mean_wait_labelling_min"],
                "mean_wait_dispatch_min":      summary["mean_wait_dispatch_min"],
                "p90_wait_picking_min":        summary["p90_wait_picking_min"],
                "p90_wait_quality_check_min":  summary["p90_wait_quality_check_min"],
                "p90_wait_packing_min":        summary["p90_wait_packing_min"],
                "p90_wait_labelling_min":      summary["p90_wait_labelling_min"],
                "p90_wait_dispatch_min":       summary["p90_wait_dispatch_min"],
            })

            print(
                f"{regime_name:<8} {policy:<14} "
                f"{summary['sla_rate']:6.4f} {summary['sla_urgent']:6.4f} "
                f"{summary['sla_normal']:6.4f} "
                f"{summary['mean_system_min']:10.1f} {summary['p90_system_min']:9.1f}"
            )

            if summary["sla_rate"] > best_sla:
                best_sla    = summary["sla_rate"]
                best_policy = policy

        print(f"         → best: {best_policy}  (SLA = {best_sla:.4f})")
        print()

    df_out   = pd.DataFrame(output_rows)
    out_path = root / OUTPUT_FILE
    df_out.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

    _print_interpretation(df_out)


if __name__ == "__main__":
    main()