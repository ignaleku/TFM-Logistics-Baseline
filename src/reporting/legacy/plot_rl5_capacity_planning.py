# src/reporting/plot_rl5_capacity_planning.py
"""
RL-5 capacity planning visualisations.

Generates 7 plots from existing CSV artefacts (no simulations).

Plots produced:
  1. Monthly SLA by policy (line chart)
  2. Monthly estimated total cost by policy (line chart)
  3. Best-total-cost regime per month (bar chart)
  4. RL-5 vs urgent_first cost savings per month (bar chart)
  5. Worker-cost sensitivity: RL-5 selection rate vs worker cost/hr (line)
  6. SLA heatmap: regime × month for RL-5 (heatmap)
  7. Decisions per stage per regime — RL-5 mean across windows (stacked bar)

Usage:
    python -m src.reporting.plot_rl5_capacity_planning
    python -m src.reporting.plot_rl5_capacity_planning --output-dir data/plots
"""
from __future__ import annotations

import calendar
import sys
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")   # headless rendering; override with --show flag
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

POLICY_COLORS = {
    "fifo":          "#5B9BD5",
    "urgent_first":  "#ED7D31",
    "rl5_dqn":       "#70AD47",
}
POLICY_LABELS = {
    "fifo":          "FIFO",
    "urgent_first":  "Urgent-First",
    "rl5_dqn":       "RL-5 DQN",
}

MONTH_ABBR = [calendar.month_abbr[m] for m in range(1, 13)]


# ── Load helpers ──────────────────────────────────────────────────────────────

def _load(path: Path, description: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [SKIP] {description}: {path.name} not found")
        return None
    df = pd.read_csv(path)
    print(f"  [OK]   {description}: {len(df):,} rows")
    return df


# ── Plot 1: Monthly SLA by policy ─────────────────────────────────────────────

def _plot_monthly_sla(df: pd.DataFrame, out: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for policy in ("fifo", "urgent_first", "rl5_dqn"):
        sub = df[df["policy"] == policy].sort_values("month")
        if sub.empty:
            continue
        ax.plot(
            sub["month"], sub["total_sla"],
            marker="o", linewidth=2,
            color=POLICY_COLORS.get(policy, "grey"),
            label=POLICY_LABELS.get(policy, policy),
        )

    ax.set_title("Monthly Total SLA Rate by Policy", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("SLA rate")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_ABBR)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    _save(fig, out, "plot1_monthly_sla.png", show)


# ── Plot 2: Monthly estimated total cost by policy ────────────────────────────

def _plot_monthly_cost(df: pd.DataFrame, out: Path, show: bool) -> None:
    if "estimated_total_cost" not in df.columns:
        print("  [SKIP] plot2: estimated_total_cost column missing")
        return

    # Lowest-cost regime for each (month, policy)
    best = df.loc[df.groupby(["month", "policy"])["estimated_total_cost"].idxmin()]

    fig, ax = plt.subplots(figsize=(10, 5))
    for policy in ("fifo", "urgent_first", "rl5_dqn"):
        sub = best[best["policy"] == policy].sort_values("month")
        if sub.empty:
            continue
        ax.plot(
            sub["month"], sub["estimated_total_cost"],
            marker="s", linewidth=2,
            color=POLICY_COLORS.get(policy, "grey"),
            label=POLICY_LABELS.get(policy, policy),
        )

    ax.set_title("Monthly Estimated Total Cost by Policy (best regime)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total cost (€)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_ABBR)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    _save(fig, out, "plot2_monthly_cost.png", show)


# ── Plot 3: Best-total-cost regime per month ──────────────────────────────────

def _plot_best_regime(summary: pd.DataFrame, out: Path, show: bool) -> None:
    if "best_total_regime" not in summary.columns:
        print("  [SKIP] plot3: best_total_regime column missing")
        return

    regimes = sorted(summary["best_total_regime"].unique())
    cmap    = plt.get_cmap("tab20", len(regimes))
    color_map = {r: cmap(i) for i, r in enumerate(regimes)}

    fig, ax = plt.subplots(figsize=(10, 4))
    for _, row in summary.iterrows():
        regime = row["best_total_regime"]
        ax.bar(
            int(row["month"]), 1,
            color=color_map[regime],
            label=regime,
        )

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), title="Regime", loc="upper right", fontsize=8)

    ax.set_title("Best-Cost Regime per Month", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_yticks([])
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_ABBR)
    fig.tight_layout()
    _save(fig, out, "plot3_best_regime_per_month.png", show)


# ── Plot 4: RL-5 vs urgent_first cost savings per month ──────────────────────

def _plot_rl5_savings(summary: pd.DataFrame, out: Path, show: bool) -> None:
    if "rl5_vs_best_cost_diff" not in summary.columns:
        print("  [SKIP] plot4: rl5_vs_best_cost_diff column missing")
        return

    months = summary["month"].tolist()
    savings = summary["rl5_vs_best_cost_diff"].tolist()
    colors  = ["#70AD47" if s <= 0 else "#FF6B6B" for s in savings]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(months, [-s for s in savings], color=colors)   # negate: positive = RL-5 cheaper
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("RL-5 Cost Savings vs Best-Total-Cost Strategy per Month",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Savings (€, positive = RL-5 cheaper)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_ABBR)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    _save(fig, out, "plot4_rl5_savings_vs_best.png", show)


# ── Plot 5: Worker-cost sensitivity — RL-5 selection rate ─────────────────────

def _plot_worker_cost_sensitivity(df_sens: pd.DataFrame, out: Path, show: bool) -> None:
    if "worker_cost_per_hour" not in df_sens.columns or "policy" not in df_sens.columns:
        print("  [SKIP] plot5: required columns missing")
        return

    wc_values = sorted(df_sens["worker_cost_per_hour"].unique())
    rl5_rates = []
    for wc in wc_values:
        sub    = df_sens[df_sens["worker_cost_per_hour"] == wc]
        months = sub["month"].nunique()
        rl5_m  = int((sub["policy"] == "rl5_dqn").sum())
        rl5_rates.append(rl5_m / months if months > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wc_values, rl5_rates, marker="o", color="#70AD47", linewidth=2)
    ax.set_title("RL-5 Selection Rate vs Worker Cost per Hour",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Worker cost (€/hr)")
    ax.set_ylabel("Fraction of months RL-5 is cheapest")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    _save(fig, out, "plot5_worker_cost_sensitivity.png", show)


# ── Plot 6: SLA heatmap — RL-5, regime × month ────────────────────────────────

def _plot_sla_heatmap(df_cap: pd.DataFrame, out: Path, show: bool) -> None:
    rl5 = df_cap[df_cap["policy"] == "rl5_dqn"].copy()
    if rl5.empty:
        print("  [SKIP] plot6: no rl5_dqn rows")
        return

    pivot = rl5.pivot_table(index="regime", columns="month", values="total_sla", aggfunc="mean")
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.9), max(5, pivot.shape[0] * 0.55)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.7, vmax=1.0)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([MONTH_ABBR[m - 1] for m in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7,
                        color="black" if 0.82 < val < 0.95 else "white")

    plt.colorbar(im, ax=ax, label="Total SLA rate")
    ax.set_title("RL-5 SLA Rate — Regime × Month", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Regime")
    fig.tight_layout()
    _save(fig, out, "plot6_sla_heatmap.png", show)


# ── Plot 7: Decisions per stage per regime (RL-5) ─────────────────────────────

def _plot_decisions_by_stage(df_ms: pd.DataFrame, out: Path, show: bool) -> None:
    stage_cols = {
        "Pick":     "decisions_pick",
        "QC":       "decisions_quality_check",
        "Pack":     "decisions_pack",
        "Label":    "decisions_labelling",
        "Dispatch": "decisions_dispatch",
    }
    rl5 = df_ms[df_ms["policy"] == "rl5_dqn"].copy()
    available = {k: v for k, v in stage_cols.items() if v in rl5.columns}
    if not available:
        print("  [SKIP] plot7: decision columns missing in multiseed results")
        return

    agg = rl5.groupby("regime")[[v for v in available.values()]].mean()
    agg.columns = list(available.keys())

    regimes = list(agg.index)
    x       = np.arange(len(regimes))
    width   = 0.6
    cmap    = plt.get_cmap("tab10", len(available))
    bottom  = np.zeros(len(regimes))

    fig, ax = plt.subplots(figsize=(max(10, len(regimes) * 0.85), 5))
    for i, stage in enumerate(available.keys()):
        vals = agg[stage].values
        ax.bar(x, vals, width, bottom=bottom, color=cmap(i), label=stage)
        bottom += vals

    ax.set_title("Mean Decisions per Stage per Regime (RL-5, across windows)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Mean decisions")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=30, ha="right")
    ax.legend(title="Stage", loc="upper right")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    _save(fig, out, "plot7_decisions_by_stage.png", show)


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_dir: Path, filename: str, show: bool) -> None:
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RL-5 capacity planning visualisations"
    )
    parser.add_argument("--output-dir", default="data/plots",
                        help="Directory to write PNG files (default: data/plots)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively as well as saving")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {out_dir}\n")

    # Load inputs
    df_monthly  = _load(DATA / "rl5_monthly_eval_results.csv",               "monthly eval")
    df_capacity = _load(DATA / "rl5_monthly_capacity_cost_results.csv",      "capacity-cost")
    df_summary  = _load(DATA / "app_exports" / "rl5_monthly_recommendations_summary.csv",
                        "recommendations summary")
    df_sens     = _load(DATA / "rl5_worker_cost_sensitivity_results.csv",    "worker-cost sensitivity")
    df_ms       = _load(DATA / "rl5_eval_multiseed_results.csv",             "multiseed eval")

    print()

    if df_monthly is not None:
        _plot_monthly_sla(df_monthly, out_dir, args.show)
    else:
        print("  [SKIP] plot1: monthly eval data not available")

    if df_capacity is not None:
        _plot_monthly_cost(df_capacity, out_dir, args.show)
    else:
        print("  [SKIP] plot2: capacity-cost data not available")

    if df_summary is not None:
        _plot_best_regime(df_summary, out_dir, args.show)
        _plot_rl5_savings(df_summary, out_dir, args.show)
    else:
        print("  [SKIP] plots 3 & 4: recommendations summary not available")

    if df_sens is not None:
        _plot_worker_cost_sensitivity(df_sens, out_dir, args.show)
    else:
        print("  [SKIP] plot5: worker-cost sensitivity data not available")

    if df_capacity is not None:
        _plot_sla_heatmap(df_capacity, out_dir, args.show)
    else:
        print("  [SKIP] plot6: capacity-cost data not available")

    if df_ms is not None:
        _plot_decisions_by_stage(df_ms, out_dir, args.show)
    else:
        print("  [SKIP] plot7: multiseed eval data not available")

    print("\nDone.")


if __name__ == "__main__":
    main()
