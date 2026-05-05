# src/reporting/plot_final_results.py
"""
Final result plots for thesis reporting.
Skips any plot whose required input file does not exist.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "figures" / "final"

# ── shared style ────────────────────────────────────────────────────────────────

POLICIES = ["fifo", "urgent_first", "rl3_dqn"]
POLICY_COLOR = {"fifo": "#4878d0", "urgent_first": "#ee854a", "rl3_dqn": "#6acc65"}
POLICY_LABEL = {"fifo": "FIFO", "urgent_first": "Urgent-first", "rl3_dqn": "DQN (RL-3)"}

SCENARIO_COLOR = {
    "base": "#4878d0",
    "peak_campaign": "#ee854a",
    "stress": "#c44e52",
}
SCENARIO_LABEL = {
    "base": "Base",
    "peak_campaign": "Peak campaign",
    "stress": "Stress",
}

WORKER_COLOR = {"1-1-1": "#c44e52", "2-1-1": "#4878d0", "2-2-1": "#6acc65"}


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")


def _skip(name: str, reason: str) -> None:
    print(f"  skip:  {name}  ({reason})")


# ── grouped bar helper ──────────────────────────────────────────────────────────

def _bar_group(
    ax: plt.Axes,
    summary: pd.DataFrame,
    value_col: str,
    regimes: list[str],
    err_col: str | None = None,
) -> None:
    x = np.arange(len(regimes))
    width = 0.25
    for i, policy in enumerate(POLICIES):
        sub = summary[summary["policy"] == policy]
        vals, errs = [], []
        for r in regimes:
            row = sub[sub["regime"] == r]
            vals.append(float(row[value_col].iloc[0]) if len(row) else 0.0)
            if err_col and err_col in sub.columns and len(row):
                errs.append(float(row[err_col].iloc[0]))
            else:
                errs.append(0.0)
        kw = {"yerr": errs, "capsize": 3, "error_kw": {"elinewidth": 0.8}} if any(e > 0 for e in errs) else {}
        ax.bar(x + i * width, vals, width,
               label=POLICY_LABEL[policy], color=POLICY_COLOR[policy], **kw)
    ax.set_xticks(x + width)
    ax.set_xticklabels(regimes)
    ax.legend(fontsize=9)


# ── data loaders ────────────────────────────────────────────────────────────────

def _load_orders() -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("base", "peak_campaign", "stress"):
        p = ROOT / "data" / f"orders_{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p, parse_dates=["arrival_time"])
    return out


def _load_eval_summary() -> tuple[pd.DataFrame | None, str]:
    """
    Returns (summary_df, source_label).
    Prefers multiseed (adds ± std error bars); falls back to single-window.
    Normalises column names to: regime, policy, sla_rate, sla_urgent,
    sla_normal, p90_system_min, plus optional *_std columns.
    """
    multi_path = ROOT / "data" / "rl3_eval_multiseed_results.csv"
    single_path = ROOT / "data" / "rl3_eval_results.csv"

    if multi_path.exists():
        df = pd.read_csv(multi_path)
        grp = df.groupby(["regime", "policy"])
        summary = grp.agg(
            sla_rate=("total_sla", "mean"),
            sla_urgent=("urgent_sla", "mean"),
            sla_normal=("normal_sla", "mean"),
            p90_system_min=("p90_system_time_min", "mean"),
            sla_rate_std=("total_sla", "std"),
            sla_urgent_std=("urgent_sla", "std"),
            sla_normal_std=("normal_sla", "std"),
            p90_std=("p90_system_time_min", "std"),
        ).reset_index().fillna(0)
        return summary, "multiseed (mean ± std)"

    if single_path.exists():
        df = pd.read_csv(single_path)
        for col in ("sla_rate_std", "sla_urgent_std", "sla_normal_std", "p90_std"):
            df[col] = 0.0
        return df, "single-window"

    return None, "none"


def _load_history() -> pd.DataFrame | None:
    p = ROOT / "data" / "rl3_train_history.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # rl3_train_history uses p_urgent_overall; normalise to p_urgent_decisions for plots
    if "p_urgent_overall" in df.columns and "p_urgent_decisions" not in df.columns:
        df["p_urgent_decisions"] = df["p_urgent_overall"]
    return df


# ── order plots ─────────────────────────────────────────────────────────────────

def plot_monthly_volume(orders: dict[str, pd.DataFrame]) -> None:
    name = "monthly_order_volume.png"
    if not orders:
        _skip(name, "no order files found")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    for scenario, df in orders.items():
        monthly = df.groupby(df["arrival_time"].dt.month).size().reindex(range(1, 13), fill_value=0)
        ax.plot(monthly.index, monthly.values, marker="o",
                label=SCENARIO_LABEL.get(scenario, scenario),
                color=SCENARIO_COLOR.get(scenario, "#999999"))

    ax.set_title("Monthly Order Volume by Scenario")
    ax.set_xlabel("Month")
    ax.set_ylabel("Orders")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, name)


def plot_hourly_profile(orders: dict[str, pd.DataFrame]) -> None:
    name = "hourly_order_profile.png"
    if not orders:
        _skip(name, "no order files found")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    for scenario, df in orders.items():
        hourly = df.groupby(df["arrival_time"].dt.hour).size().reindex(range(24), fill_value=0)
        ax.plot(hourly.index, hourly.values, marker="o", markersize=4,
                label=SCENARIO_LABEL.get(scenario, scenario),
                color=SCENARIO_COLOR.get(scenario, "#999999"))

    ax.set_title("Hourly Order Profile (all days combined)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Orders")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, name)


def plot_order_type_distribution(orders: dict[str, pd.DataFrame]) -> None:
    name = "order_type_distribution.png"
    if not orders:
        _skip(name, "no order files found")
        return

    scenarios = list(orders.keys())
    x = np.arange(len(scenarios))
    width = 0.35
    urgent_pcts = [float(orders[s]["order_type"].eq("urgent").mean()) for s in scenarios]
    normal_pcts = [float(orders[s]["order_type"].eq("normal").mean()) for s in scenarios]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars_u = ax.bar(x - width / 2, urgent_pcts, width, label="Urgent", color="#ee854a")
    bars_n = ax.bar(x + width / 2, normal_pcts, width, label="Normal", color="#4878d0")
    ax.bar_label(bars_u, labels=[f"{v:.1%}" for v in urgent_pcts], padding=2, fontsize=8)
    ax.bar_label(bars_n, labels=[f"{v:.1%}" for v in normal_pcts], padding=2, fontsize=8)
    ax.set_title("Order Type Distribution by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABEL.get(s, s) for s in scenarios])
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, name)


# ── RL eval plots ───────────────────────────────────────────────────────────────

def plot_total_sla(summary: pd.DataFrame | None) -> None:
    name = "rl_total_sla_comparison.png"
    if summary is None:
        _skip(name, "no eval data")
        return

    regimes = sorted(summary["regime"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    _bar_group(ax, summary, "sla_rate", regimes, err_col="sla_rate_std")
    ax.set_title("Total SLA by Regime and Policy")
    ax.set_ylabel("SLA rate")
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="grey", linewidth=0.7, linestyle="--")
    fig.tight_layout()
    _save(fig, name)


def plot_urgent_normal_sla(summary: pd.DataFrame | None) -> None:
    name = "rl_urgent_normal_sla_comparison.png"
    if summary is None:
        _skip(name, "no eval data")
        return

    regimes = sorted(summary["regime"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, col, std_col, title in zip(
        axes,
        ["sla_urgent", "sla_normal"],
        ["sla_urgent_std", "sla_normal_std"],
        ["Urgent SLA", "Normal SLA"],
    ):
        _bar_group(ax, summary, col, regimes, err_col=std_col)
        ax.set_title(title)
        ax.set_ylabel("SLA rate")
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, color="grey", linewidth=0.7, linestyle="--")
    fig.tight_layout()
    _save(fig, name)


def plot_p90(summary: pd.DataFrame | None) -> None:
    name = "rl_p90_comparison.png"
    if summary is None:
        _skip(name, "no eval data")
        return

    regimes = sorted(summary["regime"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    _bar_group(ax, summary, "p90_system_min", regimes, err_col="p90_std")
    ax.set_title("P90 System Time by Regime and Policy")
    ax.set_ylabel("Minutes")
    fig.tight_layout()
    _save(fig, name)


# ── training history plots ──────────────────────────────────────────────────────

def plot_training_sla_ma5(hist: pd.DataFrame | None) -> None:
    name = "dqn_training_sla_ma5.png"
    if hist is None:
        _skip(name, "no training history")
        return

    fig, ax = plt.subplots(figsize=(9, 4))

    # per-scenario scatter + MA-5 line
    for workers, grp in hist.groupby("workers"):
        color = WORKER_COLOR.get(str(workers), "#aaaaaa")
        ax.scatter(grp["episode"], grp["sla_rate"],
                   color=color, alpha=0.30, s=16, zorder=2)
        ma = grp.set_index("episode")["sla_rate"].rolling(5, min_periods=1).mean()
        ax.plot(ma.index, ma.values, color=color, linewidth=1.2,
                linestyle="--", alpha=0.8, label=f"MA-5  {workers}")

    # overall MA-5
    overall_ma = hist.set_index("episode")["sla_rate"].rolling(5, min_periods=1).mean()
    ax.plot(overall_ma.index, overall_ma.values,
            color="#222222", linewidth=2.0, label="MA-5  all", zorder=3)

    ax.set_title("DQN Training — SLA Rate per Episode (MA-5)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("SLA rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, name)


def plot_training_urgent_rate(hist: pd.DataFrame | None) -> None:
    name = "dqn_urgent_decision_rate_training.png"
    if hist is None:
        _skip(name, "no training history")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # left — urgent decision rate over episodes, colored by scenario
    ax = axes[0]
    for workers, grp in hist.groupby("workers"):
        color = WORKER_COLOR.get(str(workers), "#aaaaaa")
        ax.scatter(grp["episode"], grp["p_urgent_decisions"],
                   color=color, alpha=0.45, s=18, zorder=2)
        ma = grp.set_index("episode")["p_urgent_decisions"].rolling(5, min_periods=1).mean()
        ax.plot(ma.index, ma.values, color=color, linewidth=1.4,
                label=workers, zorder=3)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", label="50%")
    ax.set_title("Urgent Decision Rate by Scenario (MA-5)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction choosing urgent")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # right — epsilon decay
    ax2 = axes[1]
    ax2.plot(hist["episode"], hist["epsilon"], color="#333333", linewidth=1.6)
    ax2.set_title("Epsilon Decay")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    _save(fig, name)


# ── entry point ─────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT_DIR}\n")

    orders = _load_orders()
    summary, source = _load_eval_summary()
    hist = _load_history()

    print(f"Order files : {list(orders.keys()) or 'none'}")
    print(f"Eval source : {source}")
    print(f"Train hist  : {'yes' if hist is not None else 'no'}\n")

    plot_monthly_volume(orders)
    plot_hourly_profile(orders)
    plot_order_type_distribution(orders)
    plot_total_sla(summary)
    plot_urgent_normal_sla(summary)
    plot_p90(summary)
    plot_training_sla_ma5(hist)
    plot_training_urgent_rate(hist)

    print("\nDone.")


if __name__ == "__main__":
    main()
