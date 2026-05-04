# src/rl/plot_rl_results.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POLICIES = ["fifo", "urgent_first", "dqn"]
COLORS = {"fifo": "#4878d0", "urgent_first": "#ee854a", "dqn": "#6acc65"}
LABELS = {"fifo": "FIFO", "urgent_first": "Urgent-first", "dqn": "DQN"}


def _bar_group(ax: plt.Axes, df: pd.DataFrame, value_col: str, regimes: list[str]) -> None:
    n_policies = len(POLICIES)
    x = np.arange(len(regimes))
    width = 0.25

    for i, policy in enumerate(POLICIES):
        vals = [
            df.loc[(df["regime"] == r) & (df["policy"] == policy), value_col].squeeze()
            for r in regimes
        ]
        vals = [float(v) if not isinstance(v, float) else v for v in vals]
        ax.bar(x + i * width, vals, width, label=LABELS[policy], color=COLORS[policy])

    ax.set_xticks(x + width)
    ax.set_xticklabels(regimes)
    ax.legend(fontsize=9)


def plot_sla_total(df: pd.DataFrame, out: Path) -> None:
    regimes = df["regime"].unique().tolist()
    fig, ax = plt.subplots(figsize=(7, 4))
    _bar_group(ax, df, "sla_rate", regimes)
    ax.set_title("Total SLA by Regime and Policy")
    ax.set_ylabel("SLA rate")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", linewidth=0.7, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_sla_by_type(df: pd.DataFrame, out: Path) -> None:
    regimes = df["regime"].unique().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, col, title in zip(axes, ["sla_urgent", "sla_normal"], ["Urgent SLA", "Normal SLA"]):
        _bar_group(ax, df, col, regimes)
        ax.set_title(title)
        ax.set_ylabel("SLA rate")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="grey", linewidth=0.7, linestyle="--")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_p90(df: pd.DataFrame, out: Path) -> None:
    regimes = df["regime"].unique().tolist()
    fig, ax = plt.subplots(figsize=(7, 4))
    _bar_group(ax, df, "p90_system_min", regimes)
    ax.set_title("P90 System Time by Regime and Policy")
    ax.set_ylabel("Minutes")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_dqn_urgent_rate(df: pd.DataFrame, out: Path) -> None:
    dqn = df[df["policy"] == "dqn"].copy()
    regimes = dqn["regime"].tolist()
    rates = dqn["pct_urgent_decisions"].tolist()

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(regimes))
    bars = ax.bar(x, rates, color=COLORS["dqn"], width=0.5)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", label="50% baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_title("DQN % Urgent Decisions at Decision Points")
    ax.set_ylabel("Fraction choosing urgent")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "data" / "rl_eval_results.csv"
    out_dir = root / "reports" / "figures" / "rl"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    plot_sla_total(df, out_dir / "sla_total_by_regime_policy.png")
    plot_sla_by_type(df, out_dir / "sla_by_order_type.png")
    plot_p90(df, out_dir / "p90_system_time.png")
    plot_dqn_urgent_rate(df, out_dir / "dqn_urgent_decision_rate.png")

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
