# src/reporting/plot_bottleneck_sensitivity.py
"""
Reporting plots for RL-3 bottleneck sensitivity results.

Input : data/rl3_bottleneck_sensitivity_results.csv
Output: reports/figures/final/

Usage:
    python -m src.reporting.plot_bottleneck_sensitivity
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "figures" / "final"
DATA    = ROOT / "data" / "rl3_bottleneck_sensitivity_results.csv"

# ── ordering ─────────────────────────────────────────────────────────────────

SCENARIO_ORDER = ["base", "picking_slow", "packing_slow", "dispatch_slow", "all_balanced"]
REGIME_ORDER   = ["s111", "s211", "s221", "s311", "s321", "s222", "s332"]

# ── labels & colours ─────────────────────────────────────────────────────────

SENSITIVITY_LABEL = {
    "base":          "Base",
    "picking_slow":  "Picking slow\n(×1.8)",
    "packing_slow":  "Packing slow\n(×1.8)",
    "dispatch_slow": "Dispatch slow\n(×1.8)",
    "all_balanced":  "All balanced\n(×1.3)",
}
SENSITIVITY_COLOR = {
    "base":          "#4878d0",
    "picking_slow":  "#ee854a",
    "packing_slow":  "#c44e52",
    "dispatch_slow": "#8d7bae",
    "all_balanced":  "#6acc65",
}

POLICY_COLOR = {"fifo": "#4878d0", "urgent_first": "#ee854a", "rl3_dqn": "#6acc65"}
POLICY_LABEL = {"fifo": "FIFO", "urgent_first": "Urgent-first", "rl3_dqn": "DQN (RL-3)"}
POLICIES     = ["fifo", "urgent_first", "rl3_dqn"]

# ── I/O helpers ───────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")


def _skip(name: str, reason: str) -> None:
    print(f"  skip:  {name}  ({reason})")


def _ordered_scenarios(df: pd.DataFrame) -> list[str]:
    present = set(df["scenario"].unique())
    return [s for s in SCENARIO_ORDER if s in present]


def _ordered_regimes(df: pd.DataFrame) -> list[str]:
    present = set(df["regime"].unique())
    return [r for r in REGIME_ORDER if r in present]


# ── Plot 1 — RL-3 mean SLA by scenario ───────────────────────────────────────

def plot_mean_sla_by_scenario(df: pd.DataFrame) -> None:
    name = "bottleneck_sensitivity_mean_sla_by_scenario.png"
    rl = df[df["policy"] == "rl3_dqn"]
    scenarios = _ordered_scenarios(df)

    means = rl.groupby("scenario")["total_sla"].mean().reindex(scenarios)
    stds  = rl.groupby("scenario")["total_sla"].std().reindex(scenarios).fillna(0)

    x      = np.arange(len(scenarios))
    colors = [SENSITIVITY_COLOR.get(s, "#aaaaaa") for s in scenarios]
    labels = [SENSITIVITY_LABEL.get(s, s) for s in scenarios]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, means.values, color=colors, width=0.55,
                  yerr=stds.values, capsize=4,
                  error_kw={"elinewidth": 0.9, "ecolor": "#555555"})
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in means.values],
                 padding=3, fontsize=8)

    ax.set_title("DQN (RL-3) — Mean Total SLA by Bottleneck Scenario\n(mean across all regimes)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean SLA rate")
    ax.set_ylim(0, 1.12)
    ax.axhline(1.0, color="grey", linewidth=0.7, linestyle="--")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

    # reference line: base mean
    if "base" in means.index and not np.isnan(means["base"]):
        ax.axhline(means["base"], color=SENSITIVITY_COLOR["base"],
                   linewidth=1.0, linestyle=":", alpha=0.7, label="Base level")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, name)


# ── Plot 2 — SLA drop vs base ─────────────────────────────────────────────────

def plot_drop_vs_base(df: pd.DataFrame) -> None:
    name = "bottleneck_sensitivity_drop_vs_base.png"
    rl = df[df["policy"] == "rl3_dqn"]
    scenarios = _ordered_scenarios(df)
    non_base  = [s for s in scenarios if s != "base"]

    if "base" not in df["scenario"].values:
        _skip(name, "base scenario not found in data")
        return

    base_mean = float(rl[rl["scenario"] == "base"]["total_sla"].mean())
    drops     = {s: base_mean - float(rl[rl["scenario"] == s]["total_sla"].mean())
                 for s in non_base}

    x      = np.arange(len(non_base))
    values = [drops[s] for s in non_base]
    colors = [SENSITIVITY_COLOR.get(s, "#aaaaaa") for s in non_base]
    labels = [SENSITIVITY_LABEL.get(s, s) for s in non_base]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x, values, color=colors, width=0.50)
    ax.bar_label(bars, labels=[f"{v:+.4f}" for v in values],
                 padding=3, fontsize=8)

    ax.set_title(
        f"DQN (RL-3) — Mean SLA Drop vs Base (base = {base_mean:.4f})\n"
        "(positive = worse than base)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("SLA drop (base − scenario)")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")

    # add a subtle annotation showing which is hardest
    if values:
        worst_idx = int(np.argmax(values))
        ax.annotate(
            "hardest",
            xy=(worst_idx, values[worst_idx]),
            xytext=(worst_idx + 0.15, values[worst_idx] + 0.003),
            fontsize=7, color="#c44e52",
        )

    fig.tight_layout()
    _save(fig, name)


# ── Plot 3 — Policy comparison across scenarios ───────────────────────────────

def plot_policy_comparison(df: pd.DataFrame) -> None:
    name = "bottleneck_sensitivity_policy_comparison.png"
    scenarios = _ordered_scenarios(df)

    means = (
        df.groupby(["policy", "scenario"])["total_sla"]
        .mean()
        .reset_index()
    )

    n_scen  = len(scenarios)
    n_pol   = len(POLICIES)
    width   = 0.22
    x       = np.arange(n_scen)
    offsets = np.linspace(-(n_pol - 1) / 2, (n_pol - 1) / 2, n_pol) * width

    fig, ax = plt.subplots(figsize=(10, 4))
    for offset, policy in zip(offsets, POLICIES):
        sub    = means[means["policy"] == policy].set_index("scenario")
        vals   = [float(sub.loc[s, "total_sla"]) if s in sub.index else np.nan
                  for s in scenarios]
        bars = ax.bar(x + offset, vals, width,
                      label=POLICY_LABEL.get(policy, policy),
                      color=POLICY_COLOR.get(policy, "#aaaaaa"))
        ax.bar_label(bars, labels=[f"{v:.3f}" if not np.isnan(v) else "" for v in vals],
                     padding=2, fontsize=6.5, rotation=0)

    ax.set_title("Mean Total SLA by Policy and Bottleneck Scenario\n(mean across all regimes)")
    ax.set_xticks(x)
    ax.set_xticklabels([SENSITIVITY_LABEL.get(s, s) for s in scenarios], fontsize=9)
    ax.set_ylabel("Mean SLA rate")
    ax.set_ylim(0, 1.14)
    ax.axhline(1.0, color="grey", linewidth=0.7, linestyle="--")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, name)


# ── Plot 4 — Heatmap RL-3 scenario × regime ───────────────────────────────────

def plot_heatmap_rl3(df: pd.DataFrame) -> None:
    name = "bottleneck_sensitivity_heatmap_rl3.png"
    rl = df[df["policy"] == "rl3_dqn"]

    scenarios = _ordered_scenarios(df)
    regimes   = _ordered_regimes(df)

    pivot = (
        rl.groupby(["scenario", "regime"])["total_sla"]
        .mean()
        .unstack("regime")
        .reindex(index=scenarios, columns=regimes)
    )

    data = pivot.values.astype(float)
    vmin = max(0.0, float(np.nanmin(data)) - 0.02)
    vmax = 1.0

    row_labels = [SENSITIVITY_LABEL.get(s, s) for s in scenarios]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(data, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes, fontsize=9)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Capacity regime (pick–pack–disp workers)")
    ax.set_title("DQN (RL-3) Total SLA — Heatmap by Scenario × Regime")

    # text annotations
    midpoint = (vmin + vmax) / 2.0
    for i in range(len(scenarios)):
        for j in range(len(regimes)):
            val = data[i, j]
            if np.isnan(val):
                txt = "—"
                color = "#555555"
            else:
                txt   = f"{val:.3f}"
                color = "white" if val < midpoint else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Total SLA rate", fontsize=8)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

    fig.tight_layout()
    _save(fig, name)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA.exists():
        print(
            f"Input file not found: {DATA}\n"
            "Run the sensitivity evaluation first:\n"
            "  python -m src.rl.evaluate_rl3_sensitivity"
        )
        return

    df = pd.read_csv(DATA)
    print(f"Output : {OUT_DIR}")
    print(f"Rows   : {len(df)}  |  Scenarios: {sorted(df['scenario'].unique())}  "
          f"|  Regimes: {sorted(df['regime'].unique())}\n")

    plot_mean_sla_by_scenario(df)
    plot_drop_vs_base(df)
    plot_policy_comparison(df)
    plot_heatmap_rl3(df)

    print("\nDone.")


if __name__ == "__main__":
    main()