"""
THESIS-ONLY shared helpers for figure/table generation.

Observational only: reads persisted outputs under data/, writes figures under thesis/figures/
and CSV tables under thesis/tables/. Never imports or modifies production simulation code.

All persisted outputs are read AS RUN. In particular the historical and future runs use
DIFFERENT economic parameters (see thesis_support/THESIS_STATE.md 6.7), so no function here
ever combines monetary values across the two modes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "api_runs" / "latest"
FIGDIR = ROOT / "thesis" / "figures"
TABDIR = ROOT / "thesis" / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

POLICIES = ["fifo", "urgent_first", "rl3_dqn"]
POLICY_LABEL = {"fifo": "FIFO", "urgent_first": "Urgent-First", "rl3_dqn": "RL-3 DQN"}
# Colour-blind-safe, print-safe, consistent across every figure in the thesis.
POLICY_COLOUR = {"fifo": "#B3452F", "urgent_first": "#2F6DA3", "rl3_dqn": "#2E7D5B"}
POLICY_MARKER = {"fifo": "o", "urgent_first": "s", "rl3_dqn": "^"}

STAGES = ["picking", "packing", "dispatch"]
STAGE_LABEL = {"picking": "Picking", "packing": "Packing", "dispatch": "Dispatch"}
STAGE_COLOUR = {"picking": "#3E5C76", "packing": "#7A9E9F", "dispatch": "#C9A227"}

# SLA feasibility floors (configs/planning_profile.yaml::sla)
URGENT_TARGET = 0.95
NORMAL_TARGET = 0.80


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_results(mode: str) -> pd.DataFrame:
    """Full candidate x policy grid for 'historical' (576 rows) or 'future' (48 rows)."""
    return pd.read_csv(RUNS / mode / "rl3_monthly_capacity_cost_results.csv")


def load_recommendations() -> pd.DataFrame:
    """One row per month (historical mode only)."""
    return pd.read_csv(RUNS / "historical" / "rl3_monthly_recommendations_summary.csv")


def load_bottleneck(mode: str) -> dict:
    with open(RUNS / mode / "bottleneck_analysis.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(mode: str) -> dict:
    with open(RUNS / mode / "run_manifest.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_orders() -> pd.DataFrame:
    """The order-level dataset used by the Historical Analysis run (synthetic, 240,000 rows)."""
    return pd.read_csv(ROOT / "data" / "uploads" / "orders_uploaded.csv")


def cost_params(mode: str) -> dict:
    """Economic parameters AS RUN for the given mode. Never mix across modes."""
    if mode == "historical":
        return load_manifest("historical")["cost_params"]
    df = load_results("future")
    r = df.iloc[0]
    return {
        "cost_late_urgent": float(r["cost_late_urgent"]),
        "cost_late_normal": float(r["cost_late_normal"]),
        "worker_cost_per_hour": float(r["worker_cost_per_hour"]),
        "hours_per_worker_month": float(r["hours_per_worker_month"]),
    }


def save(fig, name: str) -> Path:
    """Save a figure as PDF (vector, for LaTeX) and report the path."""
    out = FIGDIR / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fig] {out.relative_to(ROOT)}")
    return out


def save_table(df: pd.DataFrame, name: str, **kwargs) -> Path:
    out = TABDIR / f"{name}.csv"
    df.to_csv(out, index=False, **kwargs)
    print(f"  [tab] {out.relative_to(ROOT)}")
    return out
