"""
RL-3 generalisation evaluation (spec §12 / §12.1 / §19.6).

Evaluates the trained checkpoint on the 12 training regimes vs. the 4 exact-held-out regimes
(configs/planning_profile.yaml::rl_generalisation) for one or more months, and reports grouped
metrics (mean cost, SLA, urgent/normal SLA, late orders, feasibility rate) alongside FIFO and
urgent_first for the same regimes — so "does RL win" is never the only lens; feasibility,
stability and degradation from seen to unseen matter too.

Usage:
    python -m src.rl.evaluate_rl3_generalisation --months December
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data.planning_profile import load_planning_profile
from src.rl.evaluate_rl3_monthly_capacity_cost import evaluate_monthly_capacity_cost, parse_months

ROOT = Path(__file__).resolve().parents[2]


def _group_summary(df: pd.DataFrame, regimes: List[str], policy: str) -> Dict:
    sub = df[(df["regime"].isin(regimes)) & (df["policy"] == policy)]
    if sub.empty:
        return {"n": 0}
    return {
        "n": int(len(sub)),
        "mean_total_cost": round(float(sub["estimated_total_cost"].mean()), 2),
        "mean_total_sla": round(float(sub["total_sla"].mean()), 4),
        "mean_urgent_sla": round(float(sub["urgent_sla"].mean()), 4),
        "mean_normal_sla": round(float(sub["normal_sla"].mean()), 4),
        "mean_urgent_late_orders": round(float(sub["urgent_late_orders"].mean()), 2),
        "mean_normal_late_orders": round(float(sub["normal_late_orders"].mean()), 2),
        "feasible_pct": round(float(sub["feasible"].mean()), 4),
    }


def run_generalisation_eval(
    orders_all: pd.DataFrame,
    checkpoint_path: Path,
    months: Optional[List[int]] = None,
    root: Optional[Path] = None,
) -> Dict:
    root = root or ROOT
    profile = load_planning_profile()
    train_regimes = profile["rl_generalisation"]["train_regimes"]
    holdout_regimes = profile["rl_generalisation"]["holdout_regimes"]

    df = evaluate_monthly_capacity_cost(
        orders_all, checkpoint_path, months=months,
        regime_names=train_regimes + holdout_regimes, root=root,
    )

    result = {"train_regimes": train_regimes, "holdout_regimes": holdout_regimes, "by_policy": {}}
    for policy in ("fifo", "urgent_first", "rl3_dqn"):
        seen = _group_summary(df, train_regimes, policy)
        unseen = _group_summary(df, holdout_regimes, policy)
        degradation = None
        if seen.get("n") and unseen.get("n"):
            degradation = {
                "total_sla_delta": round(unseen["mean_total_sla"] - seen["mean_total_sla"], 4),
                "urgent_sla_delta": round(unseen["mean_urgent_sla"] - seen["mean_urgent_sla"], 4),
                "normal_sla_delta": round(unseen["mean_normal_sla"] - seen["mean_normal_sla"], 4),
                "total_cost_delta": round(unseen["mean_total_cost"] - seen["mean_total_cost"], 2),
                "feasible_pct_delta": round(unseen["feasible_pct"] - seen["feasible_pct"], 4),
            }
        result["by_policy"][policy] = {"seen": seen, "unseen": unseen, "unseen_minus_seen": degradation}

    rl3 = result["by_policy"]["rl3_dqn"]
    result["interpretation"] = (
        "RL-3 generalisation from seen to unseen regimes: "
        f"total SLA {rl3['seen'].get('mean_total_sla')} -> {rl3['unseen'].get('mean_total_sla')}, "
        f"feasible% {rl3['seen'].get('feasible_pct')} -> {rl3['unseen'].get('feasible_pct')}. "
        "The goal of this evaluation is to characterise degradation, not to force RL to win "
        "every regime — see by_policy for the FIFO/urgent_first comparison on the same regimes."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-3 seen vs. unseen (holdout) regime evaluation")
    parser.add_argument("--orders", default="data/orders_base_seasonal.csv")
    parser.add_argument("--checkpoint", default="data/dqn_rl3_final.pt")
    parser.add_argument("--months", default=None)
    parser.add_argument("--output", default="data/rl3_generalisation_results.json")
    args = parser.parse_args()

    root = ROOT
    orders_all = pd.read_csv(root / args.orders, parse_dates=["arrival_time"])
    months = parse_months(args.months) if args.months else None

    result = run_generalisation_eval(orders_all, root / args.checkpoint, months=months, root=root)

    out_path = root / args.output
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
