"""
Two-stage screening + validation strategy for Future Planning base-regime evaluation
(spec §7).

Exhaustively evaluating every base workforce regime x 3 policies x 3 scenario replications
(16 x 3 x 3 = 144 simulations for the default settings) is unnecessarily slow: most regimes
are obviously uncompetitive after a single scenario realisation. Instead:

  Stage A — screening: evaluate every candidate regime x 3 policies on ONE scenario
  replication (seed #1) — see src/api/runners.py::run_future_planning.

  Stage B — validation: re-evaluate only the top `screening_finalists` regimes x 3 policies
  on the remaining scenario replications (seeds #2, #3), then aggregate all 3 replications
  for those regimes only (src/analysis/replication_aggregation.py).

Non-finalist regimes keep only their screening-replication metrics
(evaluation_stage="screening", replication_count=1) for diagnostic display — they are never
used to choose the final recommendation, which always comes from the finalists'
3-replication aggregate (evaluation_stage="validated").
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def _regime_rank_key(rows: pd.DataFrame) -> Tuple[float, float, float]:
    """Lower is better. Feasible regimes (any policy meets both SLA floors) always rank
    before infeasible ones; feasible regimes are ranked by their cheapest feasible policy,
    infeasible ones by lowest SLA violation then lowest cost — the same principle used to
    pick the final (regime, policy) recommendation (src/analysis/bottleneck_report.py::
    _select_recommendation), just applied per-regime instead of over the whole grid."""
    feasible_rows = rows[rows["feasible"]]
    if not feasible_rows.empty:
        return (0.0, 0.0, float(feasible_rows["estimated_total_cost"].min()))
    best = rows.sort_values(["sla_violation", "estimated_total_cost"]).iloc[0]
    return (1.0, float(best["sla_violation"]), float(best["estimated_total_cost"]))


def rank_regimes(df_screen: pd.DataFrame, regime_names: List[str]) -> List[str]:
    """Rank regime labels best-first using their single-replication screening rows."""
    return sorted(regime_names, key=lambda r: _regime_rank_key(df_screen[df_screen["regime"] == r]))


def select_finalists(df_screen: pd.DataFrame, regime_names: List[str], finalist_count: int) -> List[str]:
    ranked = rank_regimes(df_screen, regime_names)
    n = max(1, min(int(finalist_count), len(ranked)))
    return ranked[:n]
