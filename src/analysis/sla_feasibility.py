"""
SLA feasibility check (spec §10) — shared by the base grid evaluator
(src/rl/evaluate_rl3_monthly_capacity_cost.py) and the adaptive capacity search
(src/analysis/capacity_search.py). Kept in its own tiny module because both of those need it
and evaluate_rl3_monthly_capacity_cost.py must not import capacity_search.py (which itself
imports evaluation helpers back from the evaluator — that would be a circular import).
"""
from __future__ import annotations

from typing import Tuple


def check_feasibility(
    urgent_sla: float, normal_sla: float, urgent_target: float, normal_target: float
) -> Tuple[bool, float]:
    """A candidate is feasible iff it meets both SLA floors. sla_violation is a transparent,
    documented score: the sum of each floor's shortfall (0 if met)."""
    feasible = bool(urgent_sla >= urgent_target and normal_sla >= normal_target)
    violation = max(0.0, urgent_target - urgent_sla) + max(0.0, normal_target - normal_sla)
    return feasible, round(float(violation), 4)
