"""
Dynamic workforce candidate generation around the analytical capacity estimate — spec §14.

Business workforce optimisation (Future Planning + Historical monthly search) no longer relies
solely on the static 16 tiny research regimes (configs/planning_profile.yaml::regimes, which
top out at s432 = 9 total workers and remain useful for RL research/benchmark/generalisation
diagnostics — see rl_generalisation). Instead it generates a bounded set of workforce
candidates around the analytical centre (src/analysis/capacity_estimate.py): single-stage
perturbations, leaner/safer total-workforce variants, balanced multi-stage combinations, and
(if given) the client's current workforce — never a full cubic grid, never a stage below 1.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.analysis.regime_naming import format_regime

Workers = Tuple[int, int, int]


def _clamp(p: int, k: int, d: int) -> Workers:
    return (max(1, int(p)), max(1, int(k)), max(1, int(d)))


def generate_worker_candidates(
    centre: Workers,
    candidate_count: int = 16,
    current_workforce: Optional[Workers] = None,
) -> List[Workers]:
    """Deterministic, bounded, operationally-plausible neighbourhood around `centre`.
    Always includes the centre first (and the client's current workforce next, if given), then
    single-stage +/-1 perturbations, leaner/safer total-workforce variants, balanced two-stage
    combinations, and finally +/-2 single-stage perturbations if the cap allows more — capped
    at `candidate_count` distinct (picking, packing, dispatch) tuples."""
    p0, k0, d0 = centre
    seen: set[Workers] = set()
    ordered: List[Workers] = []

    def add(p: int, k: int, d: int) -> None:
        t = _clamp(p, k, d)
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    add(p0, k0, d0)
    if current_workforce is not None:
        add(*current_workforce)

    # Single-stage +/-1
    add(p0 - 1, k0, d0); add(p0 + 1, k0, d0)
    add(p0, k0 - 1, d0); add(p0, k0 + 1, d0)
    add(p0, k0, d0 - 1); add(p0, k0, d0 + 1)

    # Leaner / safer total workforce (+/-15%, rounded, min 1 per stage)
    add(math.floor(p0 * 0.85), math.floor(k0 * 0.85), math.floor(d0 * 0.85))
    add(math.ceil(p0 * 1.15), math.ceil(k0 * 1.15), math.ceil(d0 * 1.15))

    # Balanced two-stage combinations
    add(p0 + 1, k0 + 1, d0); add(p0 + 1, k0, d0 + 1); add(p0, k0 + 1, d0 + 1)
    add(p0 - 1, k0 - 1, d0); add(p0 - 1, k0, d0 - 1); add(p0, k0 - 1, d0 - 1)

    # Wider single-stage spread if the cap still allows more
    add(p0 - 2, k0, d0); add(p0 + 2, k0, d0)
    add(p0, k0 - 2, d0); add(p0, k0 + 2, d0)
    add(p0, k0, d0 - 2); add(p0, k0, d0 + 2)

    return ordered[: max(1, int(candidate_count))]


def generate_regime_candidates(
    centre: Workers,
    candidate_count: int = 16,
    current_workforce: Optional[Workers] = None,
) -> Dict[str, Workers]:
    """Same as generate_worker_candidates, keyed by regime label (format_regime — handles the
    >9-workers-per-stage naming automatically)."""
    return {
        format_regime(*w): w
        for w in generate_worker_candidates(centre, candidate_count, current_workforce)
    }
