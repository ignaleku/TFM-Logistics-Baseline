# src/rl/rl5_regimes.py
"""
Shared worker-regime definitions for the RL-5 5-stage pipeline.

Import this module wherever a regime list is needed to keep all scripts
consistent. Do not duplicate regime lists in individual evaluation scripts.

Regime format: (name, pick, qc, pack, lab, disp)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Full capacity planning regime matrix (15 regimes)
# ---------------------------------------------------------------------------
# Original 7 evaluation regimes + 5 expanded + 3 leaner alternatives

REGIMES_5STAGE: List[Tuple] = [
    # ── Original evaluation regimes ──────────────────────────────────────────
    ("s11111", 1, 1, 1, 1, 1),    # fully constrained baseline
    ("s21111", 2, 1, 1, 1, 1),    # picking relieved
    ("s31111", 3, 1, 1, 1, 1),    # picking comfortable, QC bottleneck
    ("s32111", 3, 2, 1, 1, 1),    # packing bottleneck
    ("s32211", 3, 2, 2, 1, 1),    # labelling/dispatch bottleneck
    ("s32221", 3, 2, 2, 2, 1),    # dispatch bottleneck
    ("s33322", 3, 3, 3, 2, 2),    # high capacity

    # ── Expanded capacity planning regimes ───────────────────────────────────
    ("s32121", 3, 2, 1, 2, 1),    # reinforced labelling, packing limited
    ("s32112", 3, 2, 1, 1, 2),    # reinforced dispatch, packing limited
    ("s32212", 3, 2, 2, 1, 2),    # reinforced dispatch after packing
    ("s33211", 3, 3, 2, 1, 1),    # reinforced QC
    ("s42211", 4, 2, 2, 1, 1),    # extra picking

    # ── Leaner alternatives (lower total headcount) ──────────────────────────
    ("s23211", 2, 3, 2, 1, 1),    # QC-heavy with fewer pickers
    ("s22121", 2, 2, 1, 2, 1),    # balanced with reinforced labelling
    ("s22112", 2, 2, 1, 1, 2),    # balanced with reinforced dispatch
]

# ---------------------------------------------------------------------------
# Evaluation-only subset — original 7 regimes for RL robustness evaluation
# (evaluate_rl5.py, evaluate_rl5_multiseed.py)
# ---------------------------------------------------------------------------
EVAL_REGIMES_5STAGE: List[Tuple] = REGIMES_5STAGE[:7]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def regime_to_resources(
    name: str,
    workers: Tuple[int, int, int, int, int],
    base_resources: Dict,
) -> Dict:
    """Return a resources dict with worker counts overridden from workers tuple."""
    n_pick, n_qc, n_pack, n_lab, n_disp = workers
    return {
        **base_resources,
        "picking_workers":       n_pick,
        "quality_check_workers": n_qc,
        "packing_workers":       n_pack,
        "labelling_workers":     n_lab,
        "dispatch_workers":      n_disp,
    }


def total_workers(workers: Tuple[int, int, int, int, int]) -> int:
    """Return total headcount for a worker configuration tuple."""
    return sum(workers)


def regime_names() -> List[str]:
    """Return list of all regime name strings in REGIMES_5STAGE."""
    return [r[0] for r in REGIMES_5STAGE]


def as_dict() -> Dict[str, Tuple[int, int, int, int, int]]:
    """Return REGIMES_5STAGE as {name: (pick, qc, pack, lab, disp)} dict."""
    return {r[0]: r[1:] for r in REGIMES_5STAGE}
