"""
Bottleneck ranking score (spec §7).

A transparent heuristic that combines four normalised signals per stage — utilisation,
p95 wait, late-order wait share, average queue length — into one comparable "pressure
score" so a single primary bottleneck can be named with an explanation. Weights are
configurable in configs/planning_profile.yaml (bottleneck_score).

Input: the `stage_metrics` dict produced by
src/simulation/multistage/stage_metrics.py::compute_stage_metrics (one entry per stage).
Output: one ranked row per stage, most-pressured first, with the score's components
exposed (not just the final number) plus a human-readable explanation for the top stage.

This is an explanatory operational indicator, not a formal causal proof — see
late_wait_share docstring in stage_metrics.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.data.planning_profile import load_planning_profile

STAGES = ("picking", "packing", "dispatch")
STAGE_LABELS = {"picking": "Picking", "packing": "Packing", "dispatch": "Dispatch"}


def _normalise(values: Dict[str, float]) -> Dict[str, float]:
    """Scale a per-stage dict to [0, 1] by its own max. All-zero input stays all-zero
    (never NaN)."""
    vals = np.array(list(values.values()), dtype=float)
    max_v = float(vals.max()) if vals.size else 0.0
    if max_v <= 0:
        return {k: 0.0 for k in values}
    return {k: float(max(0.0, v) / max_v) for k, v in values.items()}


def _explain(top: Dict[str, Any]) -> str:
    return (
        f"{top['stage_label']} is the primary bottleneck. It has {top['utilisation'] * 100:.1f}% "
        f"utilisation, a p95 wait of {top['p95_wait_min']:.1f} minutes, and accounts for "
        f"{top['late_wait_share'] * 100:.1f}% of the waiting time accumulated by late orders."
    )


def score_bottlenecks(
    stage_metrics: Dict[str, Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Returns one dict per stage, sorted by pressure_score descending (rank 1 = primary
    bottleneck). Handles the no-late-orders case gracefully (late_wait_share defaults to 0
    inside stage_metrics.py, never NaN)."""
    if weights is None:
        weights = load_planning_profile()["bottleneck_score"]

    utilisation = {s: min(1.0, max(0.0, float(stage_metrics[s]["utilisation"]))) for s in STAGES}
    p95_wait = {s: float(stage_metrics[s]["p95_wait_min"]) for s in STAGES}
    late_share = {s: float(stage_metrics[s].get("late_wait_share", 0.0)) for s in STAGES}
    avg_queue = {s: float(stage_metrics[s]["avg_queue_len"]) for s in STAGES}

    p95_norm = _normalise(p95_wait)
    queue_norm = _normalise(avg_queue)

    rows: List[Dict[str, Any]] = []
    for s in STAGES:
        util_c = weights["w_utilisation"] * utilisation[s]
        wait_c = weights["w_p95_wait"] * p95_norm[s]
        late_c = weights["w_late_wait_share"] * late_share[s]
        queue_c = weights["w_avg_queue"] * queue_norm[s]
        pressure = util_c + wait_c + late_c + queue_c

        rows.append({
            "stage": s,
            "stage_label": STAGE_LABELS[s],
            "pressure_score": round(float(pressure), 4),
            "utilisation_component": round(float(util_c), 4),
            "wait_component": round(float(wait_c), 4),
            "late_wait_component": round(float(late_c), 4),
            "queue_component": round(float(queue_c), 4),
            "utilisation": round(utilisation[s], 4),
            "p95_wait_min": round(p95_wait[s], 2),
            "avg_wait_min": round(float(stage_metrics[s]["avg_wait_min"]), 2),
            "avg_queue_len": round(avg_queue[s], 3),
            "max_queue_len": int(stage_metrics[s]["max_queue_len"]),
            "late_wait_share": round(late_share[s], 4),
        })

    rows.sort(key=lambda r: r["pressure_score"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r["is_primary_bottleneck"] = (i == 0)

    if rows:
        rows[0]["explanation"] = _explain(rows[0])
    return rows
