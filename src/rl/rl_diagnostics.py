"""
Diagnostic-only instrumentation for RL-3 — Q-value/state logging, state-feature documentation,
and sequential/structural comparison helpers.

This module makes ZERO changes to src/rl/env_fullstage_rl.py. `LoggingGreedyAgent` below is a
drop-in replacement for the plain greedy agent used everywhere else
(evaluate_rl3_monthly_capacity_cost.py::_GreedyAgent, rl_audit.py). Its `.act()` runs the exact
same `self.q(state)` forward pass, in the same eval() mode, and returns the exact same
`argmax` — it only additionally appends a diagnostic row as a side effect. Since
`FullStageRLRunner._decide()` calls `agent.act(s, greedy=greedy)` ONLY at real decision points
(both queues non-empty — see env_fullstage_rl.py), this wrapper naturally captures exactly, and
only, those points, with no change to which order gets serviced next and no forced-decision
noise mixed in.

Used by src/rl/rl_audit.py's historical-diagnostic CLI additions. Kept separate from rl_audit.py
(rather than folded in) because it also hosts the state-feature table and the sequence/joint
comparison helpers used by the standalone diagnostic experiments, which are not audit-report
concerns per se.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.rl.dqn_agent import QNetwork

# ── §J: 16-dimensional state feature table — read directly from env_fullstage_rl.py::_state ──
# (do not guess: this mirrors that function's feature order and formulas exactly)

STATE_FEATURE_TABLE: List[Dict[str, Any]] = [
    {"index": 0, "name": "picking_urgent_queue_len_norm", "formula": "clip01(len(pick_u.items) / 200)",
     "normalisation": "/200, clip[0,1]", "expected_scale": "0-1, saturates at >=200 orders queued"},
    {"index": 1, "name": "picking_normal_queue_len_norm", "formula": "clip01(len(pick_n.items) / 500)",
     "normalisation": "/500, clip[0,1]", "expected_scale": "0-1, saturates at >=500 orders queued"},
    {"index": 2, "name": "packing_urgent_queue_len_norm", "formula": "clip01(len(pack_u.items) / 200)",
     "normalisation": "/200, clip[0,1]", "expected_scale": "0-1, saturates at >=200 orders queued"},
    {"index": 3, "name": "packing_normal_queue_len_norm", "formula": "clip01(len(pack_n.items) / 500)",
     "normalisation": "/500, clip[0,1]", "expected_scale": "0-1, saturates at >=500 orders queued"},
    {"index": 4, "name": "dispatch_urgent_queue_len_norm", "formula": "clip01(len(disp_u.items) / 200)",
     "normalisation": "/200, clip[0,1]", "expected_scale": "0-1, saturates at >=200 orders queued"},
    {"index": 5, "name": "dispatch_normal_queue_len_norm", "formula": "clip01(len(disp_n.items) / 500)",
     "normalisation": "/500, clip[0,1]", "expected_scale": "0-1, saturates at >=500 orders queued"},
    {"index": 6, "name": "picking_wip_norm", "formula": "clip01(wip_pick / 5)",
     "normalisation": "/5, clip[0,1]", "expected_scale": "0-1; wip <= picking_workers"},
    {"index": 7, "name": "packing_wip_norm", "formula": "clip01(wip_pack / 5)",
     "normalisation": "/5, clip[0,1]", "expected_scale": "0-1; wip <= packing_workers"},
    {"index": 8, "name": "dispatch_wip_norm", "formula": "clip01(wip_disp / 5)",
     "normalisation": "/5, clip[0,1]", "expected_scale": "0-1; wip <= dispatch_workers"},
    {"index": 9, "name": "time_norm", "formula": "clip01(now_s / horizon_s)",
     "normalisation": "/episode horizon seconds, clip[0,1]", "expected_scale": "0-1, monotonic across episode"},
    {"index": 10, "name": "slack_urgent_head_current_stage",
     "formula": "(clip((sla_minutes - system_min)/sla_minutes, -1, 1) + 1) / 2 of head-of-queue urgent order in the CURRENT stage's queue, else 0.5 if empty",
     "normalisation": "affine map of clipped relative SLA slack to [0,1]", "expected_scale": "0-1; 0.5=empty/neutral, >0.5 ahead of SLA, <0.5 behind"},
    {"index": 11, "name": "slack_normal_head_current_stage",
     "formula": "same as feature 10, normal queue", "normalisation": "same",
     "expected_scale": "0-1"},
    {"index": 12, "name": "stage_id", "formula": "constant per stage: 0.0 picking / 0.5 packing / 1.0 dispatch",
     "normalisation": "none", "expected_scale": "{0.0, 0.5, 1.0}"},
    {"index": 13, "name": "cap_picking_norm", "formula": "clip01(picking_workers / capacity_feature_scale)",
     "normalisation": "/rl_generalisation.capacity_feature_scale (20 in current config), clip[0,1]",
     "expected_scale": "0-1; e.g. 8 workers -> 0.40"},
    {"index": 14, "name": "cap_packing_norm", "formula": "clip01(packing_workers / capacity_feature_scale)",
     "normalisation": "same", "expected_scale": "0-1"},
    {"index": 15, "name": "cap_dispatch_norm", "formula": "clip01(dispatch_workers / capacity_feature_scale)",
     "normalisation": "same", "expected_scale": "0-1"},
]

_QUEUE_FEATURE_IDX = {"picking": (0, 1), "packing": (2, 3), "dispatch": (4, 5)}
_QUEUE_DIVISOR = {"urgent": 200.0, "normal": 500.0}
_STAGE_BY_ID = {0.0: "picking", 0.5: "packing", 1.0: "dispatch"}


def _approx_queue_len(state: np.ndarray, stage: str, cls: str) -> Tuple[float, bool]:
    """Reconstructs an approximate raw queue length from the normalised state feature — exact
    below the saturation point (200 urgent / 500 normal), only a lower bound once saturated
    (state feature reads exactly 1.0). Avoids touching env_fullstage_rl.py to log the raw
    simpy.Store length directly; the saturation flag is the more important signal anyway."""
    idx = _QUEUE_FEATURE_IDX[stage][0 if cls == "urgent" else 1]
    norm_val = float(state[idx])
    divisor = _QUEUE_DIVISOR[cls]
    saturated = norm_val >= 0.999999
    return norm_val * divisor, saturated


class LoggingGreedyAgent:
    """Diagnostic drop-in for the plain greedy agent (_GreedyAgent in
    evaluate_rl3_monthly_capacity_cost.py / rl_audit.py) — identical forward pass and returned
    action, plus a per-decision diagnostic log. See module docstring for why this cannot change
    simulated behaviour."""

    def __init__(self, q_net: QNetwork, device: str = "cpu", resources_cfg: Optional[Dict[str, int]] = None):
        self.q = q_net
        self.device = device
        self.resources_cfg = resources_cfg or {}
        self.rows: List[Dict[str, Any]] = []

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q(s).squeeze(0)
            action = int(qvals.argmax().item())

        q0 = float(qvals[0].item())
        q1 = float(qvals[1].item())
        stage_id = float(state[12])
        stage = _STAGE_BY_ID.get(stage_id, f"unknown(stage_id={stage_id})")
        uq, uq_sat = _approx_queue_len(state, stage, "urgent") if stage in _QUEUE_FEATURE_IDX else (float("nan"), False)
        nq, nq_sat = _approx_queue_len(state, stage, "normal") if stage in _QUEUE_FEATURE_IDX else (float("nan"), False)

        self.rows.append({
            "decision_idx": len(self.rows),
            "stage": stage,
            "time_norm": float(state[9]),
            "state": [float(x) for x in state],
            "q_urgent": q0,
            "q_normal": q1,
            "q_margin_urgent_minus_normal": q0 - q1,
            "action": action,  # 0 = urgent, 1 = normal
            "approx_urgent_queue_len": uq,
            "approx_urgent_queue_len_saturated": uq_sat,
            "approx_normal_queue_len": nq,
            "approx_normal_queue_len_saturated": nq_sat,
            "picking_workers": int(self.resources_cfg.get("picking_workers", 0)),
            "packing_workers": int(self.resources_cfg.get("packing_workers", 0)),
            "dispatch_workers": int(self.resources_cfg.get("dispatch_workers", 0)),
        })
        return action


def _percentiles(arr: np.ndarray, qs=(1, 50, 99)) -> Dict[str, float]:
    if arr.size == 0:
        return {f"p{q:02d}": None for q in qs}
    return {f"p{q:02d}": float(np.percentile(arr, q)) for q in qs}


def summarize_qvalue_log(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate Q-value diagnostics for one trajectory's logged decisions (§I)."""
    n = len(rows)
    if n == 0:
        return {"decisions_both_queues_available": 0, "note": "no both-queue-nonempty decisions were recorded"}

    q0 = np.array([r["q_urgent"] for r in rows], dtype=float)
    q1 = np.array([r["q_normal"] for r in rows], dtype=float)
    margin = q0 - q1
    actions = np.array([r["action"] for r in rows], dtype=int)

    def _dist(a: np.ndarray) -> Dict[str, Any]:
        finite = a[np.isfinite(a)]
        return {
            "mean": float(finite.mean()) if finite.size else None,
            "std": float(finite.std()) if finite.size else None,
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            **_percentiles(finite),
            "nan_count": int(np.isnan(a).sum()),
            "inf_count": int(np.isinf(a).sum()),
        }

    small_margin_thresh = 0.01
    return {
        "decisions_both_queues_available": n,
        "action_urgent_pct": float((actions == 0).mean()),
        "action_normal_pct": float((actions == 1).mean()),
        "q_urgent_dist": _dist(q0),
        "q_normal_dist": _dist(q1),
        "q_margin_dist": _dist(margin),
        "pct_abs_margin_below_%.3f" % small_margin_thresh: float((np.abs(margin) < small_margin_thresh).mean()),
        "pct_strongly_prefers_urgent_margin_gt_1": float((margin > 1.0).mean()),
        "pct_strongly_prefers_normal_margin_lt_neg1": float((margin < -1.0).mean()),
        "approx_urgent_queue_saturated_pct": float(np.mean([r["approx_urgent_queue_len_saturated"] for r in rows])),
        "approx_normal_queue_saturated_pct": float(np.mean([r["approx_normal_queue_len_saturated"] for r in rows])),
    }


def sample_decision_rows(rows: List[Dict[str, Any]], n_first: int = 100, n_even: int = 100, n_top_margin: int = 50) -> List[Dict[str, Any]]:
    """Compact deterministic sample: first N, N evenly spaced, and top-N by |margin| — per spec,
    to keep diagnostic output small without losing the most informative rows. Rows keep the
    full state vector (16 floats) but nothing per-order beyond that (no giant per-order dump)."""
    if not rows:
        return []
    n = len(rows)
    idx = set(range(min(n_first, n)))
    if n_even > 0:
        idx.update(int(i) for i in np.linspace(0, n - 1, min(n_even, n), dtype=int))
    margins = sorted(range(n), key=lambda i: abs(rows[i]["q_margin_urgent_minus_normal"]), reverse=True)
    idx.update(margins[:n_top_margin])
    return [rows[i] for i in sorted(idx)]


def state_feature_distribution(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-feature min/p01/mean/median/p99/max/std across a trajectory's logged decision states
    (§7). This is an INFERENCE-TIME reference distribution over states actually visited during
    this run — not a recorded training-time distribution (no training-time state logs exist;
    see diagnostic_summary.json for that caveat stated explicitly)."""
    if not rows:
        return []
    states = np.array([r["state"] for r in rows], dtype=float)  # (n, 16)
    out = []
    for i, feat in enumerate(STATE_FEATURE_TABLE):
        col = states[:, i]
        out.append({
            "index": i, "name": feat["name"],
            "min": float(col.min()), "p01": float(np.percentile(col, 1)),
            "mean": float(col.mean()), "median": float(np.median(col)),
            "p99": float(np.percentile(col, 99)), "max": float(col.max()),
            "std": float(col.std()),
        })
    return out


def ood_indicators(target_dist: List[Dict[str, Any]], reference_dists: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compares a target trajectory's per-feature distribution against a reference ENVELOPE
    built from one or more other trajectories (min/max, and p01/p99, pooled across the
    reference runs). Explicitly an inference-time reference-trajectory comparison, not a
    training-state-log comparison (§7)."""
    if not target_dist or not reference_dists:
        return {"note": "insufficient data for OOD comparison"}

    n_feat = len(target_dist)
    ref_min = [min(rd[i]["min"] for rd in reference_dists) for i in range(n_feat)]
    ref_max = [max(rd[i]["max"] for rd in reference_dists) for i in range(n_feat)]
    ref_p01 = [min(rd[i]["p01"] for rd in reference_dists) for i in range(n_feat)]
    ref_p99 = [max(rd[i]["p99"] for rd in reference_dists) for i in range(n_feat)]

    per_feature = []
    for i, t in enumerate(target_dist):
        outside_minmax = t["min"] < ref_min[i] or t["max"] > ref_max[i]
        outside_p01p99 = t["p01"] < ref_p01[i] or t["p99"] > ref_p99[i]
        per_feature.append({
            "index": i, "name": t["name"],
            "target_range": [t["min"], t["max"]],
            "reference_minmax_envelope": [ref_min[i], ref_max[i]],
            "reference_p01p99_envelope": [ref_p01[i], ref_p99[i]],
            "outside_reference_minmax": bool(outside_minmax),
            "outside_reference_p01p99_envelope": bool(outside_p01p99),
        })

    return {
        "caveat": "reference envelope is built from other INFERENCE-TIME trajectories in this "
                  "diagnostic run (a 'training-like reference'), not from logs recorded during "
                  "actual training (no such logs exist for this checkpoint).",
        "features_outside_minmax_envelope": [f["name"] for f in per_feature if f["outside_reference_minmax"]],
        "features_outside_p01p99_envelope": [f["name"] for f in per_feature if f["outside_reference_p01p99_envelope"]],
        "per_feature": per_feature,
    }


# ── §M: sequential / joint structure comparison (Experiment 6) ────────────────────────────────

def sequence_structure_summary(orders: pd.DataFrame, window_minutes: float = 5.0) -> Dict[str, Any]:
    """Compact sequential/joint-structure statistics for one order trajectory, computed on
    operating-time-compressed `arrival_time` (or any monotonic arrival timestamp column) plus
    order_type / *_units. Designed to be small and comparable across trajectories, not an
    exhaustive feature set."""
    df = orders.sort_values("arrival_time").reset_index(drop=True)
    t0 = df["arrival_time"].min()
    minutes = (df["arrival_time"] - t0).dt.total_seconds() / 60.0
    is_urgent = (df["order_type"] == "urgent").to_numpy()

    def _window_counts(win_min: float) -> np.ndarray:
        bins = np.floor(minutes / win_min).astype(int)
        return np.bincount(bins)

    def _window_stats(win_min: float) -> Dict[str, float]:
        counts = _window_counts(win_min).astype(float)
        mean = counts.mean()
        std = counts.std()
        return {
            "mean": float(mean), "std": float(std),
            "cv": float(std / mean) if mean > 0 else None,
            "p95": float(np.percentile(counts, 95)), "p99": float(np.percentile(counts, 99)),
            "max": float(counts.max()),
            "peak_over_mean": float(counts.max() / mean) if mean > 0 else None,
        }

    gaps = np.diff(np.sort(minutes.to_numpy()))
    gaps = gaps[gaps >= 0]

    counts_60 = _window_counts(60.0).astype(float)
    urgent_counts_60 = np.bincount(np.floor(minutes[is_urgent] / 60.0).astype(int), minlength=len(counts_60)).astype(float)
    urgent_share_by_window = np.divide(urgent_counts_60, counts_60, out=np.zeros_like(counts_60), where=counts_60 > 0)
    nz = counts_60 > 0

    def _autocorr_lag(x: np.ndarray, lag: int) -> Optional[float]:
        if len(x) <= lag + 1 or np.std(x) == 0:
            return None
        a, b = x[:-lag], x[lag:]
        if np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    def _run_lengths(mask: np.ndarray) -> np.ndarray:
        if mask.size == 0:
            return np.array([])
        change = np.diff(mask.astype(int))
        starts = np.where(change != 0)[0] + 1
        segs = np.split(mask, starts)
        return np.array([len(s) for s in segs if len(s) and s[0]])

    urgent_runs = _run_lengths(is_urgent)
    normal_runs = _run_lengths(~is_urgent)

    workload_cols = [c for c in ("picking_units", "packing_units", "dispatch_units") if c in df.columns]
    workload_60: Dict[str, Any] = {}
    for c in workload_cols:
        bins = np.floor(minutes / 60.0).astype(int)
        total = np.bincount(bins, weights=df[c].to_numpy(), minlength=len(counts_60))
        mean = total.mean()
        workload_60[c] = {
            "cv": float(total.std() / mean) if mean > 0 else None,
            "p95": float(np.percentile(total, 95)), "p99": float(np.percentile(total, 99)),
            "max": float(total.max()), "peak_over_mean": float(total.max() / mean) if mean > 0 else None,
        }

    complexity_rank = {"low": 0, "medium": 1, "high": 2}
    complexity_num = df["complexity_level"].map(complexity_rank).to_numpy(dtype=float) if "complexity_level" in df.columns else None

    def _corr(a, b) -> Optional[float]:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    joint: Dict[str, Any] = {}
    if complexity_num is not None:
        bins = np.floor(minutes / 60.0).astype(int)
        mean_complexity_60 = np.array([complexity_num[bins == b].mean() if (bins == b).any() else np.nan for b in range(len(counts_60))])
        valid = ~np.isnan(mean_complexity_60)
        joint["corr_arrival_count_vs_mean_complexity"] = _corr(counts_60[valid], mean_complexity_60[valid])
    joint["corr_arrival_count_vs_urgent_share"] = _corr(counts_60[nz], urgent_share_by_window[nz])
    if "picking_units" in workload_cols:
        bins = np.floor(minutes / 60.0).astype(int)
        total_pick = np.bincount(bins, weights=df["picking_units"].to_numpy(), minlength=len(counts_60))
        joint["corr_arrival_count_vs_total_picking_workload"] = _corr(counts_60, total_pick)
        joint["corr_urgent_share_vs_total_picking_workload"] = _corr(urgent_share_by_window[nz], total_pick[nz])

    return {
        "n_orders": int(len(df)),
        "span_minutes": float(minutes.max()),
        "arrival_intensity": {
            "window_5min": _window_stats(5.0),
            "window_15min": _window_stats(15.0),
            "window_60min": _window_stats(60.0),
        },
        "interarrival_gaps_minutes": {
            "p50": float(np.percentile(gaps, 50)) if gaps.size else None,
            "p95": float(np.percentile(gaps, 95)) if gaps.size else None,
            "p99": float(np.percentile(gaps, 99)) if gaps.size else None,
            "max": float(gaps.max()) if gaps.size else None,
        },
        "temporal_autocorrelation_60min_counts": {
            "lag1": _autocorr_lag(counts_60, 1),
            "lag2": _autocorr_lag(counts_60, 2),
            "lag3": _autocorr_lag(counts_60, 3),
        },
        "class_clustering": {
            "urgent_share": float(is_urgent.mean()),
            "urgent_run_length": {"mean": float(urgent_runs.mean()) if urgent_runs.size else None, "max": float(urgent_runs.max()) if urgent_runs.size else None},
            "normal_run_length": {"mean": float(normal_runs.mean()) if normal_runs.size else None, "max": float(normal_runs.max()) if normal_runs.size else None},
            "urgent_share_by_60min_window_cv": float(np.std(urgent_share_by_window[nz]) / np.mean(urgent_share_by_window[nz])) if nz.any() and np.mean(urgent_share_by_window[nz]) > 0 else None,
        },
        "workload_clustering_60min": workload_60,
        "joint_pressure_correlations": joint,
    }
