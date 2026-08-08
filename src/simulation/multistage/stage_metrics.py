"""
Stage-level operational metrics (bottleneck instrumentation).

Two pieces, kept deliberately small and event-based (no high-frequency polling, no giant
per-order export):

  QueueAreaTracker  — call .enqueue(t)/.dequeue(t) at the exact simpy event moments a store
                       gains/loses an item. Produces an exact time-weighted average queue
                       length (area under the queue-length curve / horizon) and max queue
                       length, independent of how irregularly events land in time.

  compute_stage_metrics — a pure pandas/numpy pass over the already-completed per-order stage
                       timestamps (arrival_time, start_pick/end_pick/start_pack/end_pack/
                       start_disp/end_disp, met_sla) that both simulation engines produce.
                       Derives busy-worker time, utilisation, service/wait distributions
                       (mean, p95), and late-order wait attribution per stage. The per-order
                       timestamps are discarded by the caller after this call — only the
                       aggregated stage dict is kept/returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

STAGES = ("picking", "packing", "dispatch")

_STAGE_START_END = {
    "picking":  ("start_pick", "end_pick"),
    "packing":  ("start_pack", "end_pack"),
    "dispatch": ("start_disp", "end_disp"),
}
# wait = time between becoming available for this stage and starting service at this stage
_STAGE_WAIT_FROM = {
    "picking":  "arrival_time",
    "packing":  "end_pick",
    "dispatch": "end_pack",
}


@dataclass
class QueueAreaTracker:
    """Time-weighted queue-length accumulator for one stage. Feed it enqueue/dequeue events
    as they happen (in simulation seconds); read back avg/max queue length at the end."""

    _queue_len: int = 0
    _last_t: float = 0.0
    _area: float = 0.0
    _max_queue: int = 0

    def _advance(self, t: float) -> None:
        dt = max(0.0, float(t) - self._last_t)
        self._area += self._queue_len * dt
        self._last_t = float(t)

    def enqueue(self, t: float) -> None:
        self._advance(t)
        self._queue_len += 1
        self._max_queue = max(self._max_queue, self._queue_len)

    def dequeue(self, t: float) -> None:
        self._advance(t)
        self._queue_len = max(0, self._queue_len - 1)

    def finalize(self, horizon_seconds: float) -> Dict[str, float]:
        self._advance(horizon_seconds)  # close out the area up to the simulation horizon
        avg_queue = (self._area / horizon_seconds) if horizon_seconds > 0 else 0.0
        return {
            "avg_queue_len": float(avg_queue),
            "max_queue_len": int(self._max_queue),
            # Backlog still sitting in this stage's queue when the operating horizon ended
            # (spec §11) — distinct from max_queue_len, which may have peaked mid-month.
            "end_of_horizon_queue_len": int(self._queue_len),
        }


def _safe_percentile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if arr.size else 0.0


def _safe_mean(arr: np.ndarray) -> float:
    return float(arr.mean()) if arr.size else 0.0


def compute_stage_metrics(
    df: pd.DataFrame,
    workers: Dict[str, int],
    horizon_seconds: float,
    queue_trackers: Optional[Dict[str, QueueAreaTracker]] = None,
) -> Dict[str, Dict[str, float]]:
    """df must have columns: arrival_time, start_pick, end_pick, start_pack, end_pack,
    start_disp, end_disp (pandas datetime-like), met_sla (bool). Returns one dict per stage.
    """
    horizon_minutes = horizon_seconds / 60.0
    late_mask = ~df["met_sla"].astype(bool)

    out: Dict[str, Dict[str, float]] = {}
    late_wait_by_stage: Dict[str, float] = {}

    for stage in STAGES:
        start_col, end_col = _STAGE_START_END[stage]
        wait_from_col = _STAGE_WAIT_FROM[stage]

        completed = df[df[end_col].notna() & df[start_col].notna()]
        service_min = ((completed[end_col] - completed[start_col]).dt.total_seconds() / 60.0).to_numpy()
        wait_min = ((completed[start_col] - completed[wait_from_col]).dt.total_seconds() / 60.0).to_numpy()
        wait_min = np.clip(wait_min, 0.0, None)

        n_workers = int(workers.get(stage, 0))
        busy_minutes = float(service_min.sum())
        available_minutes = n_workers * horizon_minutes
        utilisation = (busy_minutes / available_minutes) if available_minutes > 0 else 0.0
        # A worker can only be "on the clock" for the finite operating horizon (spec §10) — a
        # worker resource physically cannot log more busy-minutes than horizon_minutes, so
        # utilisation > 1.0 (beyond float rounding) indicates a real bug upstream, not a
        # legitimate value to silently clip away.
        if utilisation > 1.0 + 1e-6:
            import warnings
            warnings.warn(
                f"Stage '{stage}' utilisation={utilisation:.4f} exceeds 1.0 "
                f"(busy_minutes={busy_minutes:.2f} > available_minutes={available_minutes:.2f}) "
                "— a worker cannot be busy longer than the operating horizon. This indicates a "
                "bug upstream, not a legitimate value.",
                RuntimeWarning,
            )

        late_completed = completed[late_mask.reindex(completed.index, fill_value=False)]
        late_wait_min = float(
            np.clip(
                ((late_completed[start_col] - late_completed[wait_from_col]).dt.total_seconds() / 60.0).to_numpy(),
                0.0, None,
            ).sum()
        ) if len(late_completed) else 0.0
        late_wait_by_stage[stage] = late_wait_min

        q = {"avg_queue_len": 0.0, "max_queue_len": 0, "end_of_horizon_queue_len": 0}
        if queue_trackers and stage in queue_trackers:
            q = queue_trackers[stage].finalize(horizon_seconds)

        out[stage] = {
            "workers": n_workers,
            "processed_orders": int(len(completed)),
            "throughput_per_hour": (len(completed) / (horizon_minutes / 60.0)) if horizon_minutes > 0 else 0.0,
            "busy_worker_minutes": busy_minutes,
            "utilisation": float(min(1.0, utilisation)),
            "idle_capacity": float(max(0.0, 1.0 - utilisation)),
            "avg_service_min": _safe_mean(service_min),
            "p95_service_min": _safe_percentile(service_min, 95),
            "avg_wait_min": _safe_mean(wait_min),
            "p95_wait_min": _safe_percentile(wait_min, 95),
            "avg_queue_len": q["avg_queue_len"],
            "max_queue_len": q["max_queue_len"],
            "end_of_horizon_queue_len": q["end_of_horizon_queue_len"],
            "total_wait_min": float(wait_min.sum()),
            "late_wait_min": late_wait_min,
        }

    total_late_wait = sum(late_wait_by_stage.values())
    for stage in STAGES:
        out[stage]["late_wait_share"] = (
            late_wait_by_stage[stage] / total_late_wait if total_late_wait > 0 else 0.0
        )

    return out
