# src/simulation/multistage/sim_5stage.py
"""
5-stage logistics SimPy simulation.

Stage order: Picking → Quality Check → Packing → Labelling → Dispatch

Policies
--------
fifo          : all orders processed in arrival order (both types share one queue)
urgent_first  : urgent orders jump ahead of normal orders at every stage

Design note: two simpy.Store objects per stage (urgent / normal).
  - FIFO policy  : all orders routed to the normal store; urgent store stays empty.
  - urgent_first : urgent orders routed to the urgent store, processed first.
Workers always check the urgent store first, then the normal store.
This gives correct FIFO behaviour (all in normal → FIFO order preserved)
and correct urgent-first behaviour with zero branching inside the workers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import simpy
import numpy as np
import pandas as pd

from src.simulation.multistage.service_times_multistage import (
    sample_service_minutes,
    minutes_to_seconds_int,
)


# ── Order record ──────────────────────────────────────────────────────────────

@dataclass
class OrderTimes5:
    order_id:      int
    arrival_time:  pd.Timestamp
    order_type:    str
    sla_minutes:   int
    num_items:     int
    product_class: str
    scenario:      str

    start_pick: Optional[pd.Timestamp] = None
    end_pick:   Optional[pd.Timestamp] = None
    start_qc:   Optional[pd.Timestamp] = None
    end_qc:     Optional[pd.Timestamp] = None
    start_pack: Optional[pd.Timestamp] = None
    end_pack:   Optional[pd.Timestamp] = None
    start_lab:  Optional[pd.Timestamp] = None
    end_lab:    Optional[pd.Timestamp] = None
    start_disp: Optional[pd.Timestamp] = None
    end_disp:   Optional[pd.Timestamp] = None


# ── Simulation ────────────────────────────────────────────────────────────────

def run_simulation_5stage(
    orders: pd.DataFrame,
    sim_cfg: Dict,
    resources_cfg: Dict,
    service_cfg: Dict,
) -> Tuple[pd.DataFrame, Dict]:
    """Run a 5-stage simulation and return (per-order DataFrame, summary dict)."""

    seed        = int(sim_cfg["random_seed"])
    policy_name = sim_cfg.get("policy", "fifo").lower()
    rng         = np.random.default_rng(seed)

    orders = orders.sort_values("arrival_time").reset_index(drop=True)
    t0     = pd.to_datetime(orders["arrival_time"].min())

    def to_seconds(ts: pd.Timestamp) -> int:
        return int((ts - t0).total_seconds())

    def to_timestamp(sec: int) -> pd.Timestamp:
        return t0 + pd.to_timedelta(int(sec), unit="s")

    env = simpy.Environment()

    # Two stores per stage — urgent and normal
    pick_u, pick_n = simpy.Store(env), simpy.Store(env)
    qc_u,   qc_n   = simpy.Store(env), simpy.Store(env)
    pack_u, pack_n = simpy.Store(env), simpy.Store(env)
    lab_u,  lab_n  = simpy.Store(env), simpy.Store(env)
    disp_u, disp_n = simpy.Store(env), simpy.Store(env)

    rec:          Dict[int, OrderTimes5] = {}
    completed:    int                    = 0
    total_orders: int                    = len(orders)
    done_event = simpy.Event(env)

    def _use_urgent_lane(order_type: str) -> bool:
        """True only for urgent orders under the urgent_first policy."""
        return policy_name == "urgent_first" and order_type == "urgent"

    # ── Arrivals ──────────────────────────────────────────────────────────────

    def arrivals():
        for _, row in orders.iterrows():
            oid = int(row["order_id"])
            ats = pd.to_datetime(row["arrival_time"])
            yield env.timeout(max(to_seconds(ats) - int(env.now), 0))
            rec[oid] = OrderTimes5(
                order_id      = oid,
                arrival_time  = ats,
                order_type    = str(row["order_type"]),
                sla_minutes   = int(row["sla_minutes"]),
                num_items     = int(row["num_items"]),
                product_class = str(row["product_class"]),
                scenario      = str(row["scenario"]),
            )
            if _use_urgent_lane(str(row["order_type"])):
                yield pick_u.put(oid)
            else:
                yield pick_n.put(oid)

    # ── Stage workers ─────────────────────────────────────────────────────────
    # Each worker polls until its input queues are non-empty, then picks
    # urgent-first (urgent store checked before normal store).
    # For FIFO policy urgent stores are always empty, so workers always
    # pull from normal in arrival order.

    def picking_worker(_wid: int):
        while True:
            while len(pick_u.items) == 0 and len(pick_n.items) == 0:
                yield env.timeout(1)

            oid = yield (pick_u.get() if len(pick_u.items) > 0 else pick_n.get())
            ot  = rec[oid]

            ot.start_pick = to_timestamp(int(env.now))
            st = sample_service_minutes(rng, ot.num_items, ot.product_class, service_cfg["picking"])
            yield env.timeout(minutes_to_seconds_int(st))
            ot.end_pick = to_timestamp(int(env.now))

            if _use_urgent_lane(ot.order_type):
                yield qc_u.put(oid)
            else:
                yield qc_n.put(oid)

    def qc_worker(_wid: int):
        while True:
            while len(qc_u.items) == 0 and len(qc_n.items) == 0:
                yield env.timeout(1)

            oid = yield (qc_u.get() if len(qc_u.items) > 0 else qc_n.get())
            ot  = rec[oid]

            ot.start_qc = to_timestamp(int(env.now))
            st = sample_service_minutes(rng, ot.num_items, ot.product_class, service_cfg["quality_check"])
            yield env.timeout(minutes_to_seconds_int(st))
            ot.end_qc = to_timestamp(int(env.now))

            if _use_urgent_lane(ot.order_type):
                yield pack_u.put(oid)
            else:
                yield pack_n.put(oid)

    def packing_worker(_wid: int):
        while True:
            while len(pack_u.items) == 0 and len(pack_n.items) == 0:
                yield env.timeout(1)

            oid = yield (pack_u.get() if len(pack_u.items) > 0 else pack_n.get())
            ot  = rec[oid]

            ot.start_pack = to_timestamp(int(env.now))
            st = sample_service_minutes(rng, ot.num_items, ot.product_class, service_cfg["packing"])
            yield env.timeout(minutes_to_seconds_int(st))
            ot.end_pack = to_timestamp(int(env.now))

            if _use_urgent_lane(ot.order_type):
                yield lab_u.put(oid)
            else:
                yield lab_n.put(oid)

    def labelling_worker(_wid: int):
        while True:
            while len(lab_u.items) == 0 and len(lab_n.items) == 0:
                yield env.timeout(1)

            oid = yield (lab_u.get() if len(lab_u.items) > 0 else lab_n.get())
            ot  = rec[oid]

            ot.start_lab = to_timestamp(int(env.now))
            st = sample_service_minutes(rng, ot.num_items, ot.product_class, service_cfg["labelling"])
            yield env.timeout(minutes_to_seconds_int(st))
            ot.end_lab = to_timestamp(int(env.now))

            if _use_urgent_lane(ot.order_type):
                yield disp_u.put(oid)
            else:
                yield disp_n.put(oid)

    def dispatch_worker(_wid: int):
        nonlocal completed
        while True:
            while len(disp_u.items) == 0 and len(disp_n.items) == 0:
                yield env.timeout(1)

            oid = yield (disp_u.get() if len(disp_u.items) > 0 else disp_n.get())
            ot  = rec[oid]

            ot.start_disp = to_timestamp(int(env.now))
            st = sample_service_minutes(rng, ot.num_items, ot.product_class, service_cfg["dispatch"])
            yield env.timeout(minutes_to_seconds_int(st))
            ot.end_disp = to_timestamp(int(env.now))

            completed += 1
            if completed >= total_orders and not done_event.triggered:
                done_event.succeed()

    # ── Launch ────────────────────────────────────────────────────────────────

    env.process(arrivals())

    for i in range(int(resources_cfg["picking_workers"])):
        env.process(picking_worker(i))
    for i in range(int(resources_cfg["quality_check_workers"])):
        env.process(qc_worker(i))
    for i in range(int(resources_cfg["packing_workers"])):
        env.process(packing_worker(i))
    for i in range(int(resources_cfg["labelling_workers"])):
        env.process(labelling_worker(i))
    for i in range(int(resources_cfg["dispatch_workers"])):
        env.process(dispatch_worker(i))

    env.run(until=done_event)

    # ── Build results ─────────────────────────────────────────────────────────

    def _wait(start: Optional[pd.Timestamp], prev_end: Optional[pd.Timestamp]) -> float:
        if start is None or prev_end is None:
            return float("nan")
        return (start - prev_end).total_seconds() / 60.0

    rows = []
    for ot in rec.values():
        if ot.end_disp is None:
            continue
        system_min = (ot.end_disp - ot.arrival_time).total_seconds() / 60.0
        rows.append({
            "order_id":                 ot.order_id,
            "arrival_time":             ot.arrival_time,
            "order_type":               ot.order_type,
            "sla_minutes":              ot.sla_minutes,
            "num_items":                ot.num_items,
            "product_class":            ot.product_class,
            "scenario":                 ot.scenario,
            "policy":                   policy_name,
            "system_time_min":          system_min,
            "met_sla":                  system_min <= ot.sla_minutes,
            "wait_picking_min":         _wait(ot.start_pick, ot.arrival_time),
            "wait_quality_check_min":   _wait(ot.start_qc,   ot.end_pick),
            "wait_packing_min":         _wait(ot.start_pack, ot.end_qc),
            "wait_labelling_min":       _wait(ot.start_lab,  ot.end_pack),
            "wait_dispatch_min":        _wait(ot.start_disp, ot.end_lab),
        })

    df = pd.DataFrame(rows).sort_values("order_id").reset_index(drop=True)

    def _p90(col: str) -> float:
        return float(df[col].quantile(0.9))

    def _mean(col: str) -> float:
        return float(df[col].mean())

    summary = {
        "policy":                      policy_name,
        "seed":                        seed,
        "num_orders":                  len(df),
        "sla_rate":                    float(df["met_sla"].mean()),
        "sla_urgent":                  float(df.loc[df["order_type"] == "urgent",  "met_sla"].mean()),
        "sla_normal":                  float(df.loc[df["order_type"] == "normal",  "met_sla"].mean()),
        "mean_system_min":             _mean("system_time_min"),
        "p90_system_min":              _p90("system_time_min"),
        "mean_wait_picking_min":       _mean("wait_picking_min"),
        "mean_wait_quality_check_min": _mean("wait_quality_check_min"),
        "mean_wait_packing_min":       _mean("wait_packing_min"),
        "mean_wait_labelling_min":     _mean("wait_labelling_min"),
        "mean_wait_dispatch_min":      _mean("wait_dispatch_min"),
        "p90_wait_picking_min":        _p90("wait_picking_min"),
        "p90_wait_quality_check_min":  _p90("wait_quality_check_min"),
        "p90_wait_packing_min":        _p90("wait_packing_min"),
        "p90_wait_labelling_min":      _p90("wait_labelling_min"),
        "p90_wait_dispatch_min":       _p90("wait_dispatch_min"),
    }

    return df, summary