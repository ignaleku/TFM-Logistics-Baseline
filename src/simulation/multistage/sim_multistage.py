from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import simpy
import numpy as np
import pandas as pd

from src.simulation.multistage.service_times_multistage import (
    sample_service_minutes,
    minutes_to_seconds_int,
)
from src.simulation.multistage.stage_metrics import QueueAreaTracker, compute_stage_metrics
from src.simulation.multistage.operating_time import SIM_EPOCH

# Required columns for the enriched workload model
_REQUIRED_WORKLOAD_COLS = {"picking_units", "packing_units", "dispatch_units"}


@dataclass
class OrderTimes:
    order_id: int
    arrival_time: pd.Timestamp
    order_type: str
    sla_minutes: int
    num_items: int
    product_class: str
    scenario: str
    picking_units: float = 1.0
    packing_units: float = 1.0
    dispatch_units: float = 1.0

    start_pick: Optional[pd.Timestamp] = None
    end_pick: Optional[pd.Timestamp] = None
    start_pack: Optional[pd.Timestamp] = None
    end_pack: Optional[pd.Timestamp] = None
    start_disp: Optional[pd.Timestamp] = None
    end_disp: Optional[pd.Timestamp] = None


def run_simulation_multistage(
    orders: pd.DataFrame,
    sim_cfg: Dict,
    resources_cfg: Dict,
    service_cfg: Dict,
    service_time_map: Optional[Dict[Tuple[int, str], float]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Run FIFO / urgent_first over `orders`, for a FINITE monthly operating horizon.

    `sim_cfg['operating_horizon_minutes']` is required (see operating_time.py) — the simulated
    clock runs for exactly this long (worker resources are only ever "on the clock" for the
    same number of minutes the economic model pays them for). Orders arriving must already be
    expressed on this clock, i.e. `orders['arrival_time']` must already be operating-time
    (SIM_EPOCH-anchored) — see operating_time.py::compress_to_operating_time /
    slice_month_operating_time, which every caller applies before reaching this function.

    Orders not completed by the time the horizon ends are BACKLOG — unresolved, not deadlocked.
    They count as SLA failures (met_sla=False, since system_time is NaN and every comparison
    against it is False) and are additionally reported separately via
    summary['unfinished_orders'] / 'unfinished_urgent_orders' / 'unfinished_normal_orders' /
    'backlog_share' so monthly capacity backlog is visible, not hidden inside the SLA rate.

    If `service_time_map` is given (see service_time_map.py::build_service_time_map), stage
    service times are looked up from it instead of sampled inline — this is what makes
    FIFO / urgent_first / RL-3 comparable under common random numbers (identical service time
    per order/stage regardless of policy or dequeue order). If omitted, falls back to inline
    per-draw sampling using sim_cfg['random_seed'] (legacy / standalone use).
    """

    if orders is None or len(orders) == 0:
        raise ValueError("run_simulation_multistage: orders DataFrame is empty.")

    missing = _REQUIRED_WORKLOAD_COLS - set(orders.columns)
    if missing:
        raise ValueError(
            f"Orders DataFrame is missing required workload columns: {sorted(missing)}. "
            "Regenerate with: python -m src.data.generate_orders_seasonal "
            "or enrich via POST /upload-orders."
        )
    if "operating_horizon_minutes" not in sim_cfg:
        raise ValueError(
            "sim_cfg['operating_horizon_minutes'] is required — see "
            "operating_time.py::with_operating_horizon. The simulation no longer runs until "
            "all orders complete; it runs for a finite monthly capacity horizon."
        )

    seed = int(sim_cfg["random_seed"])
    policy_name = sim_cfg.get("policy", "fifo").lower()
    rng = np.random.default_rng(seed)
    horizon_minutes = float(sim_cfg["operating_horizon_minutes"])
    horizon_seconds = horizon_minutes * 60.0

    def _service_minutes(order_id: int, stage: str, units: float) -> float:
        if service_time_map is not None:
            return service_time_map[(order_id, stage)]
        return sample_service_minutes(rng, units, service_cfg[stage])

    orders = orders.sort_values("arrival_time").reset_index(drop=True)
    t0 = SIM_EPOCH

    def to_seconds(ts: pd.Timestamp) -> int:
        return int((ts - t0).total_seconds())

    def to_timestamp(sec: int) -> pd.Timestamp:
        return t0 + pd.to_timedelta(int(sec), unit="s")

    env = simpy.Environment()

    if policy_name == "fifo":
        pick_store = simpy.Store(env)
        pack_store = simpy.Store(env)
        disp_store = simpy.Store(env)
    else:
        # urgent_first: single PriorityStore per stage (priority 0=urgent, 1=normal)
        pick_store = simpy.PriorityStore(env)
        pack_store = simpy.PriorityStore(env)
        disp_store = simpy.PriorityStore(env)

    store: Dict[int, OrderTimes] = {}
    completed = 0
    total_orders = len(orders)

    queue_trackers = {stage: QueueAreaTracker() for stage in ("picking", "packing", "dispatch")}

    def arrivals():
        for _, row in orders.iterrows():
            order_id = int(row["order_id"])
            arrival_ts = pd.to_datetime(row["arrival_time"])
            arrival_sec = to_seconds(arrival_ts)

            store[order_id] = OrderTimes(
                order_id=order_id,
                arrival_time=arrival_ts,
                order_type=str(row["order_type"]),
                sla_minutes=int(row["sla_minutes"]),
                num_items=int(row["num_items"]),
                product_class=str(row["product_class"]),
                scenario=str(row["scenario"]),
                picking_units=float(row["picking_units"]),
                packing_units=float(row["packing_units"]),
                dispatch_units=float(row["dispatch_units"]),
            )

            yield env.timeout(max(arrival_sec - int(env.now), 0))

            queue_trackers["picking"].enqueue(env.now)
            if policy_name == "fifo":
                yield pick_store.put(order_id)
            else:
                prio = 0 if row["order_type"] == "urgent" else 1
                yield pick_store.put(simpy.PriorityItem(prio, order_id))

    def picking_worker(_wid: int):
        while True:
            if policy_name == "fifo":
                order_id = yield pick_store.get()
            else:
                order_id = (yield pick_store.get()).item
            queue_trackers["picking"].dequeue(env.now)

            ot = store[order_id]
            ot.start_pick = to_timestamp(int(env.now))

            st_min = _service_minutes(order_id, "picking", ot.picking_units)
            yield env.timeout(minutes_to_seconds_int(st_min))

            ot.end_pick = to_timestamp(int(env.now))

            queue_trackers["packing"].enqueue(env.now)
            if policy_name == "fifo":
                yield pack_store.put(order_id)
            else:
                prio = 0 if ot.order_type == "urgent" else 1
                yield pack_store.put(simpy.PriorityItem(prio, order_id))

    def packing_worker(_wid: int):
        while True:
            if policy_name == "fifo":
                order_id = yield pack_store.get()
            else:
                order_id = (yield pack_store.get()).item
            queue_trackers["packing"].dequeue(env.now)

            ot = store[order_id]
            ot.start_pack = to_timestamp(int(env.now))

            st_min = _service_minutes(order_id, "packing", ot.packing_units)
            yield env.timeout(minutes_to_seconds_int(st_min))

            ot.end_pack = to_timestamp(int(env.now))

            queue_trackers["dispatch"].enqueue(env.now)
            if policy_name == "fifo":
                yield disp_store.put(order_id)
            else:
                prio = 0 if ot.order_type == "urgent" else 1
                yield disp_store.put(simpy.PriorityItem(prio, order_id))

    def dispatch_worker(_wid: int):
        nonlocal completed

        while True:
            if policy_name == "fifo":
                order_id = yield disp_store.get()
            else:
                order_id = (yield disp_store.get()).item
            queue_trackers["dispatch"].dequeue(env.now)

            ot = store[order_id]
            ot.start_disp = to_timestamp(int(env.now))

            st_min = _service_minutes(order_id, "dispatch", ot.dispatch_units)
            yield env.timeout(minutes_to_seconds_int(st_min))

            ot.end_disp = to_timestamp(int(env.now))
            completed += 1

    env.process(arrivals())

    for i in range(int(resources_cfg["picking_workers"])):
        env.process(picking_worker(i))

    for i in range(int(resources_cfg["packing_workers"])):
        env.process(packing_worker(i))

    for i in range(int(resources_cfg["dispatch_workers"])):
        env.process(dispatch_worker(i))

    # Run for exactly the finite operating horizon — never wait on done_event. Orders still
    # queued/in-service when the horizon ends are backlog, not a deadlock: the `horizon_seconds`
    # Timeout event scheduled by env.run(until=...) always keeps the event heap non-empty, so
    # this always returns with env.now == horizon_seconds regardless of how many orders remain.
    env.run(until=horizon_seconds)

    rows = []
    timing_rows = []
    for ot in store.values():
        system_time = (
            (ot.end_disp - ot.arrival_time).total_seconds() / 60.0
            if ot.end_disp else np.nan
        )
        met_sla = bool(system_time <= ot.sla_minutes)
        unfinished = ot.end_disp is None
        rows.append({
            "order_id": ot.order_id,
            "arrival_time": ot.arrival_time,
            "order_type": ot.order_type,
            "sla_minutes": ot.sla_minutes,
            "num_items": ot.num_items,
            "product_class": ot.product_class,
            "scenario": ot.scenario,
            "policy": policy_name,
            "system_time_min": system_time,
            "met_sla": met_sla,
            "unfinished": unfinished,
        })
        timing_rows.append({
            "arrival_time": ot.arrival_time,
            "start_pick": ot.start_pick, "end_pick": ot.end_pick,
            "start_pack": ot.start_pack, "end_pack": ot.end_pack,
            "start_disp": ot.start_disp, "end_disp": ot.end_disp,
            "met_sla": met_sla,
        })

    df = pd.DataFrame(rows).sort_values("order_id").reset_index(drop=True)

    timing_df = pd.DataFrame(timing_rows)
    stage_metrics = compute_stage_metrics(
        timing_df,
        workers={
            "picking":  int(resources_cfg["picking_workers"]),
            "packing":  int(resources_cfg["packing_workers"]),
            "dispatch": int(resources_cfg["dispatch_workers"]),
        },
        horizon_seconds=horizon_seconds,
        queue_trackers=queue_trackers,
    )

    unfinished_df = df[df["unfinished"]]
    total_n = len(df)
    unfinished_n = len(unfinished_df)

    summary = {
        "policy": policy_name,
        "seed": seed,
        "num_orders": len(df),
        "sla_rate": float(df["met_sla"].mean()),
        "sla_urgent": float(df.loc[df["order_type"] == "urgent", "met_sla"].mean()),
        "sla_normal": float(df.loc[df["order_type"] == "normal", "met_sla"].mean()),
        "mean_system_min": float(df["system_time_min"].mean()),
        "p90_system_min": float(df["system_time_min"].quantile(0.9)),
        "stage_metrics": stage_metrics,
        # Monthly capacity backlog (spec §9) — unfinished orders are already counted as SLA
        # failures above (system_time_min is NaN, so met_sla is False), reported separately
        # here so backlog is visible rather than hidden inside the aggregate SLA rate.
        "completed_orders": int(total_n - unfinished_n),
        "unfinished_orders": int(unfinished_n),
        "unfinished_urgent_orders": int((unfinished_df["order_type"] == "urgent").sum()),
        "unfinished_normal_orders": int((unfinished_df["order_type"] == "normal").sum()),
        "backlog_share": float(unfinished_n / total_n) if total_n > 0 else 0.0,
    }

    return df, summary
