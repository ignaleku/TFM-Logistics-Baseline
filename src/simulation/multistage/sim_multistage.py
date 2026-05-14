from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import simpy
import numpy as np
import pandas as pd

from src.simulation.multistage.service_times_multistage import (
    sample_service_minutes,
    minutes_to_seconds_int,
)


@dataclass
class OrderTimes:
    order_id: int
    arrival_time: pd.Timestamp
    order_type: str
    sla_minutes: int
    num_items: int
    product_class: str
    scenario: str

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
) -> Tuple[pd.DataFrame, Dict]:

    if orders is None or len(orders) == 0:
        raise ValueError("run_simulation_multistage: orders DataFrame is empty.")

    seed = int(sim_cfg["random_seed"])
    policy_name = sim_cfg.get("policy", "fifo").lower()
    rng = np.random.default_rng(seed)

    orders = orders.sort_values("arrival_time").reset_index(drop=True)
    t0 = pd.to_datetime(orders["arrival_time"].min())

    def to_seconds(ts: pd.Timestamp) -> int:
        return int((ts - t0).total_seconds())

    def to_timestamp(sec: int) -> pd.Timestamp:
        return t0 + pd.to_timedelta(int(sec), unit="s")

    env = simpy.Environment()

    # ===============================
    # Stores según política
    # ===============================
    if policy_name == "fifo":
        pick_store = simpy.Store(env)
        pack_store = simpy.Store(env)
        disp_store = simpy.Store(env)
    else:
        # urgent_first: single PriorityStore per stage.
        # priority 0 = urgent, 1 = normal.
        # Workers call one yield .get() which blocks when empty and wakes on
        # ANY arrival — fixes the deadlock where a worker blocked on the normal
        # queue while urgent items accumulated and no events remained.
        pick_store = simpy.PriorityStore(env)
        pack_store = simpy.PriorityStore(env)
        disp_store = simpy.PriorityStore(env)

    # ===============================
    # Registro y parada
    # ===============================
    store: Dict[int, OrderTimes] = {}
    completed = 0
    total_orders = len(orders)
    done_event = simpy.Event(env)

    # ===============================
    # ARRIVALS
    # ===============================
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
            )

            yield env.timeout(max(arrival_sec - int(env.now), 0))

            if policy_name == "fifo":
                yield pick_store.put(order_id)
            else:
                prio = 0 if row["order_type"] == "urgent" else 1
                yield pick_store.put(simpy.PriorityItem(prio, order_id))

    # ===============================
    # WORKERS
    # ===============================
    def picking_worker(_wid: int):
        while True:
            if policy_name == "fifo":
                order_id = yield pick_store.get()
            else:
                order_id = (yield pick_store.get()).item

            ot = store[order_id]
            ot.start_pick = to_timestamp(int(env.now))

            st_min = sample_service_minutes(
                rng, ot.num_items, ot.product_class, service_cfg["picking"]
            )
            yield env.timeout(minutes_to_seconds_int(st_min))

            ot.end_pick = to_timestamp(int(env.now))

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

            ot = store[order_id]
            ot.start_pack = to_timestamp(int(env.now))

            st_min = sample_service_minutes(
                rng, ot.num_items, ot.product_class, service_cfg["packing"]
            )
            yield env.timeout(minutes_to_seconds_int(st_min))

            ot.end_pack = to_timestamp(int(env.now))

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

            ot = store[order_id]
            ot.start_disp = to_timestamp(int(env.now))

            st_min = sample_service_minutes(
                rng, ot.num_items, ot.product_class, service_cfg["dispatch"]
            )
            yield env.timeout(minutes_to_seconds_int(st_min))

            ot.end_disp = to_timestamp(int(env.now))

            completed += 1
            if completed >= total_orders and not done_event.triggered:
                done_event.succeed()

    # ===============================
    # Lanzar procesos
    # ===============================
    env.process(arrivals())

    for i in range(int(resources_cfg["picking_workers"])):
        env.process(picking_worker(i))

    for i in range(int(resources_cfg["packing_workers"])):
        env.process(packing_worker(i))

    for i in range(int(resources_cfg["dispatch_workers"])):
        env.process(dispatch_worker(i))

    try:
        env.run(until=done_event)
    except RuntimeError as exc:
        raise RuntimeError(
            f"SimPy deadlock: done_event never triggered. "
            f"total_orders={total_orders} completed={completed} "
            f"pick_q={len(pick_store.items)} "
            f"pack_q={len(pack_store.items)} "
            f"disp_q={len(disp_store.items)} "
            f"policy={policy_name} "
            f"workers=({resources_cfg['picking_workers']},"
            f"{resources_cfg['packing_workers']},"
            f"{resources_cfg['dispatch_workers']})"
        ) from exc

    # ===============================
    # Resultados
    # ===============================
    rows = []

    for ot in store.values():
        system_time = (
            (ot.end_disp - ot.arrival_time).total_seconds() / 60.0
            if ot.end_disp else np.nan
        )

        rows.append(
            {
                "order_id": ot.order_id,
                "arrival_time": ot.arrival_time,
                "order_type": ot.order_type,
                "sla_minutes": ot.sla_minutes,
                "num_items": ot.num_items,
                "product_class": ot.product_class,
                "scenario": ot.scenario,
                "policy": policy_name,
                "system_time_min": system_time,
                "met_sla": system_time <= ot.sla_minutes,
            }
        )

    df = pd.DataFrame(rows).sort_values("order_id").reset_index(drop=True)

    summary = {
        "policy": policy_name,
        "seed": seed,
        "num_orders": len(df),
        "sla_rate": float(df["met_sla"].mean()),
        "sla_urgent": float(df.loc[df["order_type"] == "urgent", "met_sla"].mean()),
        "sla_normal": float(df.loc[df["order_type"] == "normal", "met_sla"].mean()),
        "mean_system_min": float(df["system_time_min"].mean()),
        "p90_system_min": float(df["system_time_min"].quantile(0.9)),
    }

    return df, summary
