"""
Export a per-minute simulation timeline for animation / GIF generation.

Regime: s211 (2 Picking, 1 Packing, 1 Dispatch)
Policy: set the POLICY variable below — "fifo" or "urgent_first"
Input:  data/orders_base.csv
Output: data/animation_timeline_s211_fifo.csv          (POLICY = "fifo")
        data/animation_timeline_s211_urgent_first.csv  (POLICY = "urgent_first")

The script is fully standalone and does NOT affect the main project pipeline.
It can be deleted without consequence.

Usage:
    python -m graphic_design.export_animation_timeline
    # or from the project root:
    python graphic_design/export_animation_timeline.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import simpy
import yaml

from src.simulation.multistage.service_times_multistage import (
    sample_service_minutes,
    minutes_to_seconds_int,
)

ROOT = Path(__file__).resolve().parents[1]

# ── configuration ────────────────────────────────────────────────────────────────

N_ORDERS = 2000           # number of orders to simulate (reduce for faster / shorter anim)
SNAPSHOT_EVERY_MIN = 1   # one snapshot row per this many simulated minutes
SEED = 123
PICKING_WORKERS = 2      # s211
PACKING_WORKERS = 1
DISPATCH_WORKERS = 1
POLICY = "urgent_first"  # "fifo" | "urgent_first"


# ── data record ──────────────────────────────────────────────────────────────────

class _Rec:
    __slots__ = ("order_id", "order_type", "num_items", "product_class")

    def __init__(self, order_id: int, order_type: str, num_items: int, product_class: str):
        self.order_id = order_id
        self.order_type = order_type
        self.num_items = num_items
        self.product_class = product_class


# ── simulation ───────────────────────────────────────────────────────────────────

def run_timeline(orders: pd.DataFrame, service_cfg: dict, policy: str) -> pd.DataFrame:
    fifo = policy == "fifo"
    rng = np.random.default_rng(SEED)
    orders = orders.sort_values("arrival_time").reset_index(drop=True)
    t0 = pd.to_datetime(orders["arrival_time"].min())

    def to_sec(ts: pd.Timestamp) -> int:
        return int((ts - t0).total_seconds())

    env = simpy.Environment()

    # FIFO: one store per stage.  urgent_first: two stores per stage (urgent / normal).
    if fifo:
        pick_store = simpy.Store(env)
        pack_store = simpy.Store(env)
        disp_store = simpy.Store(env)
        pick_u = pick_n = pack_u = pack_n = disp_u = disp_n = None
    else:
        pick_u = simpy.Store(env); pick_n = simpy.Store(env)
        pack_u = simpy.Store(env); pack_n = simpy.Store(env)
        disp_u = simpy.Store(env); disp_n = simpy.Store(env)
        pick_store = pack_store = disp_store = None

    recs: dict[int, _Rec] = {}
    wip = {"pick": 0, "pack": 0, "disp": 0}
    in_system: set[int] = set()
    completed_u = 0
    completed_n = 0
    total_orders = len(orders)
    done_event = simpy.Event(env)
    snapshots: list[dict] = []

    # ── arrivals ──────────────────────────────────────────────────────────────────

    def arrivals():
        for _, row in orders.iterrows():
            oid = int(row["order_id"])
            ats = pd.to_datetime(row["arrival_time"])
            delay = max(to_sec(ats) - int(env.now), 0)
            yield env.timeout(delay)
            recs[oid] = _Rec(oid, str(row["order_type"]), int(row["num_items"]),
                             str(row["product_class"]))
            in_system.add(oid)
            if fifo:
                yield pick_store.put(oid)
            elif recs[oid].order_type == "urgent":
                yield pick_u.put(oid)
            else:
                yield pick_n.put(oid)

    # ── workers ───────────────────────────────────────────────────────────────────

    def _service(stage: str, r: _Rec) -> int:
        return minutes_to_seconds_int(
            sample_service_minutes(rng, r.num_items, r.product_class, service_cfg[stage])
        )

    def picking_worker():
        while True:
            if fifo:
                oid = yield pick_store.get()
            else:
                while len(pick_u.items) == 0 and len(pick_n.items) == 0:
                    yield env.timeout(1)
                oid = yield (pick_u.get() if len(pick_u.items) > 0 else pick_n.get())
            wip["pick"] += 1
            r = recs[oid]
            yield env.timeout(_service("picking", r))
            wip["pick"] -= 1
            if fifo:
                yield pack_store.put(oid)
            elif r.order_type == "urgent":
                yield pack_u.put(oid)
            else:
                yield pack_n.put(oid)

    def packing_worker():
        while True:
            if fifo:
                oid = yield pack_store.get()
            else:
                while len(pack_u.items) == 0 and len(pack_n.items) == 0:
                    yield env.timeout(1)
                oid = yield (pack_u.get() if len(pack_u.items) > 0 else pack_n.get())
            wip["pack"] += 1
            r = recs[oid]
            yield env.timeout(_service("packing", r))
            wip["pack"] -= 1
            if fifo:
                yield disp_store.put(oid)
            elif r.order_type == "urgent":
                yield disp_u.put(oid)
            else:
                yield disp_n.put(oid)

    def dispatch_worker():
        nonlocal completed_u, completed_n
        while True:
            if fifo:
                oid = yield disp_store.get()
            else:
                while len(disp_u.items) == 0 and len(disp_n.items) == 0:
                    yield env.timeout(1)
                oid = yield (disp_u.get() if len(disp_u.items) > 0 else disp_n.get())
            wip["disp"] += 1
            r = recs[oid]
            yield env.timeout(_service("dispatch", r))
            wip["disp"] -= 1
            in_system.discard(oid)
            if r.order_type == "urgent":
                completed_u += 1
            else:
                completed_n += 1
            if (completed_u + completed_n) >= total_orders and not done_event.triggered:
                done_event.succeed()

    # ── monitor ───────────────────────────────────────────────────────────────────

    def _snapshot():
        if fifo:
            def _uf(s: simpy.Store) -> int:
                return sum(1 for oid in s.items if recs.get(oid) and recs[oid].order_type == "urgent")
            pu = _uf(pick_store); pn = len(pick_store.items) - pu
            ku = _uf(pack_store); kn = len(pack_store.items) - ku
            du = _uf(disp_store); dn = len(disp_store.items) - du
        else:
            pu, pn = len(pick_u.items), len(pick_n.items)
            ku, kn = len(pack_u.items), len(pack_n.items)
            du, dn = len(disp_u.items), len(disp_n.items)

        in_u = sum(1 for oid in in_system if recs.get(oid) and recs[oid].order_type == "urgent")
        in_n = len(in_system) - in_u
        snapshots.append({
            "time_min": env.now / 60.0,
            "pick_queue_urgent": pu,
            "pick_queue_normal": pn,
            "pack_queue_urgent": ku,
            "pack_queue_normal": kn,
            "dispatch_queue_urgent": du,
            "dispatch_queue_normal": dn,
            "picking_busy": wip["pick"],
            "packing_busy": wip["pack"],
            "dispatch_busy": wip["disp"],
            "completed_total": completed_u + completed_n,
            "completed_urgent": completed_u,
            "completed_normal": completed_n,
            "in_system_total": len(in_system),
            "in_system_urgent": in_u,
            "in_system_normal": in_n,
        })

    def monitor():
        interval_s = int(SNAPSHOT_EVERY_MIN * 60)
        while True:
            _snapshot()
            yield env.timeout(interval_s)

    # ── run ───────────────────────────────────────────────────────────────────────

    env.process(arrivals())
    for _ in range(PICKING_WORKERS):
        env.process(picking_worker())
    for _ in range(PACKING_WORKERS):
        env.process(packing_worker())
    for _ in range(DISPATCH_WORKERS):
        env.process(dispatch_worker())
    env.process(monitor())

    env.run(until=done_event)

    # final snapshot at exact end time
    _snapshot()

    return pd.DataFrame(snapshots)


# ── entry point ───────────────────────────────────────────────────────────────────

def main() -> None:
    if POLICY not in ("fifo", "urgent_first"):
        raise ValueError(
            f"POLICY must be 'fifo' or 'urgent_first', got: {POLICY!r}\n"
            "Edit the POLICY variable at the top of this file."
        )

    output_path = ROOT / "data" / f"animation_timeline_s{PICKING_WORKERS}{PACKING_WORKERS}{DISPATCH_WORKERS}_{POLICY}.csv"

    with open(ROOT / "configs" / "sim_multistage.yaml", encoding="utf-8") as f:
        service_cfg = yaml.safe_load(f)["service_time"]

    orders_path = ROOT / "data" / "orders_base.csv"
    if not orders_path.exists():
        print(f"ERROR: {orders_path} not found.")
        print("Run first:  python -m src.data_generation.main_data_generation")
        return

    orders = (
        pd.read_csv(orders_path, parse_dates=["arrival_time"])
        .sort_values("arrival_time")
        .reset_index(drop=True)
        .iloc[:N_ORDERS]
        .copy()
    )

    urgent_count = int((orders["order_type"] == "urgent").sum())
    normal_count = len(orders) - urgent_count
    print(f"Regime    : s{PICKING_WORKERS}{PACKING_WORKERS}{DISPATCH_WORKERS}  "
          f"({PICKING_WORKERS} pick / {PACKING_WORKERS} pack / {DISPATCH_WORKERS} disp)")
    print(f"Policy    : {POLICY}")
    print(f"Orders    : {len(orders):,}  (urgent={urgent_count}, normal={normal_count})")
    print(f"Snapshot  : every {SNAPSHOT_EVERY_MIN} min\n")
    print("Running simulation...")

    df = run_timeline(orders, service_cfg, policy=POLICY)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    total_min = df["time_min"].max()
    print(f"Done.  {len(df)} snapshots  |  sim duration: {total_min:.0f} min  ({total_min/60:.1f} h)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()