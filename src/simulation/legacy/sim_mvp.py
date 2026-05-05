from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import simpy
import numpy as np
import pandas as pd

from src.simulation.legacy.service_times import sample_service_time_minutes


@dataclass
class OrderResult:
    order_id: int
    arrival_time: pd.Timestamp
    start_service_time: pd.Timestamp
    end_service_time: pd.Timestamp
    waiting_time_min: float
    service_time_min: float
    system_time_min: float
    order_type: str
    sla_minutes: int
    met_sla: bool
    num_items: int
    product_class: str
    scenario: str


def run_simulation_mvp(
    orders: pd.DataFrame,
    sim_cfg: Dict,
    service_cfg: Dict,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulación MVP con 1 etapa (1 recurso con capacidad N).

    Nota de robustez: el reloj del entorno (env.now) se mide en SEGUNDOS ENTEROS
    para evitar errores de coma flotante al comparar timestamps.
    """

    workers = int(sim_cfg["workers"])
    seed = int(sim_cfg["random_seed"])

    rng = np.random.default_rng(seed)

    orders = orders.sort_values("arrival_time").reset_index(drop=True)

    # Base de tiempo: mínimo arrival_time
    t0 = pd.to_datetime(orders["arrival_time"].min())

    def to_seconds(ts: pd.Timestamp) -> int:
        return int((ts - t0).total_seconds())

    def to_timestamp(seconds: int) -> pd.Timestamp:
        return t0 + pd.to_timedelta(int(seconds), unit="s")

    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=workers)

    results: List[OrderResult] = []

    # Métricas online
    in_system = 0
    max_backlog = 0
    busy_time_total_min = 0.0  # suma de service times en minutos

    def process_order(row: pd.Series):
        nonlocal in_system, max_backlog, busy_time_total_min

        order_id = int(row["order_id"])
        arrival_ts = pd.to_datetime(row["arrival_time"])
        arrival_sec = to_seconds(arrival_ts)

        # Espera hasta la llegada
        yield env.timeout(max(arrival_sec - int(env.now), 0))

        # Entra al sistema
        in_system += 1
        if in_system > max_backlog:
            max_backlog = in_system

        with resource.request() as req:
            yield req
            start_sec = int(env.now)

            # Service time (min float) -> segundos int
            st_min = sample_service_time_minutes(
                rng=rng,
                num_items=int(row["num_items"]),
                product_class=str(row["product_class"]),
                cfg=service_cfg,
            )
            st_sec = max(1, int(round(float(st_min) * 60.0)))

            busy_time_total_min += float(st_sec) / 60.0

            yield env.timeout(st_sec)
            end_sec = int(env.now)

        start_ts = to_timestamp(start_sec)
        end_ts = to_timestamp(end_sec)

        waiting_min = max(0.0, (start_sec - arrival_sec) / 60.0)
        system_min = max(0.0, (end_sec - arrival_sec) / 60.0)
        service_min = float(st_sec) / 60.0

        sla = int(row["sla_minutes"])
        met = system_min <= sla

        results.append(
            OrderResult(
                order_id=order_id,
                arrival_time=arrival_ts,
                start_service_time=start_ts,
                end_service_time=end_ts,
                waiting_time_min=waiting_min,
                service_time_min=service_min,
                system_time_min=system_min,
                order_type=str(row["order_type"]),
                sla_minutes=sla,
                met_sla=bool(met),
                num_items=int(row["num_items"]),
                product_class=str(row["product_class"]),
                scenario=str(row["scenario"]),
            )
        )

        # Sale del sistema
        in_system -= 1

    for _, row in orders.iterrows():
        env.process(process_order(row))

    env.run()

    df = pd.DataFrame([r.__dict__ for r in results]).sort_values("order_id").reset_index(drop=True)

    # Makespan en minutos (desde primer arrival hasta último end)
    start_ts = pd.to_datetime(df["arrival_time"].min())
    end_ts = pd.to_datetime(df["end_service_time"].max())
    makespan_min = (end_ts - start_ts).total_seconds() / 60.0
    makespan_min = max(makespan_min, 1e-9)

    utilization_est = float(busy_time_total_min) / (workers * makespan_min)

    summary = {
        "workers": workers,
        "seed": seed,
        "num_orders": len(df),
        "sla_rate": float(df["met_sla"].mean()),
        "mean_wait_min": float(df["waiting_time_min"].mean()),
        "p90_wait_min": float(df["waiting_time_min"].quantile(0.9)),
        "mean_system_min": float(df["system_time_min"].mean()),
        "p90_system_min": float(df["system_time_min"].quantile(0.9)),
        "makespan_min": float(makespan_min),
        "busy_time_total_min": float(busy_time_total_min),
        "utilization_est": float(utilization_est),
        "max_backlog": int(max_backlog),
    }

    return df, summary
