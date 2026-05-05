from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import simpy
import numpy as np
import pandas as pd

from src.simulation.multistage.service_times_multistage import (
    sample_service_minutes,
    minutes_to_seconds_int,
)
from src.rl.replay_buffer import ReplayBuffer, Transition


@dataclass
class OrderRec:
    order_id: int
    arrival_time: pd.Timestamp
    order_type: str
    sla_minutes: int
    num_items: int
    product_class: str
    scenario: str
    end_disp: Optional[pd.Timestamp] = None


class PickRLRunner:
    """
    Ejecuta 1 episodio SimPy multi-etapa donde el agente decide en PICKING:
    acción 0=urgent, 1=normal (solo cuando hay ambas colas; si no, fallback determinista).

    Cambios v2:
    - Replay buffer SOLO en puntos de decisión reales (ambas colas > 0).
    - Estado enriquecido: slack de cabecera y slack del peor normal (min slack en cola normal).
    - Reward continua por tardanza (configurable con reward_cfg).
    - Métricas de comportamiento: dec/forced/%U.
    """

    def __init__(
        self,
        sim_cfg: Dict,
        resources_cfg: Dict,
        service_cfg: Dict,
        seed: int,
        reward_cfg: Optional[Dict] = None,
    ):
        self.sim_cfg = sim_cfg
        self.resources_cfg = resources_cfg
        self.service_cfg = service_cfg
        self.rng = np.random.default_rng(int(seed))

        # reward shaping (valores por defecto razonables)
        self.reward_cfg = reward_cfg or {
            "w_urgent": 5.0,
            "w_normal": 2.0,
            "late_penalty_urgent": 2.0,
            "late_penalty_normal": 1.0,
            "lateness_cap_mult": 2.0,  # cap a 2x SLA
        }

    @staticmethod
    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, x)))

    def _slack_norm(self, now_ts: pd.Timestamp, r: OrderRec) -> float:
        """
        Slack normalizado en [0,1] donde:
          1.0 = mucho margen,
          0.5 = justo en SLA,
          0.0 = muy tarde (cap).
        """
        system_min = (now_ts - r.arrival_time).total_seconds() / 60.0
        slack = float(r.sla_minutes - system_min)  # >0 si queda margen
        denom = max(1.0, float(r.sla_minutes))
        x = self._clip(slack / denom, -1.0, 1.0)   # [-1, 1]
        return (x + 1.0) / 2.0                     # -> [0, 1]

    def _state(
        self,
        now_s: int,
        now_ts: pd.Timestamp,
        pick_u: simpy.Store,
        pick_n: simpy.Store,
        pack: simpy.Store,
        disp: simpy.Store,
        recs: Dict[int, OrderRec],
        horizon_s: int,
    ) -> np.ndarray:
        # normalización simple y robusta
        time_norm = self._clip01(now_s / max(1, horizon_s))
        q_u = len(pick_u.items)
        q_n = len(pick_n.items)
        wip_pack = len(pack.items)
        wip_disp = len(disp.items)

        q_u_n = self._clip01(q_u / 200.0)
        q_n_n = self._clip01(q_n / 500.0)
        wip_pack_n = self._clip01(wip_pack / 500.0)
        wip_disp_n = self._clip01(wip_disp / 500.0)

        # slack de cabecera (si no hay, neutro 0.5)
        head_u = 0.5
        if q_u > 0:
            ru = recs[pick_u.items[0]]
            head_u = self._slack_norm(now_ts, ru)

        head_n = 0.5
        worst_n = 0.5
        if q_n > 0:
            # cabecera normal
            rn0 = recs[pick_n.items[0]]
            head_n = self._slack_norm(now_ts, rn0)

            # peor slack normal (min slack) — recortamos coste mirando hasta 50 ítems
            sample_ids = pick_n.items[: min(50, q_n)]
            worst = 1.0
            for oid in sample_ids:
                rn = recs[oid]
                worst = min(worst, self._slack_norm(now_ts, rn))
            worst_n = worst

        return np.array(
            [q_u_n, q_n_n, wip_pack_n, wip_disp_n, time_norm, head_u, head_n, worst_n],
            dtype=np.float32,
        )

    def _reward(self, r: OrderRec) -> float:
        assert r.end_disp is not None
        system_min = (r.end_disp - r.arrival_time).total_seconds() / 60.0
        lateness = float(system_min - r.sla_minutes)  # >0 if late
        is_urgent = (r.order_type == "urgent")
        on_time = lateness <= 0.0

        mode = self.reward_cfg.get("reward_mode", "rl1_current")

        if mode == "urgent_protection":
            if is_urgent:
                return 10.0 if on_time else -5.0
            else:
                return 1.0 if on_time else 0.0

        # rl1_current: continuous reward proportional to lateness
        w = float(self.reward_cfg["w_urgent"] if is_urgent else self.reward_cfg["w_normal"])
        p = float(self.reward_cfg["late_penalty_urgent"] if is_urgent else self.reward_cfg["late_penalty_normal"])
        if on_time:
            return w
        cap_mult = float(self.reward_cfg.get("lateness_cap_mult", 2.0))
        denom = max(1.0, float(r.sla_minutes) * cap_mult)
        frac = min(1.0, lateness / denom)
        return -p * frac

    def run_episode(
        self,
        orders: pd.DataFrame,
        agent,
        buffer: ReplayBuffer,
        episode_seed: int,
        greedy: bool = False,
    ) -> Dict:
        # reproducibilidad por episodio
        self.rng = np.random.default_rng(int(episode_seed))

        orders = orders.sort_values("arrival_time").reset_index(drop=True)
        t0 = pd.to_datetime(orders["arrival_time"].min())

        def to_seconds(ts: pd.Timestamp) -> int:
            return int((ts - t0).total_seconds())

        def to_timestamp(sec: int) -> pd.Timestamp:
            return t0 + pd.to_timedelta(int(sec), unit="s")

        env = simpy.Environment()

        # colas
        pick_u = simpy.Store(env)
        pick_n = simpy.Store(env)
        pack = simpy.Store(env)   # FIFO
        disp = simpy.Store(env)   # FIFO

        # registro
        recs: Dict[int, OrderRec] = {}
        total_orders = len(orders)
        completed = 0
        done_event = simpy.Event(env)

        # mapping order_id -> idx transición en buffer (reward diferido)
        decision_idx: Dict[int, int] = {}

        horizon_s = max(1, to_seconds(pd.to_datetime(orders["arrival_time"].max())))

        # métricas de decisiones
        decision_points = 0
        decision_action_urgent = 0
        decision_action_normal = 0
        forced_urgent = 0
        forced_normal = 0

        # arrivals
        def arrivals():
            for _, row in orders.iterrows():
                oid = int(row["order_id"])
                ats = pd.to_datetime(row["arrival_time"])
                yield env.timeout(max(to_seconds(ats) - int(env.now), 0))

                recs[oid] = OrderRec(
                    order_id=oid,
                    arrival_time=ats,
                    order_type=str(row["order_type"]),
                    sla_minutes=int(row["sla_minutes"]),
                    num_items=int(row["num_items"]),
                    product_class=str(row["product_class"]),
                    scenario=str(row["scenario"]),
                )

                if row["order_type"] == "urgent":
                    yield pick_u.put(oid)
                else:
                    yield pick_n.put(oid)

        # picking workers
        def picking_worker(_wid: int):
            nonlocal decision_points, decision_action_urgent, decision_action_normal, forced_urgent, forced_normal
            while True:
                # espera a que haya algo en alguna cola
                while len(pick_u.items) == 0 and len(pick_n.items) == 0:
                    yield env.timeout(1)

                now_s = int(env.now)
                now_ts = to_timestamp(now_s)

                q_u = len(pick_u.items)
                q_n = len(pick_n.items)

                # estado actual
                s = self._state(now_s, now_ts, pick_u, pick_n, pack, disp, recs, horizon_s)

                # decidir solo si hay ambos; si no, determinista (y NO guardamos transición)
                store_transition = False
                if q_u > 0 and q_n > 0:
                    decision_points += 1
                    a = int(agent.act(s, greedy=greedy))
                    store_transition = (not greedy)

                    if a == 0:
                        decision_action_urgent += 1
                    else:
                        decision_action_normal += 1
                elif q_u > 0:
                    a = 0
                    forced_urgent += 1
                else:
                    a = 1
                    forced_normal += 1

                # aplicar acción con fallback
                if a == 0 and q_u == 0:
                    a = 1
                if a == 1 and q_n == 0:
                    a = 0

                oid = yield (pick_u.get() if a == 0 else pick_n.get())

                # next_state tras sacar 1 de picking
                now_s2 = int(env.now)
                now_ts2 = to_timestamp(now_s2)
                s2 = self._state(now_s2, now_ts2, pick_u, pick_n, pack, disp, recs, horizon_s)

                # guardamos transición SOLO si hubo decisión real
                if store_transition:
                    idx = buffer.add(Transition(state=s, action=a, next_state=s2, reward=0.0, done=False))
                    decision_idx[oid] = idx

                # servicio picking
                r = recs[oid]
                st_min = sample_service_minutes(self.rng, r.num_items, r.product_class, self.service_cfg["picking"])
                yield env.timeout(minutes_to_seconds_int(st_min))

                # a packing
                yield pack.put(oid)

        # packing workers (FIFO)
        def packing_worker(_wid: int):
            while True:
                oid = yield pack.get()
                r = recs[oid]
                st_min = sample_service_minutes(self.rng, r.num_items, r.product_class, self.service_cfg["packing"])
                yield env.timeout(minutes_to_seconds_int(st_min))
                yield disp.put(oid)

        # dispatch workers (FIFO + reward diferido)
        def dispatch_worker(_wid: int):
            nonlocal completed
            while True:
                oid = yield disp.get()
                r = recs[oid]
                st_min = sample_service_minutes(self.rng, r.num_items, r.product_class, self.service_cfg["dispatch"])
                yield env.timeout(minutes_to_seconds_int(st_min))

                r.end_disp = to_timestamp(int(env.now))
                completed += 1

                idx = decision_idx.get(oid)
                if idx is not None:
                    buffer.set_reward(idx, self._reward(r))

                if completed >= total_orders and not done_event.triggered:
                    done_event.succeed()

        # lanzar procesos
        env.process(arrivals())
        for i in range(int(self.resources_cfg["picking_workers"])):
            env.process(picking_worker(i))
        for i in range(int(self.resources_cfg["packing_workers"])):
            env.process(packing_worker(i))
        for i in range(int(self.resources_cfg["dispatch_workers"])):
            env.process(dispatch_worker(i))

        env.run(until=done_event)

        # métricas episodio (SLA)
        df = pd.DataFrame(
            {
                "order_id": [r.order_id for r in recs.values()],
                "order_type": [r.order_type for r in recs.values()],
                "arrival_time": [r.arrival_time for r in recs.values()],
                "end_disp": [r.end_disp for r in recs.values()],
                "sla_minutes": [r.sla_minutes for r in recs.values()],
            }
        )
        df["system_min"] = (df["end_disp"] - df["arrival_time"]).dt.total_seconds() / 60.0
        df["met_sla"] = df["system_min"] <= df["sla_minutes"]

        sla_rate = float(df["met_sla"].mean())
        sla_urgent = float(df.loc[df["order_type"] == "urgent", "met_sla"].mean())
        sla_normal = float(df.loc[df["order_type"] != "urgent", "met_sla"].mean())

        pu = (decision_action_urgent / decision_points) if decision_points > 0 else 0.0

        return {
            "sla_rate": sla_rate,
            "sla_urgent": sla_urgent,
            "sla_normal": sla_normal,
            "mean_system_min": float(df["system_min"].mean()),
            "p90_system_min": float(df["system_min"].quantile(0.9)),
            "decision_points": int(decision_points),
            "decision_action_urgent": int(decision_action_urgent),
            "decision_action_normal": int(decision_action_normal),
            "p_urgent_decisions": float(pu),
            "forced_urgent": int(forced_urgent),
            "forced_normal": int(forced_normal),
        }
