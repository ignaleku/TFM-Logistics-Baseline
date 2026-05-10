# src/rl/env_5stage_rl.py
"""
RL-5 environment: single shared DQN agent acting at all five stages.

Stage order : Picking → Quality Check → Packing → Labelling → Dispatch
Stage IDs   : 0.00       0.25            0.50       0.75        1.00

State vector (19 features, input_dim=19):
  Features  0– 9 : queue lengths (urgent/normal) at each of the 5 stages, normalised.
  Features 10–14 : WIP per stage (orders currently in service), normalised.
  Feature   15   : time_norm.
  Features 16–17 : slack of head urgent / normal order at the current stage.
  Feature   18   : stage_id.

Action   : 0 = select urgent order, 1 = select normal order.
           The agent only acts when both queues are non-empty;
           otherwise the non-empty queue is served automatically (no RL transition stored).
Reward   : deferred to Dispatch completion — same design as RL-3.
           All RL transitions for a given order receive the same final reward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import simpy
import numpy as np
import pandas as pd

from src.simulation.multistage.service_times_multistage import (
    sample_service_minutes,
    minutes_to_seconds_int,
)
from src.rl.replay_buffer import ReplayBuffer, Transition


# ── Stage ID constants ────────────────────────────────────────────────────────

STAGE_PICK = 0.00
STAGE_QC   = 0.25
STAGE_PACK = 0.50
STAGE_LAB  = 0.75
STAGE_DISP = 1.00


# ── Order record ──────────────────────────────────────────────────────────────

@dataclass
class OrderRec5:
    order_id:      int
    arrival_time:  pd.Timestamp
    order_type:    str
    sla_minutes:   int
    num_items:     int
    product_class: str
    scenario:      str
    end_disp:      Optional[pd.Timestamp] = None


# ── Runner ────────────────────────────────────────────────────────────────────

class FiveStageRLRunner:
    """
    RL-5: a single DQN agent decides at Picking, Quality Check, Packing,
    Labelling, and Dispatch.

    Key design choices mirror RL-3 (FullStageRLRunner):
    - Separate urgent/normal queues at every stage.
    - stage_id encodes which stage the agent is deciding at (0.0 → 1.0 linear).
    - Reward deferred to Dispatch; all decisions for an order share the same reward.
    - Buffer transitions stored only at real decision points (both queues non-empty).
    """

    def __init__(
        self,
        sim_cfg:       Dict,
        resources_cfg: Dict,
        service_cfg:   Dict,
        seed:          int,
        reward_cfg:    Optional[Dict] = None,
    ) -> None:
        self.sim_cfg       = sim_cfg
        self.resources_cfg = resources_cfg
        self.service_cfg   = service_cfg
        self.rng           = np.random.default_rng(int(seed))
        self.reward_cfg    = reward_cfg or {
            "w_urgent":             5.0,
            "w_normal":             2.0,
            "late_penalty_urgent":  2.0,
            "late_penalty_normal":  1.0,
            "lateness_cap_mult":    2.0,
        }

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, x)))

    # ── Slack helpers ─────────────────────────────────────────────────────────

    def _slack_norm(self, now_ts: pd.Timestamp, r: OrderRec5) -> float:
        system_min = (now_ts - r.arrival_time).total_seconds() / 60.0
        slack      = float(r.sla_minutes - system_min)
        denom      = max(1.0, float(r.sla_minutes))
        x          = self._clip(slack / denom, -1.0, 1.0)
        return (x + 1.0) / 2.0

    def _head_slack(
        self, now_ts: pd.Timestamp, store: simpy.Store, recs: Dict
    ) -> float:
        if len(store.items) > 0:
            return self._slack_norm(now_ts, recs[store.items[0]])
        return 0.5

    # ── State builder ─────────────────────────────────────────────────────────

    def _state(
        self,
        now_s:    int,
        now_ts:   pd.Timestamp,
        pick_u:   simpy.Store,
        pick_n:   simpy.Store,
        qc_u:     simpy.Store,
        qc_n:     simpy.Store,
        pack_u:   simpy.Store,
        pack_n:   simpy.Store,
        lab_u:    simpy.Store,
        lab_n:    simpy.Store,
        disp_u:   simpy.Store,
        disp_n:   simpy.Store,
        wip_pick: int,
        wip_qc:   int,
        wip_pack: int,
        wip_lab:  int,
        wip_disp: int,
        recs:     Dict,
        horizon_s: int,
        stage_id: float,
        curr_u:   simpy.Store,
        curr_n:   simpy.Store,
    ) -> np.ndarray:
        time_norm = self._clip01(now_s / max(1, horizon_s))

        pu_n = self._clip01(len(pick_u.items) / 200.0)
        pn_n = self._clip01(len(pick_n.items) / 500.0)
        qu_n = self._clip01(len(qc_u.items)   / 200.0)
        qn_n = self._clip01(len(qc_n.items)   / 500.0)
        ku_n = self._clip01(len(pack_u.items)  / 200.0)
        kn_n = self._clip01(len(pack_n.items)  / 500.0)
        lu_n = self._clip01(len(lab_u.items)   / 200.0)
        ln_n = self._clip01(len(lab_n.items)   / 500.0)
        du_n = self._clip01(len(disp_u.items)  / 200.0)
        dn_n = self._clip01(len(disp_n.items)  / 500.0)

        wip_pick_n = self._clip01(wip_pick / 5.0)
        wip_qc_n   = self._clip01(wip_qc   / 5.0)
        wip_pack_n = self._clip01(wip_pack  / 5.0)
        wip_lab_n  = self._clip01(wip_lab   / 5.0)
        wip_disp_n = self._clip01(wip_disp  / 5.0)

        slack_u = self._head_slack(now_ts, curr_u, recs)
        slack_n = self._head_slack(now_ts, curr_n, recs)

        return np.array(
            [
                pu_n, pn_n,
                qu_n, qn_n,
                ku_n, kn_n,
                lu_n, ln_n,
                du_n, dn_n,
                wip_pick_n, wip_qc_n, wip_pack_n, wip_lab_n, wip_disp_n,
                time_norm,
                slack_u, slack_n,
                stage_id,
            ],
            dtype=np.float32,
        )

    # ── Reward ────────────────────────────────────────────────────────────────

    def _reward(self, r: OrderRec5) -> float:
        assert r.end_disp is not None
        system_min = (r.end_disp - r.arrival_time).total_seconds() / 60.0
        lateness   = float(system_min - r.sla_minutes)
        is_urgent  = r.order_type == "urgent"
        on_time    = lateness <= 0.0

        mode = self.reward_cfg.get("reward_mode", "rl1_current")
        if mode == "urgent_protection":
            if is_urgent:
                return 10.0 if on_time else -5.0
            return 1.0 if on_time else 0.0

        # rl1_current: continuous reward proportional to lateness
        w = float(self.reward_cfg["w_urgent"] if is_urgent else self.reward_cfg["w_normal"])
        p = float(
            self.reward_cfg["late_penalty_urgent"]
            if is_urgent
            else self.reward_cfg["late_penalty_normal"]
        )
        if on_time:
            return w
        cap_mult = float(self.reward_cfg.get("lateness_cap_mult", 2.0))
        denom    = max(1.0, float(r.sla_minutes) * cap_mult)
        frac     = min(1.0, lateness / denom)
        return -p * frac

    # ── Episode ───────────────────────────────────────────────────────────────

    def run_episode(
        self,
        orders:       pd.DataFrame,
        agent,
        buffer:       ReplayBuffer,
        episode_seed: int,
        greedy:       bool = False,
    ) -> Dict:
        self.rng = np.random.default_rng(int(episode_seed))
        orders   = orders.sort_values("arrival_time").reset_index(drop=True)
        t0       = pd.to_datetime(orders["arrival_time"].min())

        def to_seconds(ts: pd.Timestamp) -> int:
            return int((ts - t0).total_seconds())

        def to_timestamp(sec: int) -> pd.Timestamp:
            return t0 + pd.to_timedelta(int(sec), unit="s")

        env = simpy.Environment()

        # Urgent and normal queues at each of the five stages
        pick_u = simpy.Store(env)
        pick_n = simpy.Store(env)
        qc_u   = simpy.Store(env)
        qc_n   = simpy.Store(env)
        pack_u = simpy.Store(env)
        pack_n = simpy.Store(env)
        lab_u  = simpy.Store(env)
        lab_n  = simpy.Store(env)
        disp_u = simpy.Store(env)
        disp_n = simpy.Store(env)

        recs:             Dict[int, OrderRec5]       = {}
        total_orders:     int                        = len(orders)
        completed:        int                        = 0
        done_event        = simpy.Event(env)
        decision_indices: Dict[int, List[int]]       = {}

        horizon_s = max(1, to_seconds(pd.to_datetime(orders["arrival_time"].max())))

        wip = {"pick": 0, "qc": 0, "pack": 0, "lab": 0, "disp": 0}

        sm = {
            "pick": {"dec_pts": 0, "dec_u": 0, "dec_n": 0},
            "qc":   {"dec_pts": 0, "dec_u": 0, "dec_n": 0},
            "pack": {"dec_pts": 0, "dec_u": 0, "dec_n": 0},
            "lab":  {"dec_pts": 0, "dec_u": 0, "dec_n": 0},
            "disp": {"dec_pts": 0, "dec_u": 0, "dec_n": 0},
        }

        # ── inner helpers ─────────────────────────────────────────────────────

        def _build_state(
            stage_id: float,
            curr_u: simpy.Store,
            curr_n: simpy.Store,
            now_s: int,
            now_ts: pd.Timestamp,
        ) -> np.ndarray:
            return self._state(
                now_s, now_ts,
                pick_u, pick_n, qc_u, qc_n, pack_u, pack_n, lab_u, lab_n, disp_u, disp_n,
                wip["pick"], wip["qc"], wip["pack"], wip["lab"], wip["disp"],
                recs, horizon_s, stage_id, curr_u, curr_n,
            )

        def _decide(
            stage_name: str,
            stage_id:   float,
            store_u:    simpy.Store,
            store_n:    simpy.Store,
            now_s:      int,
            now_ts:     pd.Timestamp,
        ):
            """Compute state, invoke agent if both queues non-empty, update metrics.
            Returns (action, state_pre, should_record)."""
            q_u   = len(store_u.items)
            q_n   = len(store_n.items)
            s     = _build_state(stage_id, store_u, store_n, now_s, now_ts)
            stage = sm[stage_name]

            if q_u > 0 and q_n > 0:
                stage["dec_pts"] += 1
                a             = int(agent.act(s, greedy=greedy))
                should_record = not greedy
                if a == 0:
                    stage["dec_u"] += 1
                else:
                    stage["dec_n"] += 1
            elif q_u > 0:
                a, should_record = 0, False
            else:
                a, should_record = 1, False

            # safety fallback: queue may have been drained by another worker
            if a == 0 and q_u == 0:
                a = 1
            if a == 1 and q_n == 0:
                a = 0

            return a, s, should_record

        def _record_transition(
            oid:      int,
            s_pre:    np.ndarray,
            a:        int,
            stage_id: float,
            store_u:  simpy.Store,
            store_n:  simpy.Store,
            now_s2:   int,
            now_ts2:  pd.Timestamp,
        ) -> None:
            s_post = _build_state(stage_id, store_u, store_n, now_s2, now_ts2)
            idx    = buffer.add(
                Transition(state=s_pre, action=a, next_state=s_post, reward=0.0, done=False)
            )
            if oid not in decision_indices:
                decision_indices[oid] = []
            decision_indices[oid].append(idx)

        # ── arrivals ──────────────────────────────────────────────────────────

        def arrivals():
            for _, row in orders.iterrows():
                oid = int(row["order_id"])
                ats = pd.to_datetime(row["arrival_time"])
                yield env.timeout(max(to_seconds(ats) - int(env.now), 0))
                recs[oid] = OrderRec5(
                    order_id      = oid,
                    arrival_time  = ats,
                    order_type    = str(row["order_type"]),
                    sla_minutes   = int(row["sla_minutes"]),
                    num_items     = int(row["num_items"]),
                    product_class = str(row["product_class"]),
                    scenario      = str(row["scenario"]),
                )
                if row["order_type"] == "urgent":
                    yield pick_u.put(oid)
                else:
                    yield pick_n.put(oid)

        # ── stage workers ─────────────────────────────────────────────────────

        def picking_worker(wid: int):
            while True:
                while len(pick_u.items) == 0 and len(pick_n.items) == 0:
                    yield env.timeout(1)

                now_s  = int(env.now)
                now_ts = to_timestamp(now_s)
                a, s_pre, should_record = _decide(
                    "pick", STAGE_PICK, pick_u, pick_n, now_s, now_ts
                )

                oid = yield (pick_u.get() if a == 0 else pick_n.get())
                wip["pick"] += 1

                if should_record:
                    now_s2 = int(env.now)
                    _record_transition(
                        oid, s_pre, a, STAGE_PICK, pick_u, pick_n,
                        now_s2, to_timestamp(now_s2),
                    )

                r  = recs[oid]
                st = sample_service_minutes(
                    self.rng, r.num_items, r.product_class, self.service_cfg["picking"]
                )
                yield env.timeout(minutes_to_seconds_int(st))
                wip["pick"] -= 1

                if r.order_type == "urgent":
                    yield qc_u.put(oid)
                else:
                    yield qc_n.put(oid)

        def qc_worker(wid: int):
            while True:
                while len(qc_u.items) == 0 and len(qc_n.items) == 0:
                    yield env.timeout(1)

                now_s  = int(env.now)
                now_ts = to_timestamp(now_s)
                a, s_pre, should_record = _decide(
                    "qc", STAGE_QC, qc_u, qc_n, now_s, now_ts
                )

                oid = yield (qc_u.get() if a == 0 else qc_n.get())
                wip["qc"] += 1

                if should_record:
                    now_s2 = int(env.now)
                    _record_transition(
                        oid, s_pre, a, STAGE_QC, qc_u, qc_n,
                        now_s2, to_timestamp(now_s2),
                    )

                r  = recs[oid]
                st = sample_service_minutes(
                    self.rng, r.num_items, r.product_class, self.service_cfg["quality_check"]
                )
                yield env.timeout(minutes_to_seconds_int(st))
                wip["qc"] -= 1

                if r.order_type == "urgent":
                    yield pack_u.put(oid)
                else:
                    yield pack_n.put(oid)

        def packing_worker(wid: int):
            while True:
                while len(pack_u.items) == 0 and len(pack_n.items) == 0:
                    yield env.timeout(1)

                now_s  = int(env.now)
                now_ts = to_timestamp(now_s)
                a, s_pre, should_record = _decide(
                    "pack", STAGE_PACK, pack_u, pack_n, now_s, now_ts
                )

                oid = yield (pack_u.get() if a == 0 else pack_n.get())
                wip["pack"] += 1

                if should_record:
                    now_s2 = int(env.now)
                    _record_transition(
                        oid, s_pre, a, STAGE_PACK, pack_u, pack_n,
                        now_s2, to_timestamp(now_s2),
                    )

                r  = recs[oid]
                st = sample_service_minutes(
                    self.rng, r.num_items, r.product_class, self.service_cfg["packing"]
                )
                yield env.timeout(minutes_to_seconds_int(st))
                wip["pack"] -= 1

                if r.order_type == "urgent":
                    yield lab_u.put(oid)
                else:
                    yield lab_n.put(oid)

        def labelling_worker(wid: int):
            while True:
                while len(lab_u.items) == 0 and len(lab_n.items) == 0:
                    yield env.timeout(1)

                now_s  = int(env.now)
                now_ts = to_timestamp(now_s)
                a, s_pre, should_record = _decide(
                    "lab", STAGE_LAB, lab_u, lab_n, now_s, now_ts
                )

                oid = yield (lab_u.get() if a == 0 else lab_n.get())
                wip["lab"] += 1

                if should_record:
                    now_s2 = int(env.now)
                    _record_transition(
                        oid, s_pre, a, STAGE_LAB, lab_u, lab_n,
                        now_s2, to_timestamp(now_s2),
                    )

                r  = recs[oid]
                st = sample_service_minutes(
                    self.rng, r.num_items, r.product_class, self.service_cfg["labelling"]
                )
                yield env.timeout(minutes_to_seconds_int(st))
                wip["lab"] -= 1

                if r.order_type == "urgent":
                    yield disp_u.put(oid)
                else:
                    yield disp_n.put(oid)

        def dispatch_worker(wid: int):
            nonlocal completed
            while True:
                while len(disp_u.items) == 0 and len(disp_n.items) == 0:
                    yield env.timeout(1)

                now_s  = int(env.now)
                now_ts = to_timestamp(now_s)
                a, s_pre, should_record = _decide(
                    "disp", STAGE_DISP, disp_u, disp_n, now_s, now_ts
                )

                oid = yield (disp_u.get() if a == 0 else disp_n.get())
                wip["disp"] += 1

                if should_record:
                    now_s2 = int(env.now)
                    _record_transition(
                        oid, s_pre, a, STAGE_DISP, disp_u, disp_n,
                        now_s2, to_timestamp(now_s2),
                    )

                r  = recs[oid]
                st = sample_service_minutes(
                    self.rng, r.num_items, r.product_class, self.service_cfg["dispatch"]
                )
                yield env.timeout(minutes_to_seconds_int(st))
                wip["disp"] -= 1

                r.end_disp = to_timestamp(int(env.now))
                completed += 1

                # All RL decisions for this order receive the same deferred reward
                reward_val = self._reward(r)
                for idx in decision_indices.get(oid, []):
                    buffer.set_reward(idx, reward_val)

                if completed >= total_orders and not done_event.triggered:
                    done_event.succeed()

        # ── launch ────────────────────────────────────────────────────────────

        env.process(arrivals())
        for i in range(int(self.resources_cfg["picking_workers"])):
            env.process(picking_worker(i))
        for i in range(int(self.resources_cfg["quality_check_workers"])):
            env.process(qc_worker(i))
        for i in range(int(self.resources_cfg["packing_workers"])):
            env.process(packing_worker(i))
        for i in range(int(self.resources_cfg["labelling_workers"])):
            env.process(labelling_worker(i))
        for i in range(int(self.resources_cfg["dispatch_workers"])):
            env.process(dispatch_worker(i))

        env.run(until=done_event)

        # ── build results ─────────────────────────────────────────────────────

        df = pd.DataFrame({
            "order_id":    [r.order_id    for r in recs.values()],
            "order_type":  [r.order_type  for r in recs.values()],
            "arrival_time":[r.arrival_time for r in recs.values()],
            "end_disp":    [r.end_disp    for r in recs.values()],
            "sla_minutes": [r.sla_minutes for r in recs.values()],
        })
        df["system_min"] = (df["end_disp"] - df["arrival_time"]).dt.total_seconds() / 60.0
        df["met_sla"]    = df["system_min"] <= df["sla_minutes"]

        sla_rate   = float(df["met_sla"].mean())
        sla_urgent = float(df.loc[df["order_type"] == "urgent", "met_sla"].mean())
        sla_normal = float(df.loc[df["order_type"] != "urgent", "met_sla"].mean())

        total_dec   = sum(s["dec_pts"] for s in sm.values())
        total_dec_u = sum(s["dec_u"]   for s in sm.values())
        p_urgent    = (total_dec_u / total_dec) if total_dec > 0 else 0.0

        def _stage_rate(s: Dict) -> float:
            return (s["dec_u"] / s["dec_pts"]) if s["dec_pts"] > 0 else 0.0

        return {
            "sla_rate":           sla_rate,
            "sla_urgent":         sla_urgent,
            "sla_normal":         sla_normal,
            "mean_system_min":    float(df["system_min"].mean()),
            "p90_system_min":     float(df["system_min"].quantile(0.9)),
            "total_decisions":    int(total_dec),
            "p_urgent_decisions": float(p_urgent),
            "pick_dec_pts":       int(sm["pick"]["dec_pts"]),
            "qc_dec_pts":         int(sm["qc"]["dec_pts"]),
            "pack_dec_pts":       int(sm["pack"]["dec_pts"]),
            "lab_dec_pts":        int(sm["lab"]["dec_pts"]),
            "disp_dec_pts":       int(sm["disp"]["dec_pts"]),
            "pick_pct_urgent":    float(_stage_rate(sm["pick"])),
            "qc_pct_urgent":      float(_stage_rate(sm["qc"])),
            "pack_pct_urgent":    float(_stage_rate(sm["pack"])),
            "lab_pct_urgent":     float(_stage_rate(sm["lab"])),
            "disp_pct_urgent":    float(_stage_rate(sm["disp"])),
        }
