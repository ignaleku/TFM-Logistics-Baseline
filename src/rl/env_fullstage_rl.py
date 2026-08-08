from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import simpy
import numpy as np
import pandas as pd

from src.simulation.multistage.service_times_multistage import (
    sample_service_minutes,
    minutes_to_seconds_int,
)
from src.simulation.multistage.stage_metrics import QueueAreaTracker, compute_stage_metrics
from src.simulation.multistage.operating_time import SIM_EPOCH
from src.rl.replay_buffer import ReplayBuffer, Transition
from src.data.planning_profile import load_planning_profile

# Required columns for the enriched workload model
_REQUIRED_WORKLOAD_COLS = {"picking_units", "packing_units", "dispatch_units"}

# stage_id feature values (feature index 12 in the state vector)
STAGE_PICK = 0.0
STAGE_PACK = 0.5
STAGE_DISP = 1.0


class _StageSignal:
    """Event-driven "wake me when either queue at this stage gets an item" notification.

    Replaces a `while both queues empty: yield env.timeout(1)` polling loop, which is fine
    during a short done_event-bounded run but is prohibitively expensive once the horizon is a
    fixed monthly operating window (9600 minutes = 576,000 one-second polls per idle worker for
    the tail after work drains). Multiple workers may wait on the same event and all wake when
    it fires — exactly the same race as the polling version (several workers re-check the queue
    and only as many as there are available items actually dequeue), just resolved immediately
    instead of on the next 1-second tick.

    The event is replaced by `notify()` itself, exactly once per firing — NOT by each waiting
    worker before it waits. An earlier version had every worker call a separate `reset()` right
    before `wait()`; since `reset()` unconditionally replaced `self.event`, the Nth worker to
    reach that line silently orphaned the first N-1 workers' event references (they were still
    holding the previous `self.event`, which `notify()` would never touch again). With W workers
    sharing one stage, that left only ~1/W of them ever reachable — exactly reproducing the
    "picking utilisation stuck near 1/W regardless of backlog" symptom this fix addresses.
    Workers must read `signal.event` fresh at each `yield` (never cache it across a loop
    iteration) so a re-check after a spurious wake waits on the current event, not a stale one.
    """

    __slots__ = ("env", "event")

    def __init__(self, env: simpy.Environment) -> None:
        self.env = env
        self.event = env.event()

    def notify(self) -> None:
        if not self.event.triggered:
            self.event.succeed()
            self.event = self.env.event()

    def wait(self):
        """Must be called fresh at each wait — never cache the returned event across a loop
        iteration, since `notify()` replaces `self.event` on every firing."""
        return self.event


@dataclass
class OrderRec:
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


class FullStageRLRunner:
    """
    RL-3: a single DQN agent decides at Picking, Packing, and Dispatch.

    Key design choices:
    - Separate urgent/normal queues at every stage so the agent can act there.
    - State vector has 16 features including stage_id (0.0/0.5/1.0) and 3 normalised
      capacity features (picking/packing/dispatch worker counts) — added per the RL
      generalisation audit (§12, data/api_runs/latest/rl3_audit_report.json), which found
      the agent could not distinguish a low-capacity regime from a high-capacity one and
      behaved inconsistently as a result.
    - Reward is RL-1 (rl1_current): deferred until dispatch completion.
    - Reward strategy: all buffer transitions for a given order (possibly from
      multiple stages) receive the same final reward. This is safe because the
      SLA outcome is fully determined at dispatch and each decision contributed
      to it. Stored as decision_indices: Dict[int, List[int]].
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
        self.reward_cfg = reward_cfg or {
            "w_urgent": 5.0,
            "w_normal": 3.0,
            "late_penalty_urgent": 2.0,
            "late_penalty_normal": 2.0,
            "lateness_cap_mult": 2.0,
        }
        # Capacity-feature normalisation reference — a fixed, documented training-range scale
        # (not per-episode max, and NOT the adaptive-search worker-limit config, which is an
        # unrelated search-bound parameter) so the same worker count always maps to the same
        # feature value across regimes, and dynamic candidates with >9 workers per stage remain
        # a valid (if compressed) feature rather than silently exceeding the intended range —
        # see configs/planning_profile.yaml::rl_generalisation.capacity_feature_scale (§21).
        self.max_workers_per_stage = float(
            load_planning_profile()["rl_generalisation"]["capacity_feature_scale"]
        )

    @staticmethod
    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, x)))

    def _slack_norm(self, now_ts: pd.Timestamp, r: OrderRec) -> float:
        system_min = (now_ts - r.arrival_time).total_seconds() / 60.0
        slack = float(r.sla_minutes - system_min)
        denom = max(1.0, float(r.sla_minutes))
        x = self._clip(slack / denom, -1.0, 1.0)
        return (x + 1.0) / 2.0

    def _head_slack(
        self, now_ts: pd.Timestamp, store: simpy.Store, recs: Dict
    ) -> float:
        if len(store.items) > 0:
            return self._slack_norm(now_ts, recs[store.items[0]])
        return 0.5

    def _state(
        self,
        now_s: int,
        now_ts: pd.Timestamp,
        pick_u: simpy.Store,
        pick_n: simpy.Store,
        pack_u: simpy.Store,
        pack_n: simpy.Store,
        disp_u: simpy.Store,
        disp_n: simpy.Store,
        wip_pick: int,
        wip_pack: int,
        wip_disp: int,
        recs: Dict,
        horizon_s: int,
        stage_id: float,
        curr_u: simpy.Store,
        curr_n: simpy.Store,
    ) -> np.ndarray:
        """
        16-feature state vector.
        Features 0-5: queue lengths (urgent/normal) at each stage, normalized.
        Features 6-8: WIP per stage (orders currently in service), normalized.
        Feature 9: time_norm.
        Features 10-11: slack of head urgent/normal order in the current stage.
        Feature 12: stage_id (0.0=picking, 0.5=packing, 1.0=dispatch).
        Features 13-15: normalised worker count at picking/packing/dispatch (this episode's
          regime) — lets the agent condition its choice on current capacity instead of
          treating every regime identically (§12).
        """
        time_norm = self._clip01(now_s / max(1, horizon_s))

        pu_n = self._clip01(len(pick_u.items) / 200.0)
        pn_n = self._clip01(len(pick_n.items) / 500.0)
        ku_n = self._clip01(len(pack_u.items) / 200.0)
        kn_n = self._clip01(len(pack_n.items) / 500.0)
        du_n = self._clip01(len(disp_u.items) / 200.0)
        dn_n = self._clip01(len(disp_n.items) / 500.0)

        wip_pick_n = self._clip01(wip_pick / 5.0)
        wip_pack_n = self._clip01(wip_pack / 5.0)
        wip_disp_n = self._clip01(wip_disp / 5.0)

        slack_u = self._head_slack(now_ts, curr_u, recs)
        slack_n = self._head_slack(now_ts, curr_n, recs)

        cap_pick = self._clip01(float(self.resources_cfg["picking_workers"]) / self.max_workers_per_stage)
        cap_pack = self._clip01(float(self.resources_cfg["packing_workers"]) / self.max_workers_per_stage)
        cap_disp = self._clip01(float(self.resources_cfg["dispatch_workers"]) / self.max_workers_per_stage)

        return np.array(
            [
                pu_n, pn_n,
                ku_n, kn_n,
                du_n, dn_n,
                wip_pick_n, wip_pack_n, wip_disp_n,
                time_norm,
                slack_u, slack_n,
                stage_id,
                cap_pick, cap_pack, cap_disp,
            ],
            dtype=np.float32,
        )

    def _reward(self, r: OrderRec) -> float:
        assert r.end_disp is not None
        system_min = (r.end_disp - r.arrival_time).total_seconds() / 60.0
        lateness = float(system_min - r.sla_minutes)
        is_urgent = r.order_type == "urgent"
        on_time = lateness <= 0.0

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
        service_time_map: Optional[Dict[Tuple[int, str], float]] = None,
        decision_log: Optional[List[Tuple[str, int]]] = None,
    ) -> Dict:
        """Run one RL-3 episode.

        If `service_time_map` is given (see service_time_map.py::build_service_time_map),
        stage service times are looked up from it instead of sampled inline from self.rng —
        this is what makes RL-3 comparable to FIFO/urgent_first under common random numbers.
        Training (main_train_rl3.py) intentionally omits it, so each training episode keeps
        its own independent stochastic service times.

        If `decision_log` is given (an empty list), every real decision point (both queues
        non-empty) appends (stage_name, action) to it in chronological order — used only by
        the RL-3 audit (src/rl/rl_audit.py) to compute streaks/diagnostics; never populated
        during training or routine evaluation.
        """
        missing = _REQUIRED_WORKLOAD_COLS - set(orders.columns)
        if missing:
            raise ValueError(
                f"Orders DataFrame is missing required workload columns: {sorted(missing)}. "
                "Regenerate with: python -m src.data.generate_orders_seasonal."
            )
        if "operating_horizon_minutes" not in self.sim_cfg:
            raise ValueError(
                "sim_cfg['operating_horizon_minutes'] is required — see "
                "operating_time.py::with_operating_horizon. The episode no longer runs until "
                "all orders complete; it runs for a finite monthly capacity horizon, and "
                "orders left unfinished at the end are backlog."
            )

        self.rng = np.random.default_rng(int(episode_seed))

        def _service_minutes(order_id: int, stage: str, units: float) -> float:
            if service_time_map is not None:
                return service_time_map[(order_id, stage)]
            return sample_service_minutes(self.rng, units, self.service_cfg[stage])

        orders = orders.sort_values("arrival_time").reset_index(drop=True)
        t0 = SIM_EPOCH

        def to_seconds(ts: pd.Timestamp) -> int:
            return int((ts - t0).total_seconds())

        def to_timestamp(sec: int) -> pd.Timestamp:
            return t0 + pd.to_timedelta(int(sec), unit="s")

        env = simpy.Environment()

        pick_u = simpy.Store(env)
        pick_n = simpy.Store(env)
        pack_u = simpy.Store(env)
        pack_n = simpy.Store(env)
        disp_u = simpy.Store(env)
        disp_n = simpy.Store(env)

        pick_signal = _StageSignal(env)
        pack_signal = _StageSignal(env)
        disp_signal = _StageSignal(env)

        recs: Dict[int, OrderRec] = {}
        total_orders = len(orders)
        completed = 0

        decision_indices: Dict[int, List[int]] = {}

        queue_trackers = {stage: QueueAreaTracker() for stage in ("picking", "packing", "dispatch")}

        horizon_minutes = float(self.sim_cfg["operating_horizon_minutes"])
        horizon_seconds = horizon_minutes * 60.0
        horizon_s = int(horizon_seconds)

        wip = {"pick": 0, "pack": 0, "disp": 0}

        sm = {
            "pick": {"dec_pts": 0, "dec_u": 0, "dec_n": 0},
            "pack": {"dec_pts": 0, "dec_u": 0, "dec_n": 0},
            "disp": {"dec_pts": 0, "dec_u": 0, "dec_n": 0},
        }

        def _build_state(stage_id: float, curr_u: simpy.Store, curr_n: simpy.Store,
                         now_s: int, now_ts: pd.Timestamp) -> np.ndarray:
            return self._state(
                now_s, now_ts,
                pick_u, pick_n, pack_u, pack_n, disp_u, disp_n,
                wip["pick"], wip["pack"], wip["disp"],
                recs, horizon_s, stage_id, curr_u, curr_n,
            )

        def _decide(stage_name: str, stage_id: float,
                    store_u: simpy.Store, store_n: simpy.Store,
                    now_s: int, now_ts: pd.Timestamp):
            q_u = len(store_u.items)
            q_n = len(store_n.items)
            s = _build_state(stage_id, store_u, store_n, now_s, now_ts)
            stage = sm[stage_name]

            if q_u > 0 and q_n > 0:
                stage["dec_pts"] += 1
                a = int(agent.act(s, greedy=greedy))
                should_record = not greedy
                if a == 0:
                    stage["dec_u"] += 1
                else:
                    stage["dec_n"] += 1
                if decision_log is not None:
                    decision_log.append((stage_name, a))
            elif q_u > 0:
                a, should_record = 0, False
            else:
                a, should_record = 1, False

            if a == 0 and q_u == 0:
                a = 1
            if a == 1 and q_n == 0:
                a = 0

            return a, s, should_record

        def _record_transition(oid: int, s_pre: np.ndarray, a: int,
                               stage_id: float, store_u: simpy.Store, store_n: simpy.Store,
                               now_s2: int, now_ts2: pd.Timestamp) -> None:
            s_post = _build_state(stage_id, store_u, store_n, now_s2, now_ts2)
            idx = buffer.add(Transition(state=s_pre, action=a, next_state=s_post, reward=0.0, done=False))
            if oid not in decision_indices:
                decision_indices[oid] = []
            decision_indices[oid].append(idx)

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
                    picking_units=float(row["picking_units"]),
                    packing_units=float(row["packing_units"]),
                    dispatch_units=float(row["dispatch_units"]),
                )
                queue_trackers["picking"].enqueue(env.now)
                if row["order_type"] == "urgent":
                    yield pick_u.put(oid)
                else:
                    yield pick_n.put(oid)
                pick_signal.notify()

        def picking_worker(wid: int):
            while True:
                while len(pick_u.items) == 0 and len(pick_n.items) == 0:
                    yield pick_signal.wait()

                now_s = int(env.now)
                now_ts = to_timestamp(now_s)
                a, s_pre, should_record = _decide("pick", STAGE_PICK, pick_u, pick_n, now_s, now_ts)

                oid = yield (pick_u.get() if a == 0 else pick_n.get())
                queue_trackers["picking"].dequeue(env.now)
                wip["pick"] += 1

                if should_record:
                    now_s2 = int(env.now)
                    _record_transition(oid, s_pre, a, STAGE_PICK, pick_u, pick_n, now_s2, to_timestamp(now_s2))

                r = recs[oid]
                r.start_pick = to_timestamp(int(env.now))
                st_min = _service_minutes(oid, "picking", r.picking_units)
                yield env.timeout(minutes_to_seconds_int(st_min))
                wip["pick"] -= 1
                r.end_pick = to_timestamp(int(env.now))

                queue_trackers["packing"].enqueue(env.now)
                if r.order_type == "urgent":
                    yield pack_u.put(oid)
                else:
                    yield pack_n.put(oid)
                pack_signal.notify()

        def packing_worker(wid: int):
            while True:
                while len(pack_u.items) == 0 and len(pack_n.items) == 0:
                    yield pack_signal.wait()

                now_s = int(env.now)
                now_ts = to_timestamp(now_s)
                a, s_pre, should_record = _decide("pack", STAGE_PACK, pack_u, pack_n, now_s, now_ts)

                oid = yield (pack_u.get() if a == 0 else pack_n.get())
                queue_trackers["packing"].dequeue(env.now)
                wip["pack"] += 1

                if should_record:
                    now_s2 = int(env.now)
                    _record_transition(oid, s_pre, a, STAGE_PACK, pack_u, pack_n, now_s2, to_timestamp(now_s2))

                r = recs[oid]
                r.start_pack = to_timestamp(int(env.now))
                st_min = _service_minutes(oid, "packing", r.packing_units)
                yield env.timeout(minutes_to_seconds_int(st_min))
                wip["pack"] -= 1
                r.end_pack = to_timestamp(int(env.now))

                queue_trackers["dispatch"].enqueue(env.now)
                if r.order_type == "urgent":
                    yield disp_u.put(oid)
                else:
                    yield disp_n.put(oid)
                disp_signal.notify()

        def dispatch_worker(wid: int):
            nonlocal completed
            while True:
                while len(disp_u.items) == 0 and len(disp_n.items) == 0:
                    yield disp_signal.wait()

                now_s = int(env.now)
                now_ts = to_timestamp(now_s)
                a, s_pre, should_record = _decide("disp", STAGE_DISP, disp_u, disp_n, now_s, now_ts)

                oid = yield (disp_u.get() if a == 0 else disp_n.get())
                queue_trackers["dispatch"].dequeue(env.now)
                wip["disp"] += 1

                if should_record:
                    now_s2 = int(env.now)
                    _record_transition(oid, s_pre, a, STAGE_DISP, disp_u, disp_n, now_s2, to_timestamp(now_s2))

                r = recs[oid]
                r.start_disp = to_timestamp(int(env.now))
                st_min = _service_minutes(oid, "dispatch", r.dispatch_units)
                yield env.timeout(minutes_to_seconds_int(st_min))
                wip["disp"] -= 1

                r.end_disp = to_timestamp(int(env.now))
                completed += 1

                reward_val = self._reward(r)
                for idx in decision_indices.get(oid, []):
                    buffer.set_reward(idx, reward_val)

        env.process(arrivals())
        for i in range(int(self.resources_cfg["picking_workers"])):
            env.process(picking_worker(i))
        for i in range(int(self.resources_cfg["packing_workers"])):
            env.process(packing_worker(i))
        for i in range(int(self.resources_cfg["dispatch_workers"])):
            env.process(dispatch_worker(i))

        # Run for exactly the finite operating horizon — never wait on done_event. Orders still
        # queued/in-service when the horizon ends are backlog, not a deadlock (spec §9).
        env.run(until=horizon_seconds)

        # Backlog transitions never reached dispatch completion, so _reward() was never called
        # for them and their buffered transitions still carry the placeholder reward=0.0 from
        # _record_transition — an uninformative signal that would teach the agent "leaving
        # orders unresolved at month end is neutral". Treat backlog as maximally late (same cap
        # used by _reward for a completed-but-very-late order) so training sees it as the SLA
        # failure it is.
        for oid, r in recs.items():
            if r.end_disp is not None:
                continue
            idxs = decision_indices.get(oid)
            if not idxs:
                continue
            is_urgent = r.order_type == "urgent"
            penalty = float(
                self.reward_cfg["late_penalty_urgent"] if is_urgent
                else self.reward_cfg["late_penalty_normal"]
            )
            for idx in idxs:
                buffer.set_reward(idx, -penalty)

        df = pd.DataFrame(
            {
                "order_id": [r.order_id for r in recs.values()],
                "order_type": [r.order_type for r in recs.values()],
                "arrival_time": [r.arrival_time for r in recs.values()],
                "start_pick": [r.start_pick for r in recs.values()],
                "end_pick": [r.end_pick for r in recs.values()],
                "start_pack": [r.start_pack for r in recs.values()],
                "end_pack": [r.end_pack for r in recs.values()],
                "start_disp": [r.start_disp for r in recs.values()],
                "end_disp": [r.end_disp for r in recs.values()],
                "sla_minutes": [r.sla_minutes for r in recs.values()],
            }
        )
        df["system_min"] = (df["end_disp"] - df["arrival_time"]).dt.total_seconds() / 60.0
        df["met_sla"] = df["system_min"] <= df["sla_minutes"]

        stage_metrics = compute_stage_metrics(
            df,
            workers={
                "picking":  int(self.resources_cfg["picking_workers"]),
                "packing":  int(self.resources_cfg["packing_workers"]),
                "dispatch": int(self.resources_cfg["dispatch_workers"]),
            },
            horizon_seconds=horizon_seconds,
            queue_trackers=queue_trackers,
        )

        sla_rate = float(df["met_sla"].mean())
        sla_urgent = float(df.loc[df["order_type"] == "urgent", "met_sla"].mean())
        sla_normal = float(df.loc[df["order_type"] != "urgent", "met_sla"].mean())

        unfinished_mask = df["end_disp"].isna()
        unfinished_n = int(unfinished_mask.sum())
        total_n = len(df)

        total_dec = sum(s["dec_pts"] for s in sm.values())
        total_dec_u = sum(s["dec_u"] for s in sm.values())
        p_urgent = (total_dec_u / total_dec) if total_dec > 0 else 0.0

        def _stage_rate(s: Dict) -> float:
            return (s["dec_u"] / s["dec_pts"]) if s["dec_pts"] > 0 else 0.0

        # Audit diagnostics — cheap vectorised pass, always computed (used by rl_audit.py).
        pick_wait = (df["start_pick"] - df["arrival_time"]).dt.total_seconds() / 60.0
        pack_wait = (df["start_pack"] - df["end_pick"]).dt.total_seconds() / 60.0
        disp_wait = (df["start_disp"] - df["end_pack"]).dt.total_seconds() / 60.0
        total_wait = pick_wait.fillna(0).clip(lower=0) + pack_wait.fillna(0).clip(lower=0) + disp_wait.fillna(0).clip(lower=0)
        urgent_wait = total_wait[df["order_type"] == "urgent"]
        normal_wait = total_wait[df["order_type"] != "urgent"]

        longest_urgent_streak = longest_normal_streak = None
        if decision_log:
            longest_urgent_streak = longest_normal_streak = 0
            cur_u = cur_n = 0
            for _stage_name, a in decision_log:
                if a == 0:
                    cur_u += 1; cur_n = 0
                    longest_urgent_streak = max(longest_urgent_streak, cur_u)
                else:
                    cur_n += 1; cur_u = 0
                    longest_normal_streak = max(longest_normal_streak, cur_n)

        return {
            "sla_rate": sla_rate,
            "sla_urgent": sla_urgent,
            "sla_normal": sla_normal,
            "mean_system_min": float(df["system_min"].mean()),
            "p90_system_min": float(df["system_min"].quantile(0.9)),
            "max_urgent_wait_min": float(urgent_wait.max()) if len(urgent_wait) else 0.0,
            "p95_urgent_wait_min": float(urgent_wait.quantile(0.95)) if len(urgent_wait) else 0.0,
            "max_normal_wait_min": float(normal_wait.max()) if len(normal_wait) else 0.0,
            "p95_normal_wait_min": float(normal_wait.quantile(0.95)) if len(normal_wait) else 0.0,
            "longest_urgent_streak": longest_urgent_streak,
            "longest_normal_streak": longest_normal_streak,
            "late_normal_orders": int((~df.loc[df["order_type"] != "urgent", "met_sla"]).sum()),
            "total_decisions": int(total_dec),
            "p_urgent_decisions": float(p_urgent),
            "pick_dec_pts": int(sm["pick"]["dec_pts"]),
            "pack_dec_pts": int(sm["pack"]["dec_pts"]),
            "disp_dec_pts": int(sm["disp"]["dec_pts"]),
            "pick_pct_urgent": float(_stage_rate(sm["pick"])),
            "pack_pct_urgent": float(_stage_rate(sm["pack"])),
            "disp_pct_urgent": float(_stage_rate(sm["disp"])),
            "stage_metrics": stage_metrics,
            "completed_orders": int(total_n - unfinished_n),
            "unfinished_orders": unfinished_n,
            "unfinished_urgent_orders": int((unfinished_mask & (df["order_type"] == "urgent")).sum()),
            "unfinished_normal_orders": int((unfinished_mask & (df["order_type"] != "urgent")).sum()),
            "backlog_share": float(unfinished_n / total_n) if total_n > 0 else 0.0,
        }
