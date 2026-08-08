# src/rl/main_train_rl3.py
"""
RL-3 training under the corrected operating-time capacity model (spec §21).

The previous training regime sampled a random contiguous 10,000-order window from the
full-year CSV (spanning arbitrary, often multi-week, wall-clock periods) and a workforce
regime uniformly from the 12 static train_regimes (1-4 workers/stage, capacity feature
normalised by a scale of 6). That distribution no longer matches what the corrected
operating-time model actually produces: a fixed 9600-minute (160h) monthly horizon per
episode, real finite-capacity backlog, and dynamically generated workforce candidates that
routinely need 10+ workers per stage at peak demand (see data/rl3_dynamic_candidate_report.json
for the empirical evidence — the old checkpoint collapses to ~1% SLA on a 47-worker December
regime).

New training regime:
  - Three representative demand months (June = low, October = medium, December = peak;
    spec §21) — each month's real seasonal orders, compressed onto its own 9600-minute
    operating horizon (operating_time.py), exactly as Future/Historical evaluation does.
  - For each month, an analytical capacity estimate (capacity_estimate.py) plus a
    dynamically generated candidate set (candidate_generation.py) spanning
    under-capacity / near-capacity / over-capacity workforce — not just the tiny static grid.
  - A held-out slice of those candidates (per month) is never sampled during training, for
    exact-configuration generalisation testing (evaluate_rl3_generalisation.py-style).
  - Capacity features (cap_pick/cap_pack/cap_disp) are normalised against
    rl_generalisation.capacity_feature_scale (20), wide enough that dynamic candidates don't
    silently saturate the feature.

Usage:
    python -m src.rl.main_train_rl3
"""
from __future__ import annotations

from pathlib import Path
import time
import csv
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import torch

from src.rl.replay_buffer import ReplayBuffer
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.rl.env_fullstage_rl import FullStageRLRunner
from src.data.planning_profile import load_planning_profile
from src.analysis.capacity_estimate import estimate_workers
from src.analysis.candidate_generation import generate_worker_candidates
from src.analysis.regime_naming import format_regime
from src.simulation.multistage.operating_time import (
    operating_horizon_minutes, slice_month_operating_time, with_operating_horizon,
)

REPRESENTATIVE_MONTHS = ["June", "October", "December"]  # low / medium / peak — spec §21
CANDIDATES_PER_MONTH = 9
HOLDOUT_PER_MONTH = 2  # exact configurations held out from training, per month


def _make_resources(base_resources: Dict, workers: Tuple[int, int, int]) -> Dict:
    r = dict(base_resources)
    r["picking_workers"] = int(workers[0])
    r["packing_workers"] = int(workers[1])
    r["dispatch_workers"] = int(workers[2])
    return r


def build_training_pool(
    root: Path,
    orders_all: pd.DataFrame,
    service_cfg: Dict,
    hours_per_worker_month: float,
    target_utilisation: float,
) -> Tuple[Dict[int, pd.DataFrame], List[Tuple[int, Tuple[int, int, int]]], List[Tuple[int, Tuple[int, int, int]]], Dict]:
    """Returns (month_orders_by_num, train_pool, holdout_pool, report) — see module docstring."""
    profile = load_planning_profile()
    name_to_num = {v["name"]: k for k, v in profile["months"].items()}
    horizon_minutes = operating_horizon_minutes(hours_per_worker_month)

    month_orders: Dict[int, pd.DataFrame] = {}
    train_pool: List[Tuple[int, Tuple[int, int, int]]] = []
    holdout_pool: List[Tuple[int, Tuple[int, int, int]]] = []
    report: Dict = {"months": {}}

    for name in REPRESENTATIVE_MONTHS:
        m = name_to_num[name]
        mo = slice_month_operating_time(orders_all, m, horizon_minutes)
        month_orders[m] = mo

        estimate = estimate_workers(mo, service_cfg, hours_per_worker_month, target_utilisation)
        centre = tuple(int(estimate["workers"][s]) for s in ("picking", "packing", "dispatch"))
        candidates = generate_worker_candidates(centre, candidate_count=CANDIDATES_PER_MONTH)

        holdout = candidates[-HOLDOUT_PER_MONTH:] if len(candidates) > HOLDOUT_PER_MONTH else []
        train = candidates[: len(candidates) - len(holdout)]

        for w in train:
            train_pool.append((m, w))
        for w in holdout:
            holdout_pool.append((m, w))

        report["months"][name] = {
            "month_num": m,
            "num_orders": int(len(mo)),
            "analytical_centre": centre,
            "train_regimes": [format_regime(*w) for w in train],
            "holdout_regimes": [format_regime(*w) for w in holdout],
        }

    return month_orders, train_pool, holdout_pool, report


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sim_cfg_path = root / "configs" / "sim_multistage.yaml"
    rl_cfg_path = root / "configs" / "rl3.yaml"
    orders_path = root / "data" / "orders_base_seasonal.csv"
    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("RL-3 Training — operating-time model, dynamic-candidate representative months")
    print(f"Config : {rl_cfg_path}")
    print("Starting...\n")

    with open(sim_cfg_path, "r", encoding="utf-8") as f:
        sim_cfg_full = yaml.safe_load(f)
    with open(rl_cfg_path, "r", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)

    profile = load_planning_profile()
    hours_per_worker_month = float(profile["cost_defaults"]["hours_per_worker_month"])
    target_utilisation = float(profile["capacity_planning"]["target_utilisation"])

    simulation = with_operating_horizon(sim_cfg_full["simulation"], hours_per_worker_month)
    base_resources = sim_cfg_full["resources"]
    service_time = sim_cfg_full["service_time"]

    base_seed = int(simulation.get("random_seed", 123))
    rng_np = np.random.default_rng(base_seed)
    rng_py = random.Random(2026)

    orders_all = pd.read_csv(orders_path, parse_dates=["arrival_time"])
    if "month" not in orders_all.columns:
        orders_all["month"] = orders_all["arrival_time"].dt.month

    month_orders, train_pool, holdout_pool, pool_report = build_training_pool(
        root, orders_all, service_time, hours_per_worker_month, target_utilisation,
    )
    pool_path = out_dir / "rl3_train_pool.json"
    pool_path.write_text(json.dumps(pool_report, indent=2), encoding="utf-8")
    print("Training pool (spec §21):")
    for name, info in pool_report["months"].items():
        print(f"  {name:<10} orders={info['num_orders']:>6}  centre={info['analytical_centre']}  "
              f"train={info['train_regimes']}  holdout={info['holdout_regimes']}")
    print(f"  Total train (month,regime) pairs: {len(train_pool)}  |  holdout: {len(holdout_pool)}")
    print(f"  Saved: {pool_path}\n")

    episodes = int(rl_cfg["training"]["episodes"])
    updates_per_episode = int(rl_cfg["training"].get("updates_per_episode", 500))
    train_start_size = int(rl_cfg["training"]["train_start_size"])
    train_every_steps = int(rl_cfg["training"]["train_every_steps"])
    target_update_steps = int(rl_cfg["training"]["target_update_steps"])
    ckpt_every = int(rl_cfg["training"].get("ckpt_every", 10))

    agent_cfg = DQNConfig(
        input_dim=int(rl_cfg["network"].get("input_dim", 16)),
        hidden_dim=int(rl_cfg["network"]["hidden_dim"]),
        lr=float(rl_cfg["training"]["lr"]),
        gamma=float(rl_cfg["training"]["gamma"]),
        batch_size=int(rl_cfg["training"]["batch_size"]),
        target_update_steps=target_update_steps,
        train_start_size=train_start_size,
        train_every_steps=train_every_steps,
        eps_start=float(rl_cfg["epsilon"]["start"]),
        eps_end=float(rl_cfg["epsilon"]["end"]),
        eps_decay_steps=int(rl_cfg["epsilon"]["decay_steps"]),
    )
    agent = DQNAgent(agent_cfg, seed=base_seed)

    cap = int(rl_cfg["buffer"]["capacity"])
    buffer = ReplayBuffer(capacity=cap)

    hist_path = out_dir / "rl3_train_history.csv"
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "episode", "month", "regime", "num_orders",
            "sla_rate", "sla_urgent", "sla_normal", "ma5_sla",
            "unfinished_orders", "backlog_share",
            "epsilon", "loss_mean", "buffer_size", "updates_done", "grad_step_total",
            "total_decisions", "p_urgent_overall",
            "pick_dec_pts", "pick_pct_urgent",
            "pack_dec_pts", "pack_pct_urgent",
            "disp_dec_pts", "disp_pct_urgent",
            "runtime_s",
        ])

    grad_step_total = 0
    last_slas: List[float] = []
    ma_window = 5

    for ep in range(1, episodes + 1):
        t0 = time.time()
        ep_seed = int(rng_np.integers(0, 1_000_000_000))

        month_num, workers = train_pool[rng_py.randrange(len(train_pool))]
        orders_ep = month_orders[month_num]
        regime_label = format_regime(*workers)

        resources_ep = _make_resources(base_resources, workers)
        runner = FullStageRLRunner(
            sim_cfg=simulation,
            resources_cfg=resources_ep,
            service_cfg=service_time,
            seed=base_seed,
            reward_cfg=rl_cfg.get("reward", None),
        )

        metrics = runner.run_episode(
            orders=orders_ep,
            agent=agent,
            buffer=buffer,
            episode_seed=ep_seed,
            greedy=False,
        )

        sla = float(metrics["sla_rate"])
        sla_u = float(metrics["sla_urgent"])
        sla_n = float(metrics["sla_normal"])
        total_dec = int(metrics.get("total_decisions", 0))
        p_urgent = float(metrics.get("p_urgent_decisions", 0.0))
        pick_dec = int(metrics.get("pick_dec_pts", 0))
        pick_pu = float(metrics.get("pick_pct_urgent", 0.0))
        pack_dec = int(metrics.get("pack_dec_pts", 0))
        pack_pu = float(metrics.get("pack_pct_urgent", 0.0))
        disp_dec = int(metrics.get("disp_dec_pts", 0))
        disp_pu = float(metrics.get("disp_pct_urgent", 0.0))
        unfinished = int(metrics.get("unfinished_orders", 0))
        backlog_share = float(metrics.get("backlog_share", 0.0))

        updates_done = 0
        losses: List[float] = []
        if len(buffer) >= train_start_size:
            for _ in range(updates_per_episode):
                if grad_step_total % train_every_steps == 0:
                    batch = buffer.sample(agent.cfg.batch_size, agent.rng)
                    loss = agent.train_step(batch, grad_step_total)
                    if loss is not None:
                        losses.append(float(loss))
                    updates_done += 1
                grad_step_total += 1
                if grad_step_total % target_update_steps == 0:
                    agent.update_target()

        loss_mean = (sum(losses) / len(losses)) if losses else None
        eps = float(agent.epsilon())
        runtime = time.time() - t0

        last_slas.append(sla)
        last_slas = last_slas[-ma_window:]
        ma5 = float(sum(last_slas) / len(last_slas))

        loss_str = "NA" if loss_mean is None else f"{loss_mean:.4f}"
        print(
            f"[EP {ep:03d}] month={month_num:>2} regime={regime_label:<10} n={len(orders_ep):>6} "
            f"SLA={sla:.4f} U={sla_u:.4f} N={sla_n:.4f} ma5={ma5:.4f} backlog={backlog_share:.3f} | "
            f"eps={eps:.3f} loss={loss_str} buf={len(buffer)} upd={updates_done} grad={grad_step_total} | "
            f"dec={total_dec} %U={p_urgent:.2f} "
            f"[pick:{pick_dec}/{pick_pu:.2f} pack:{pack_dec}/{pack_pu:.2f} disp:{disp_dec}/{disp_pu:.2f}] | "
            f"t={runtime:.1f}s"
        )

        with open(hist_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                ep, month_num, regime_label, len(orders_ep),
                sla, sla_u, sla_n, ma5,
                unfinished, backlog_share,
                eps,
                loss_mean if loss_mean is not None else "",
                len(buffer), updates_done, grad_step_total,
                total_dec, p_urgent,
                pick_dec, pick_pu,
                pack_dec, pack_pu,
                disp_dec, disp_pu,
                round(runtime, 2),
            ])

        if ep % ckpt_every == 0:
            ckpt_path = out_dir / f"dqn_rl3_ckpt_ep{ep:03d}.pt"
            torch.save(agent.q.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    final_path = out_dir / "dqn_rl3_final.pt"
    torch.save(agent.q.state_dict(), final_path)
    print(f"\nTraining complete.")
    print(f"Model   : {final_path}")
    print(f"History : {hist_path}")
    print(f"Pool    : {pool_path}")


if __name__ == "__main__":
    main()
