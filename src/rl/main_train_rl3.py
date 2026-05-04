# src/rl/main_train_rl3.py
from __future__ import annotations

from pathlib import Path
import time
import csv
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import torch

from src.rl.replay_buffer import ReplayBuffer
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.rl.env_fullstage_rl import FullStageRLRunner


def _weighted_choice(items: List[dict], probs: List[float], rng: random.Random) -> dict:
    r = rng.random()
    acc = 0.0
    for it, p in zip(items, probs):
        acc += float(p)
        if r <= acc:
            return it
    return items[-1]


def _make_resources(base_resources: Dict, workers: Tuple[int, int, int]) -> Dict:
    r = dict(base_resources)
    r["picking_workers"] = int(workers[0])
    r["packing_workers"] = int(workers[1])
    r["dispatch_workers"] = int(workers[2])
    return r


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sim_cfg_path = root / "configs" / "sim_multistage.yaml"
    rl_cfg_path = root / "configs" / "rl3.yaml"
    orders_path = root / "data" / "orders_base.csv"
    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("RL-3 Training — single DQN, decisions at Picking + Packing + Dispatch")
    print(f"Config : {rl_cfg_path}")
    print("Starting...\n")

    with open(sim_cfg_path, "r", encoding="utf-8") as f:
        sim_cfg_full = yaml.safe_load(f)
    with open(rl_cfg_path, "r", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)

    simulation = sim_cfg_full["simulation"]
    base_resources = sim_cfg_full["resources"]
    service_time = sim_cfg_full["service_time"]

    base_seed = int(simulation.get("random_seed", 123))
    rng_np = np.random.default_rng(base_seed)
    rng_py = random.Random(2026)

    orders = pd.read_csv(orders_path, parse_dates=["arrival_time"])
    orders = orders.sort_values("arrival_time").reset_index(drop=True)

    episode_orders = int(rl_cfg["training"]["episode_orders"])
    episodes = int(rl_cfg["training"]["episodes"])
    updates_per_episode = int(rl_cfg["training"].get("updates_per_episode", 500))
    train_start_size = int(rl_cfg["training"]["train_start_size"])
    train_every_steps = int(rl_cfg["training"]["train_every_steps"])
    target_update_steps = int(rl_cfg["training"]["target_update_steps"])
    ckpt_every = int(rl_cfg["training"].get("ckpt_every", 10))

    agent_cfg = DQNConfig(
        input_dim=int(rl_cfg["network"].get("input_dim", 13)),
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

    mix = rl_cfg.get("scenario_mix", {})
    scenarios = mix.get("scenarios", []) if bool(mix.get("enabled", True)) else []
    if not scenarios:
        scenarios = [{"name": "default", "prob": 1.0, "workers": [1, 1, 1]}]
    probs = [float(s["prob"]) for s in scenarios]
    ssum = sum(probs) or 1.0
    probs = [p / ssum for p in probs]

    hist_path = out_dir / "rl3_train_history.csv"
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "episode", "scenario", "workers", "start_idx",
            "sla_rate", "sla_urgent", "sla_normal", "ma5_sla",
            "epsilon", "loss_mean", "buffer_size", "updates_done", "grad_step_total",
            "total_decisions", "p_urgent_overall",
            "pick_dec_pts", "pick_pct_urgent",
            "pack_dec_pts", "pack_pct_urgent",
            "disp_dec_pts", "disp_pct_urgent",
            "runtime_s",
        ])

    n_orders = len(orders)
    grad_step_total = 0
    last_slas: List[float] = []
    ma_window = 5

    for ep in range(1, episodes + 1):
        t0 = time.time()
        ep_seed = int(rng_np.integers(0, 1_000_000_000))

        scen = _weighted_choice(scenarios, probs, rng_py)
        workers = tuple(int(x) for x in scen["workers"])

        if n_orders <= episode_orders:
            start_idx = 0
            orders_ep = orders
        else:
            start_idx = int(rng_np.integers(0, n_orders - episode_orders))
            orders_ep = orders.iloc[start_idx : start_idx + episode_orders].copy()

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

        # curriculum: optionally skip gradient updates when episode is too "easy"
        dec_cap = int(scen.get("dec_cap", 999_999))
        curriculum_mode = str(rl_cfg.get("curriculum", {}).get("mode", "none"))
        do_train = not (curriculum_mode == "skip_train" and total_dec > dec_cap)

        updates_done = 0
        losses: List[float] = []
        if do_train and len(buffer) >= train_start_size:
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
            f"[EP {ep:03d}] scen={scen.get('name','s')} W={workers[0]}-{workers[1]}-{workers[2]} "
            f"SLA={sla:.4f} U={sla_u:.4f} N={sla_n:.4f} ma5={ma5:.4f} | "
            f"eps={eps:.3f} loss={loss_str} buf={len(buffer)} upd={updates_done} grad={grad_step_total} | "
            f"dec={total_dec} %U={p_urgent:.2f} "
            f"[pick:{pick_dec}/{pick_pu:.2f} pack:{pack_dec}/{pack_pu:.2f} disp:{disp_dec}/{disp_pu:.2f}] | "
            f"t={runtime:.1f}s"
        )

        with open(hist_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                ep,
                scen.get("name", "s"),
                f"{workers[0]}-{workers[1]}-{workers[2]}",
                start_idx,
                sla, sla_u, sla_n, ma5,
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


if __name__ == "__main__":
    main()