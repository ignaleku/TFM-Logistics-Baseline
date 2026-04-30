from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from src.rl.replay_buffer import ReplayBuffer
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.rl.env_pick_rl import PickRLRunner


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    # --- Paths
    cfg_path = root / "configs" / "sim_multistage.yaml"
    orders_path = root / "data" / "orders_base.csv"
    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🤖 RL-1 — DQN v1 (picking urgent vs normal)")

    # --- Load cfg
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg["simulation"]
    resources_cfg = cfg["resources"]
    service_cfg = cfg["service_time"]

    # OJO: en RL ignoramos sim_cfg["policy"], la política la define el agente.
    # Pero usamos random_seed.
    base_seed = int(sim_cfg.get("random_seed", 123))

    # --- Load orders
    orders = pd.read_csv(orders_path, parse_dates=["arrival_time"])
    orders = orders.sort_values("arrival_time").reset_index(drop=True)

    # Para entrenamiento: subset (iteración rápida)
    EPISODE_ORDERS = 10_000
    orders_ep = orders.head(EPISODE_ORDERS).copy()

    # --- RL components
    buffer = ReplayBuffer(capacity=200_000)

    agent_cfg = DQNConfig(
        input_dim=5,
        hidden_dim=64,
        lr=1e-3,
        gamma=0.99,
        batch_size=256,
        target_update_steps=2000,
        train_start_size=3000,
        train_every_steps=4,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=50_000,
    )
    agent = DQNAgent(agent_cfg, seed=base_seed)

    runner = PickRLRunner(
        sim_cfg=sim_cfg,
        resources_cfg=resources_cfg,
        service_cfg=service_cfg,
        seed=base_seed,
    )

    # --- Training loop
    EPISODES = 30
    rng = np.random.default_rng(base_seed)
    global_step = 0

    for ep in range(1, EPISODES + 1):
        ep_seed = int(rng.integers(0, 1_000_000_000))
        t_start = time.time()

        metrics = runner.run_episode(
            orders=orders_ep,
            agent=agent,
            buffer=buffer,
            episode_seed=ep_seed,
        )

        # Entrenamiento: varios updates por episodio (simple)
        losses = []
        if len(buffer) >= agent_cfg.train_start_size:
            # número de mini-updates por episodio (ajustable)
            updates = 500
            for _ in range(updates):
                global_step += 1
                if global_step % agent_cfg.train_every_steps != 0:
                    continue
                batch = buffer.sample(agent_cfg.batch_size, rng=rng)
                loss = agent.train_step(batch, global_step=global_step)
                losses.append(loss)

        dt = time.time() - t_start
        loss_mean = float(np.mean(losses)) if losses else float("nan")

        print(
            f"[ep {ep:03d}] "
            f"sla={metrics['sla_rate']:.4f} "
            f"u={metrics['sla_urgent']:.4f} "
            f"n={metrics['sla_normal']:.4f} "
            f"mean_sys={metrics['mean_system_min']:.1f} "
            f"p90={metrics['p90_system_min']:.1f} "
            f"buffer={len(buffer)} "
            f"eps={agent.epsilon():.3f} "
            f"loss={loss_mean:.4f} "
            f"time={dt:.1f}s"
        )

        # Guardar checkpoint cada X episodios
        if ep % 10 == 0:
            ckpt_path = out_dir / f"dqn_ckpt_ep{ep:03d}.pt"
            agent.save(str(ckpt_path))
            print(f"💾 guardado: {ckpt_path}")

    # Guardado final
    final_path = out_dir / "dqn_final.pt"
    agent.save(str(final_path))
    print(f"✅ Entrenamiento terminado. Modelo: {final_path}")


if __name__ == "__main__":
    main()