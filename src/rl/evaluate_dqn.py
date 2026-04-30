# src/rl/evaluate_dqn.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.rl.dqn_agent import QNetwork
from src.rl.env_pick_rl import PickRLRunner
from src.rl.replay_buffer import ReplayBuffer
from src.simulation.multistage.sim_multistage import run_simulation_multistage

EPISODE_ORDERS = 10_000
EVAL_SEED = 123

REGIMES = [
    ("s111", {"picking_workers": 1, "packing_workers": 1, "dispatch_workers": 1}),
    ("s211", {"picking_workers": 2, "packing_workers": 1, "dispatch_workers": 1}),
    ("s221", {"picking_workers": 2, "packing_workers": 2, "dispatch_workers": 1}),
]


class _GreedyDQNAgent:
    def __init__(self, q_net: QNetwork, device: str):
        self.q = q_net
        self.device = device

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(s).squeeze(0).argmax().item())


def _run_baseline(
    orders: pd.DataFrame,
    policy: str,
    resources_cfg: dict,
    sim_cfg: dict,
    service_cfg: dict,
) -> dict:
    cfg = dict(sim_cfg)
    cfg["policy"] = policy
    _, summary = run_simulation_multistage(orders, cfg, resources_cfg, service_cfg)
    return summary


def _run_dqn(
    orders: pd.DataFrame,
    agent: _GreedyDQNAgent,
    resources_cfg: dict,
    sim_cfg: dict,
    service_cfg: dict,
    reward_cfg: dict,
) -> dict:
    runner = PickRLRunner(
        sim_cfg=sim_cfg,
        resources_cfg=resources_cfg,
        service_cfg=service_cfg,
        seed=EVAL_SEED,
        reward_cfg=reward_cfg,
    )
    buf = ReplayBuffer(capacity=1)
    return runner.run_episode(
        orders=orders,
        agent=agent,
        buffer=buf,
        episode_seed=EVAL_SEED,
        greedy=True,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    with open(root / "configs" / "sim_multistage.yaml", encoding="utf-8") as f:
        sim_cfg_full = yaml.safe_load(f)
    with open(root / "configs" / "rl.yaml", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)

    sim_cfg = sim_cfg_full["simulation"]
    base_resources = sim_cfg_full["resources"]
    service_cfg = sim_cfg_full["service_time"]
    reward_cfg = rl_cfg.get("reward", {})

    orders = (
        pd.read_csv(root / "data" / "orders_base.csv", parse_dates=["arrival_time"])
        .sort_values("arrival_time")
        .reset_index(drop=True)
        .iloc[:EPISODE_ORDERS]
        .copy()
    )

    ckpt_path = root / "data" / "dqn_final.pt"
    device = "cpu"
    input_dim = int(rl_cfg["network"].get("input_dim", 8))
    hidden_dim = int(rl_cfg["network"]["hidden_dim"])
    q_net = QNetwork(input_dim, hidden_dim, output_dim=2)
    q_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    q_net.eval()
    dqn_agent = _GreedyDQNAgent(q_net, device)

    print(f"Checkpoint loaded: {ckpt_path}")
    print(f"Orders: {len(orders):,} | Seed: {EVAL_SEED}\n")
    print(f"{'Regime':<8} {'Policy':<14} {'SLA':>6} {'SLA-U':>6} {'SLA-N':>6} "
          f"{'mean(min)':>10} {'p90(min)':>9} {'%U-dec':>7}")
    print("-" * 72)

    rows = []

    for regime_name, res_override in REGIMES:
        resources_cfg = {**base_resources, **res_override}

        for policy in ("fifo", "urgent_first"):
            m = _run_baseline(orders, policy, resources_cfg, sim_cfg, service_cfg)
            row = {
                "regime": regime_name,
                "policy": policy,
                "sla_rate": m["sla_rate"],
                "sla_urgent": m["sla_urgent"],
                "sla_normal": m["sla_normal"],
                "mean_system_min": m["mean_system_min"],
                "p90_system_min": m["p90_system_min"],
                "pct_urgent_decisions": float("nan"),
            }
            rows.append(row)
            print(
                f"{regime_name:<8} {policy:<14} {row['sla_rate']:6.4f} {row['sla_urgent']:6.4f} "
                f"{row['sla_normal']:6.4f} {row['mean_system_min']:10.1f} {row['p90_system_min']:9.1f}   {'—':>5}"
            )

        m = _run_dqn(orders, dqn_agent, resources_cfg, sim_cfg, service_cfg, reward_cfg)
        pct_u = m.get("p_urgent_decisions", float("nan"))
        row = {
            "regime": regime_name,
            "policy": "dqn",
            "sla_rate": m["sla_rate"],
            "sla_urgent": m["sla_urgent"],
            "sla_normal": m["sla_normal"],
            "mean_system_min": m.get("mean_system_min", float("nan")),
            "p90_system_min": m.get("p90_system_min", float("nan")),
            "pct_urgent_decisions": pct_u,
        }
        rows.append(row)
        pct_u_str = f"{pct_u:.2%}" if not np.isnan(pct_u) else "—"
        print(
            f"{regime_name:<8} {'dqn':<14} {row['sla_rate']:6.4f} {row['sla_urgent']:6.4f} "
            f"{row['sla_normal']:6.4f} {row['mean_system_min']:10.1f} {row['p90_system_min']:9.1f} "
            f"{pct_u_str:>7}"
        )
        print()

    df_out = pd.DataFrame(rows)
    out_path = root / "data" / "rl_eval_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
