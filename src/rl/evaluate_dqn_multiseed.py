# src/rl/evaluate_dqn_multiseed.py
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
N_WINDOWS = 5

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
    seed: int,
) -> dict:
    runner = PickRLRunner(
        sim_cfg=sim_cfg,
        resources_cfg=resources_cfg,
        service_cfg=service_cfg,
        seed=seed,
        reward_cfg=reward_cfg,
    )
    buf = ReplayBuffer(capacity=1)
    return runner.run_episode(
        orders=orders, agent=agent, buffer=buf, episode_seed=seed, greedy=True
    )


def _window_starts(n_orders: int, n_windows: int, episode_orders: int) -> list[int]:
    max_start = n_orders - episode_orders
    if max_start <= 0:
        return [0] * n_windows
    step = max_start // (n_windows - 1) if n_windows > 1 else 0
    return [min(i * step, max_start) for i in range(n_windows)]


def _print_summary(df: pd.DataFrame) -> None:
    metrics = [
        "total_sla",
        "urgent_sla",
        "normal_sla",
        "mean_system_time_min",
        "p90_system_time_min",
    ]
    print("\n=== Aggregated results — mean ± std across windows ===\n")
    print(
        f"{'Regime':<8} {'Policy':<14} {'total_sla':>17} {'urgent_sla':>17} "
        f"{'normal_sla':>17} {'mean_sys(min)':>17} {'p90_sys(min)':>17} {'%U_dec':>14}"
    )
    print("-" * 110)
    for (regime, policy), grp in df.groupby(["regime", "policy"]):
        vals = [
            f"{grp[m].mean():>7.4f} ± {grp[m].std():>6.4f}" for m in metrics
        ]
        if policy == "dqn":
            pct_u = grp["dqn_urgent_decision_rate"].dropna()
            u_str = f"{pct_u.mean():>5.4f} ± {pct_u.std():>5.4f}"
        else:
            u_str = f"{'—':>13}"
        print(f"{regime:<8} {policy:<14} {'  '.join(vals)}  {u_str}")
    print()


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

    orders_all = (
        pd.read_csv(root / "data" / "orders_base.csv", parse_dates=["arrival_time"])
        .sort_values("arrival_time")
        .reset_index(drop=True)
    )

    ckpt_path = root / "data" / "dqn_final.pt"
    device = "cpu"
    q_net = QNetwork(
        input_dim=int(rl_cfg["network"].get("input_dim", 8)),
        hidden_dim=int(rl_cfg["network"]["hidden_dim"]),
        output_dim=2,
    )
    q_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    q_net.eval()
    dqn_agent = _GreedyDQNAgent(q_net, device)

    n_orders = len(orders_all)
    starts = _window_starts(n_orders, N_WINDOWS, EPISODE_ORDERS)
    total_runs = len(REGIMES) * 3 * N_WINDOWS  # 3 policies

    print(f"Orders available : {n_orders:,}")
    print(f"Orders per window: {EPISODE_ORDERS:,}")
    print(f"Windows ({N_WINDOWS})       : start indices {starts}")
    print(f"Regimes          : {[r for r, _ in REGIMES]}")
    print(f"Checkpoint       : {ckpt_path}\n")

    rows = []
    done = 0

    for regime_name, res_override in REGIMES:
        resources_cfg = {**base_resources, **res_override}

        for w_id, start_idx in enumerate(starts):
            orders = orders_all.iloc[start_idx : start_idx + EPISODE_ORDERS].copy()

            for policy in ("fifo", "urgent_first"):
                m = _run_baseline(orders, policy, resources_cfg, sim_cfg, service_cfg)
                rows.append(
                    {
                        "regime": regime_name,
                        "policy": policy,
                        "window_id": w_id,
                        "start_idx": start_idx,
                        "num_orders": len(orders),
                        "total_sla": m["sla_rate"],
                        "urgent_sla": m["sla_urgent"],
                        "normal_sla": m["sla_normal"],
                        "mean_system_time_min": m["mean_system_min"],
                        "p90_system_time_min": m["p90_system_min"],
                        "dqn_urgent_decision_rate": None,
                    }
                )
                done += 1
                print(
                    f"[{done:>3}/{total_runs}] {regime_name}  {policy:<14}  w{w_id}  "
                    f"SLA={m['sla_rate']:.4f}  U={m['sla_urgent']:.4f}  N={m['sla_normal']:.4f}"
                )

            seed = 42 + w_id
            m = _run_dqn(
                orders, dqn_agent, resources_cfg, sim_cfg, service_cfg, reward_cfg, seed
            )
            pct_u = m.get("p_urgent_decisions")
            rows.append(
                {
                    "regime": regime_name,
                    "policy": "dqn",
                    "window_id": w_id,
                    "start_idx": start_idx,
                    "num_orders": len(orders),
                    "total_sla": m["sla_rate"],
                    "urgent_sla": m["sla_urgent"],
                    "normal_sla": m["sla_normal"],
                    "mean_system_time_min": m.get("mean_system_min"),
                    "p90_system_time_min": m.get("p90_system_min"),
                    "dqn_urgent_decision_rate": float(pct_u) if pct_u is not None else None,
                }
            )
            done += 1
            pct_u_str = f"{pct_u:.4f}" if pct_u is not None else "n/a"
            print(
                f"[{done:>3}/{total_runs}] {regime_name}  {'dqn':<14}  w{w_id}  "
                f"SLA={m['sla_rate']:.4f}  U={m['sla_urgent']:.4f}  N={m['sla_normal']:.4f}  "
                f"%U={pct_u_str}"
            )

    df = pd.DataFrame(rows)
    out_path = root / "data" / "rl_eval_multiseed_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")

    _print_summary(df)


if __name__ == "__main__":
    main()
