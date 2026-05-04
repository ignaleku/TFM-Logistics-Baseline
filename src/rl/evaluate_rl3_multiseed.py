# src/rl/evaluate_rl3_multiseed.py
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import yaml

from src.rl.dqn_agent import QNetwork
from src.rl.env_fullstage_rl import FullStageRLRunner
from src.rl.replay_buffer import ReplayBuffer
from src.simulation.multistage.sim_multistage import run_simulation_multistage

EPISODE_ORDERS = 10_000
N_WINDOWS = 5

REGIMES = [
    ("s111", {"picking_workers": 1, "packing_workers": 1, "dispatch_workers": 1}),
    ("s211", {"picking_workers": 2, "packing_workers": 1, "dispatch_workers": 1}),
    ("s221", {"picking_workers": 2, "packing_workers": 2, "dispatch_workers": 1}),
]


class _GreedyAgent:
    def __init__(self, q_net: QNetwork, device: str):
        self.q = q_net
        self.device = device

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(s).squeeze(0).argmax().item())


def _run_baseline(orders, policy, resources_cfg, sim_cfg, service_cfg):
    cfg = dict(sim_cfg)
    cfg["policy"] = policy
    _, summary = run_simulation_multistage(orders, cfg, resources_cfg, service_cfg)
    return summary


def _run_rl3(orders, agent, resources_cfg, sim_cfg, service_cfg, reward_cfg, seed):
    runner = FullStageRLRunner(
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


def _window_starts(n_orders: int, n_windows: int, episode_orders: int) -> list:
    max_start = n_orders - episode_orders
    if max_start <= 0:
        return [0] * n_windows
    step = max_start // (n_windows - 1) if n_windows > 1 else 0
    return [min(i * step, max_start) for i in range(n_windows)]


def _mean_std(series: pd.Series) -> str:
    v = series.dropna()
    if len(v) == 0:
        return f"{'n/a':>17}"
    return f"{v.mean():>7.4f} ± {v.std(ddof=0):>6.4f}"


def _print_summary(df: pd.DataFrame) -> None:
    sla_cols = ["total_sla", "urgent_sla", "normal_sla", "mean_system_time_min", "p90_system_time_min"]
    dec_cols = ["rl3_urgent_dec_overall", "rl3_urgent_dec_pick", "rl3_urgent_dec_pack", "rl3_urgent_dec_disp"]

    print("\n=== Aggregated results — mean ± std across windows ===\n")
    print(
        f"{'Regime':<8} {'Policy':<14} "
        f"{'total_sla':>17} {'urgent_sla':>17} {'normal_sla':>17} "
        f"{'mean_sys(min)':>17} {'p90_sys(min)':>17} "
        f"{'%U_overall':>17} {'%U_pick':>17} {'%U_pack':>17} {'%U_disp':>17}"
    )
    sep = "-" * 170
    print(sep)

    for (regime, policy), grp in df.groupby(["regime", "policy"]):
        sla_vals = "  ".join(_mean_std(grp[c]) for c in sla_cols)
        if policy == "rl3_dqn":
            dec_vals = "  ".join(_mean_std(grp[c]) for c in dec_cols)
        else:
            dec_vals = "  ".join(f"{'—':>17}" for _ in dec_cols)
        print(f"{regime:<8} {policy:<14} {sla_vals}  {dec_vals}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-window evaluation of RL-3 DQN vs FIFO and urgent_first"
    )
    parser.add_argument("--checkpoint", default="data/dqn_rl3_final.pt",
                        help="Path to RL-3 checkpoint (relative to repo root or absolute)")
    parser.add_argument("--output", default="data/rl3_eval_multiseed_results.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]

    with open(root / "configs" / "sim_multistage.yaml", encoding="utf-8") as f:
        sim_cfg_full = yaml.safe_load(f)
    with open(root / "configs" / "rl3.yaml", encoding="utf-8") as f:
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

    ckpt_path = root / args.checkpoint
    device = "cpu"
    input_dim = int(rl_cfg["network"].get("input_dim", 13))
    hidden_dim = int(rl_cfg["network"]["hidden_dim"])
    q_net = QNetwork(input_dim, hidden_dim, output_dim=2)
    q_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    q_net.eval()
    rl3_agent = _GreedyAgent(q_net, device)

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
                        "rl3_urgent_dec_overall": None,
                        "rl3_urgent_dec_pick": None,
                        "rl3_urgent_dec_pack": None,
                        "rl3_urgent_dec_disp": None,
                    }
                )
                done += 1
                print(
                    f"[{done:>3}/{total_runs}] {regime_name}  {policy:<14}  w{w_id}  "
                    f"SLA={m['sla_rate']:.4f}  U={m['sla_urgent']:.4f}  N={m['sla_normal']:.4f}"
                )

            seed = 42 + w_id
            m = _run_rl3(orders, rl3_agent, resources_cfg, sim_cfg, service_cfg, reward_cfg, seed)
            pu_overall = m.get("p_urgent_decisions")
            pu_pick = m.get("pick_pct_urgent")
            pu_pack = m.get("pack_pct_urgent")
            pu_disp = m.get("disp_pct_urgent")

            rows.append(
                {
                    "regime": regime_name,
                    "policy": "rl3_dqn",
                    "window_id": w_id,
                    "start_idx": start_idx,
                    "num_orders": len(orders),
                    "total_sla": m["sla_rate"],
                    "urgent_sla": m["sla_urgent"],
                    "normal_sla": m["sla_normal"],
                    "mean_system_time_min": m.get("mean_system_min"),
                    "p90_system_time_min": m.get("p90_system_min"),
                    "rl3_urgent_dec_overall": float(pu_overall) if pu_overall is not None else None,
                    "rl3_urgent_dec_pick": float(pu_pick) if pu_pick is not None else None,
                    "rl3_urgent_dec_pack": float(pu_pack) if pu_pack is not None else None,
                    "rl3_urgent_dec_disp": float(pu_disp) if pu_disp is not None else None,
                }
            )
            done += 1
            print(
                f"[{done:>3}/{total_runs}] {regime_name}  {'rl3_dqn':<14}  w{w_id}  "
                f"SLA={m['sla_rate']:.4f}  U={m['sla_urgent']:.4f}  N={m['sla_normal']:.4f}  "
                f"%U={pu_overall:.4f}"
                f" (pick:{pu_pick:.2f} pack:{pu_pack:.2f} disp:{pu_disp:.2f})"
            )

    df = pd.DataFrame(rows)
    out_path = root / args.output
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")

    _print_summary(df)


if __name__ == "__main__":
    main()